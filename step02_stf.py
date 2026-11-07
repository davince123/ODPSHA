"""Step 2. Build magnitude-conditioned stochastic STFs, or read original STFs."""

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from step01_scenarios import (check_dt, h5_output, identifiers, magnitude_eligible, numbers,
                              read_scenarios, run_cli, strings, table, write_csv)


KEY = ["event_id", "mw"]


def import_stfs(root, event_id, mw, dt):
    folder = root / event_id / str(mw)
    paths = sorted(folder.glob("*/STF0.txt"), key=lambda path: path.parent.name)
    records = []
    for path in paths:
        with path.open(encoding="utf-8-sig") as handle:
            fc = float(handle.readline().split()[1])
            handle.readline()
            file_dt = float(handle.readline().split()[1])
            if not np.isclose(file_dt, dt, atol=1e-10, rtol=0):
                raise ValueError(f"{path}: dt differs from the project dt")
            series = np.loadtxt(handle, ndmin=1)
        if (not np.isfinite(fc) or fc <= 0 or series.ndim != 1 or series.size == 0
                or not np.isfinite(series).all() or not np.any(series)):
            raise ValueError(f"{path}: invalid source time function")
        records.append((path.parent.name, fc, 1.0, series))
    return records


def read_fc_samples(project, scenarios):
    if "legacy_res_point_csv" in project.config:
        matrix = np.loadtxt(project.path("legacy_res_point_csv"), delimiter=",", ndmin=2)
        magnitudes = project.option("stf").get("res_point_magnitudes", sorted(scenarios.mw.unique()))
        if matrix.shape[1] != len(magnitudes):
            raise ValueError("res_point.csv: one column is required per configured magnitude")
        samples = pd.DataFrame(
            [(mw, str(index), fc, 1.0) for mw, column in zip(magnitudes, matrix.T)
             for index, fc in enumerate(column)],
            columns=["mw", "sample_id", "fc_hz", "weight"],
        )
    else:
        samples = table(project.path("fc_samples_csv"), ["mw", "sample_id", "fc_hz"],
                        ["sample_id"], ["mw", "sample_id"])
        if "weight" not in samples:
            samples["weight"] = 1.0
    identifiers(samples, "sample_id")
    numbers(samples, "mw")
    samples["mw"] = samples.mw.round(8)
    numbers(samples, "fc_hz", "weight", positive=True)
    if samples.duplicated(["mw", "sample_id"]).any():
        raise ValueError("Duplicate fc sample identifiers at the same magnitude")
    eligible = magnitude_eligible(scenarios.mw, scenarios.egf_mw)
    missing = set(scenarios.loc[eligible, "mw"]) - set(samples.mw)
    if missing:
        raise ValueError(f"No fc samples for magnitudes {sorted(missing)}")
    return {mw: rows for mw, rows in samples.groupby("mw")}


def generate_stfs(row, samples, settings, dt):
    mode = settings.get("egf_fc_model", "regression")
    if mode not in ("regression", "measured"):
        raise ValueError("stf.egf_fc_model must be regression or measured")
    synthesis_fc = row.egf_fc_hz
    if mode == "regression":
        synthesis_fc = np.exp(-0.923669 * row.egf_mw + 4.292291)
    records = []
    repetitions = int(settings.get("delay_repetitions", 1))
    if repetitions < 1:
        raise ValueError("stf.delay_repetitions must be at least 1")
    for sample in samples.itertuples(index=False):
        if sample.fc_hz >= row.egf_fc_hz or sample.fc_hz > synthesis_fc:
            continue
        for repetition in range(repetitions):
            identity = f"{settings.get('seed', 2026)}/{row.event_id}/{row.mw}/{sample.sample_id}/{repetition}"
            seed = int.from_bytes(hashlib.blake2s(identity.encode(), digest_size=4).digest(), "little")
            series = synthesize(row.mw, row.egf_mw, sample.fc_hz, synthesis_fc, dt, seed)
            if not np.any(series):
                raise ValueError(f"{identity}: STF is zero after the original six-decimal rounding")
            records.append((f"{sample.sample_id}-{repetition}", sample.fc_hz, sample.weight, series))
    return records


def run(project):
    scenarios = read_scenarios(project)
    eligible = scenarios[magnitude_eligible(scenarios.mw, scenarios.egf_mw)]
    settings = project.option("stf")
    dt = check_dt(project.config["dt_s"])
    import_root = project.path("legacy_stf_dir") if "legacy_stf_dir" in project.config else None
    samples = None if import_root or eligible.empty else read_fc_samples(project, scenarios)
    output = project.output / "02_stf"
    manifest = []
    for row in eligible.itertuples(index=False):
        if import_root:
            records = import_stfs(import_root, row.event_id, row.mw, dt)
        else:
            records = generate_stfs(row, samples[row.mw], settings, dt)
        if not records:
            continue
        sample_ids, frequencies, weights, series = zip(*records)
        path = Path(row.event_id) / f"{row.mw}.h5"
        with h5_output(output / path) as archive:
            archive.attrs["dt_s"] = dt
            strings(archive, "sample_id", sample_ids)
            archive.create_dataset("fc_hz", data=frequencies)
            archive.create_dataset("weight", data=np.asarray(weights) / np.sum(weights))
            pulses = archive.create_group("stf")
            for index, pulse in enumerate(series):
                pulses.create_dataset(str(index), data=pulse)
        manifest.append((row.event_id, row.mw, path.as_posix(), len(records)))
    write_csv(pd.DataFrame(manifest, columns=[*KEY, "file", "n_stf"]), output / "manifest.csv")
    print(f"Step 2: STFs for {len(manifest)}/{len(scenarios)} scenarios")


# Numerical kernel adapted from the original STF_cal_2.py.
TC = 6.1425


def calf(Fnum, Fden):
    return 2.0 / (1.0 + (Fden / Fnum) ** 2)


@njit(cache=True, fastmath=True)
def _rho_core(tb, Fnum, Fden, alpha):
    t_abs = abs(tb)
    t_used = t_abs
    if t_used > 1.0 / Fden:
        n1 = 3
        nd0 = 10000
        t_used = 1.0 / Fden
    else:
        nd0 = int(np.ceil(100.0 / t_used))
        if nd0 < 10000:
            nd0 = 10000
            n1 = 3
        elif nd0 < 10000000:
            n1 = 2
            rem = nd0 % 4
            if rem != 0:
                nd0 += 4 - rem
        else:
            nd0 = 10000000
            n1 = 1
            if t_used < 1e-7:
                t_used = 1e-7

    lsup = n1 / t_used
    n2 = nd0 / n1
    df = 1.0 / (t_used * n2)
    nd = nd0 + 1
    step = lsup / nd0
    total = 0.0

    for i in range(0, nd - 1, 4):
        f0 = i * step
        f1 = (i + 1) * step
        f2 = (i + 2) * step
        f3 = (i + 3) * step
        f4 = (i + 4) * step

        y0 = np.sqrt(1.0 + alpha * (f0 / Fnum) ** 2) * np.cos((2.0 * np.pi * t_used) * f0) / (1.0 + (f0 / Fden) ** 2)
        y1 = np.sqrt(1.0 + alpha * (f1 / Fnum) ** 2) * np.cos((2.0 * np.pi * t_used) * f1) / (1.0 + (f1 / Fden) ** 2)
        y2 = np.sqrt(1.0 + alpha * (f2 / Fnum) ** 2) * np.cos((2.0 * np.pi * t_used) * f2) / (1.0 + (f2 / Fden) ** 2)
        y3 = np.sqrt(1.0 + alpha * (f3 / Fnum) ** 2) * np.cos((2.0 * np.pi * t_used) * f3) / (1.0 + (f3 / Fden) ** 2)
        y4 = np.sqrt(1.0 + alpha * (f4 / Fnum) ** 2) * np.cos((2.0 * np.pi * t_used) * f4) / (1.0 + (f4 / Fden) ** 2)
        total += 7.0 * (y0 + y4) + 32.0 * (y1 + y3) + y2

    ans = 4.0 * df * total / 45.0
    if t_abs > 1.0 / Fden:
        rhodb = np.pi * Fden * np.exp(-2.0 * np.pi * Fden * t_used)
        fcc = ans / rhodb
        return fcc * (0.99 * t_used / t_abs + 0.01) * np.pi * Fden * np.exp(-2.0 * np.pi * Fden * t_abs)
    return ans


@njit(cache=True, fastmath=True)
def _rho_masa_core(t_values, Fnum, Fden, alpha):
    res = np.empty(t_values.size, dtype=np.float64)
    for n in range(t_values.size):
        t = abs(t_values[n])
        nd0 = int(np.ceil(100.0 / t))
        if nd0 < 10000:
            nd0 = 10000
            n1 = 3
        elif nd0 < 10000000:
            n1 = 2
            rem = nd0 % 4
            if rem != 0:
                nd0 += 4 - rem
        else:
            nd0 = 10000000
            n1 = 1
            if t < 1e-7:
                t = 1e-7

        lsup = n1 / t
        n2 = nd0 / n1
        df = 1.0 / (t * n2)
        step = lsup / nd0
        total = 0.0

        for i in range(0, nd0, 4):
            f0 = i * step
            f1 = (i + 1) * step
            f2 = (i + 2) * step
            f3 = (i + 3) * step
            f4 = (i + 4) * step

            y0 = np.sqrt(1.0 + alpha * (f0 / Fnum) ** 2) * np.cos((2.0 * np.pi * t) * f0) / (1.0 + (f0 / Fden) ** 2)
            y1 = np.sqrt(1.0 + alpha * (f1 / Fnum) ** 2) * np.cos((2.0 * np.pi * t) * f1) / (1.0 + (f1 / Fden) ** 2)
            y2 = np.sqrt(1.0 + alpha * (f2 / Fnum) ** 2) * np.cos((2.0 * np.pi * t) * f2) / (1.0 + (f2 / Fden) ** 2)
            y3 = np.sqrt(1.0 + alpha * (f3 / Fnum) ** 2) * np.cos((2.0 * np.pi * t) * f3) / (1.0 + (f3 / Fden) ** 2)
            y4 = np.sqrt(1.0 + alpha * (f4 / Fnum) ** 2) * np.cos((2.0 * np.pi * t) * f4) / (1.0 + (f4 / Fden) ** 2)
            total += 7.0 * (y0 + y4) + 32.0 * (y1 + y3) + 12.0 * y2

        res[n] = 4.0 * df * total / 45.0
    return res


@njit(cache=True, fastmath=True)
def _interp_rho(tb, Fden, t_grid, rho_grid):
    if tb < t_grid[0]:
        return -1.0
    n = t_grid.size
    if tb > t_grid[n - 1]:
        return rho_grid[n - 1] * np.exp(-2.0 * np.pi * Fden * tb) / np.exp(-2.0 * np.pi) * (0.99 / (Fden * tb) + 0.01)

    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if t_grid[mid] <= tb:
            lo = mid
        else:
            hi = mid

    x0 = t_grid[lo]
    x1 = t_grid[hi]
    y0 = rho_grid[lo]
    y1 = rho_grid[hi]
    if x1 == x0:
        return y0
    return y0 + (tb - x0) * (y1 - y0) / (x1 - x0)


@njit(cache=True, fastmath=True)
def _tgen_fast_numba(Fden, c, t_grid, rho_grid):
    for _ in range(1000000):
        u0 = np.random.random()
        u1 = np.random.random()
        if u0 < 0.5:
            t = 0.5 * np.log(2.0 * u0) / (np.pi * Fden)
        else:
            t = -0.5 * np.log(2.0 * (1.0 - u0)) / (np.pi * Fden)

        tb = abs(t)
        ld = u1 * c * np.pi * Fden * np.exp(-2.0 * np.pi * Fden * tb)
        rhof = _interp_rho(tb, Fden, t_grid, rho_grid)
        if ld <= rhof:
            return t
    raise RuntimeError("SSEGF delay rejection sampler did not converge")


@njit(cache=True, fastmath=True)
def _generate_rs_numba(lrs, lrm0, frem, Tce0, etac, etad, kappa, Fc, crc, vtc, rhoc, Fd, crd, vtd, rhod, seed):
    np.random.seed(seed)
    rs = np.zeros(lrs + 1, dtype=np.float64)
    lrm = lrm0
    Tce = Tce0

    for _ in range(etac):
        tc = _tgen_fast_numba(Fc, crc, vtc, rhoc)
        for _ in range(etad):
            tfi = tc + _tgen_fast_numba(Fd, crd, vtd, rhod) + Tce
            if tfi < 0.0:
                add = int(np.round(-tfi * frem))
                if add < 1:
                    add = 1
                new_rs = np.zeros(rs.size + add, dtype=np.float64)
                for i in range(rs.size):
                    new_rs[i + add] = rs[i]
                rs = new_rs
                Tce = Tce - tfi
                rs[0] += kappa
            elif tfi > lrm:
                add = int(np.round((tfi - lrm) * frem))
                if add < 1:
                    add = 1
                old_size = rs.size
                new_rs = np.zeros(old_size + add, dtype=np.float64)
                for i in range(old_size):
                    new_rs[i] = rs[i]
                rs = new_rs
                lrm = tfi
                rs[rs.size - 1] += kappa
            else:
                idx = int(np.round(tfi * frem))
                if idx >= rs.size:
                    idx = rs.size - 1
                rs[idx] += kappa
    return rs


@lru_cache(maxsize=128)
def delay_grid(stage, numerator_hz, denominator_hz, alpha, dt):
    upper = TC if stage == "c" else 1.0 / denominator_hz
    time = np.concatenate((10 ** np.arange(-6, np.log10(dt) - 0.1, 1.0),
                           np.arange(dt, upper + 0.2 * dt, 1.5 * dt)))
    return time, _rho_masa_core(time, numerator_hz, denominator_hz, alpha)


def synthesize(mw, egf_mw, target_fc, egf_fc, dt, seed):
    """Generate a two-stage STF with the original moment and time scaling."""
    n = egf_fc / target_fc
    if n < 1:
        raise ValueError("target fc exceeds the EGF fc used for SSEGF synthesis")
    etac = int(n ** 2)
    etad = int(round(n ** 4 / etac))
    kappa = 10 ** (1.5 * (mw - egf_mw)) / n ** 4
    fd = etac ** 0.25 * target_fc
    alc, ald = calf(fd, target_fc), calf(egf_fc, fd)
    t0 = 1e-7
    crc = _rho_core(t0, fd, target_fc, alc) / (np.pi * target_fc * np.exp(-2 * np.pi * target_fc * t0))
    # The original implementation uses alc for crd, and ald for its sampled delay grid.
    crd = _rho_core(t0, egf_fc, fd, alc) / (np.pi * fd * np.exp(-2 * np.pi * fd * t0))
    vtc, rhoc = delay_grid("c", fd, target_fc, alc, dt)
    vtd, rhod = delay_grid("d", egf_fc, fd, ald, dt)
    length = int(round(24 * TC / dt))
    series = _generate_rs_numba(length, length * dt, 1 / dt, 12 * TC, etac, etad,
                                kappa, target_fc, crc, vtc, rhoc, fd, crd, vtd, rhod, seed)
    nonzero = np.flatnonzero(series)
    return series[nonzero[0]:nonzero[-1] + 1].round(6)


if __name__ == "__main__":
    raise SystemExit(run_cli(run, __doc__))
