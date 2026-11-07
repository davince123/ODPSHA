"""Step 5. Rotate E/N records and compute PGA and absolute acceleration Sa."""

import h5py
import numpy as np
import pandas as pd
from numba import njit

from step01_scenarios import CASE, case_path, check_dt, h5_output, read_manifest, run_cli, strings, write_csv


def periods(settings):
    values = np.asarray(settings.get("periods_s", [0.5, 2.0]), dtype=float)
    if (values.ndim != 1 or not values.size or not np.isfinite(values).all()
            or np.any(values <= 0) or np.any(np.diff(values) <= 0)):
        raise ValueError("intensity.periods_s must be strictly increasing and positive")
    return values


def newmark_constants(period, damping, dt):
    beta, gamma = 0.25, 0.5
    a0 = 1 / (beta * dt ** 2)
    a1 = gamma / (beta * dt)
    a2 = 1 / (beta * dt)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = dt * (gamma / (2 * beta) - 1)
    omega = 2 * np.pi / period
    k, c = omega ** 2, 2 * damping * omega
    return k, c, k + a0 + a1 * c, dt, a0, a1, a2, a3, a4, a5, gamma


def write_im(archive, name, values, quantize):
    if quantize:
        scale = values.max() / np.iinfo(np.int16).max or 1.0
        dataset = archive.create_dataset(name, data=np.rint(values / scale).astype(np.int16),
                                         compression="gzip", shuffle=True)
        dataset.attrs["scale"] = scale
    else:
        archive.create_dataset(name, data=values, compression="gzip", shuffle=True)


def run(project):
    motion = read_manifest(project.output / "04_motion", CASE, allow_empty=True)
    settings = project.option("intensity")
    output = project.output / "05_im"
    spectral_periods = periods(settings)
    damping = float(settings.get("damping", 0.05))
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("intensity.damping must be nonnegative and finite")
    raw_angles = np.arange(180, dtype=float)
    radians = np.deg2rad(raw_angles)
    cosine, sine = np.cos(radians), np.sin(radians)
    manifest = []
    for row in motion.itertuples(index=False):
        path = case_path(output, row.site_id, row.event_id, row.mw, "RotD_IM.h5")
        with h5py.File(row.file) as wave:
            dt = check_dt(wave.attrs["dt_s"])
            unit = wave.attrs["acceleration_unit"]
            stiffness, damping_coefficient, effective, *constants = newmark_constants(spectral_periods, damping, dt)
            names = wave["record_names"].asstr()[:]
            weights = wave["weights"][:]
            pga = np.empty((len(raw_angles), len(names)))
            sa = np.empty((len(raw_angles), len(spectral_periods), len(names)))
            for index in range(len(names)):
                dataset = wave["acceleration"][str(index)]
                pair = dataset[:].astype(float)
                if "scale" in dataset.attrs:
                    pair *= dataset.attrs["scale"][:, None]
                pga[:, index], sa[:, :, index] = rotd_pga_and_sa_numba(
                    pair[0], pair[1], stiffness, damping_coefficient, effective,
                    cosine, sine, *constants)
        if not np.isfinite(pga).all() or not np.isfinite(sa).all():
            raise ValueError(f"{path}: non-finite intensity measure")
        with h5_output(path) as archive:
            archive.attrs.update(dt=dt, xi=damping, acceleration_unit=unit)
            write_im(archive, "PGA", pga, settings.get("int16", False))
            write_im(archive, "Sa", sa, settings.get("int16", False))
            archive.create_dataset("periods", data=spectral_periods)
            archive.create_dataset("angles_deg", data=raw_angles)
            archive.create_dataset("weights", data=weights)
            strings(archive, "record_names", names)
        manifest.append((row.site_id, row.event_id, row.mw, path.relative_to(output).as_posix(), len(names)))
    write_csv(pd.DataFrame(manifest, columns=[*CASE, "file", "n_records"]), output / "manifest.csv")
    print(f"Step 5: rotational IMs for {len(manifest)} site/scenario pairs")


# Serial Newmark kernel adapted from the original IM_rotd_all.py.
@njit(cache=True, fastmath=True)
def rotd_pga_and_sa_numba(
    acc_x,
    acc_y,
    period_k,
    period_c,
    period_k_eff,
    angle_cos,
    angle_sin,
    dt,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    gamma,
):
    n_angles = angle_cos.size
    n_periods = period_k.size
    n_steps = acc_x.size

    pga = np.empty(n_angles, dtype=np.float64)
    sa = np.empty((n_angles, n_periods), dtype=np.float64)

    for ia in range(n_angles):
        cos_t = angle_cos[ia]
        sin_t = angle_sin[ia]
        acc_rot = np.empty(n_steps, dtype=np.float64)

        ag0 = cos_t * acc_x[0] + sin_t * acc_y[0]
        acc_rot[0] = ag0
        pga_max = abs(ag0)

        for it in range(1, n_steps):
            ag = cos_t * acc_x[it] + sin_t * acc_y[it]
            acc_rot[it] = ag
            abs_ag = abs(ag)
            if abs_ag > pga_max:
                pga_max = abs_ag

        pga[ia] = pga_max

        for ip in range(n_periods):
            c = period_c[ip]
            k_eff = period_k_eff[ip]

            u = 0.0
            v = 0.0
            a_rel = -ag0
            sa_max = 0.0

            for it in range(n_steps - 1):
                ag = acc_rot[it + 1]

                p_eff = (
                    -ag
                    + (a0 * u + a2 * v + a3 * a_rel)
                    + c * (a1 * u + a4 * v + a5 * a_rel)
                )

                u_new = p_eff / k_eff
                a_new = a0 * (u_new - u) - a2 * v - a3 * a_rel
                v_new = v + dt * ((1.0 - gamma) * a_rel + gamma * a_new)

                a_abs = abs(a_new + ag)
                if a_abs > sa_max:
                    sa_max = a_abs

                u = u_new
                v = v_new
                a_rel = a_new

            sa[ia, ip] = sa_max

    return pga, sa


if __name__ == "__main__":
    raise SystemExit(run_cli(run, __doc__))
