"""Step 6. Integrate scenario-weighted empirical and lognormal exceedance."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.special import ndtr

from step01_scenarios import (CASE, coordinate_table, magnitude_eligible, read_manifest,
                              read_scenarios, run_cli, write_csv)
from step05_intensity import ANGLE_CONVENTION, periods


def im_labels(spectral_periods):
    return ["PGA", *(f"Sa_{period:g}s" for period in spectral_periods)]


def read_im(path, spectral_periods, unit):
    with h5py.File(path) as archive:
        if archive.attrs.get("acceleration_unit", unit) != unit:
            raise ValueError(f"{path}: acceleration unit differs from project configuration")
        if archive.attrs.get("angle_convention") != ANGLE_CONVENTION:
            raise ValueError(f"{path}: IM archive must use azimuths measured clockwise from north. "
                             "Regenerate the archive with step 5.")
        pga = archive["PGA"][:].astype(float) * archive["PGA"].attrs.get("scale", 1.0)
        sa = archive["Sa"][:].astype(float) * archive["Sa"].attrs.get("scale", 1.0)
        azimuths = archive["angles_deg"][:]
        archive_periods = archive["periods"][:]
        weights = archive["weights"][:] if "weights" in archive else np.ones(pga.shape[-1])
    if (pga.ndim != 2 or sa.shape != (pga.shape[0], len(archive_periods), pga.shape[1])
            or pga.shape[1] == 0 or weights.shape != (pga.shape[1],)
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError(f"{path}: incompatible IM shapes or invalid sample weights")
    if pga.shape[0] != azimuths.size or not np.array_equal(azimuths, np.arange(180)):
        raise ValueError(f"{path}: expected geographic azimuths 0–179°")
    selected = []
    for period in spectral_periods:
        hits = np.flatnonzero(np.isclose(archive_periods, period, atol=1e-9, rtol=0))
        if hits.size != 1:
            raise ValueError(f"{path}: expected one spectral period matching {period:g} s")
        selected.append(sa[:, hits.item(), :])
    im = np.stack([pga, *selected])
    if not np.isfinite(im).all() or np.any(im < 0):
        raise ValueError(f"{path}: expected finite nonnegative IM values")
    return im, weights / weights.sum(), azimuths


def cases(project):
    scenarios = read_scenarios(project)
    sites = coordinate_table(project.path("sites_csv"), "site_id")
    frame = sites[["site_id"]].merge(scenarios, how="cross")
    if "legacy_im_dir" in project.config:
        root = project.path("legacy_im_dir")
        if not root.is_dir():
            raise FileNotFoundError(root)
        frame["file"] = [root / row.site_id / str(row.mw) / row.event_id / "RotD_IM.h5"
                         for row in frame.itertuples(index=False)]
    else:
        index = read_manifest(project.output / "05_im", CASE, allow_empty=True)
        frame = frame.merge(index[[*CASE, "file"]], on=CASE, how="left", validate="one_to_one")
    frame["magnitude_eligible"] = magnitude_eligible(frame.mw, frame.egf_mw)
    frame["available"] = frame.magnitude_eligible & frame.file.map(
        lambda path: isinstance(path, Path) and path.is_file())
    return frame


def levels_and_angles(settings):
    grid = settings.get("im_grid", [0.0001, 10.0, 2000])
    if (len(grid) != 3 or not np.isfinite(grid).all() or not 0 < grid[0] < grid[1]
            or int(grid[2]) != grid[2] or grid[2] < 2):
        raise ValueError("hazard.im_grid must be [positive min, larger max, integer count >= 2]")
    angles = np.asarray(settings.get("fixed_azimuths_deg", list(range(180))), dtype=float)
    if (angles.ndim != 1 or not np.isfinite(angles).all()
            or np.any((angles < 0) | (angles >= 180) | (angles != np.rint(angles)))
            or np.unique(angles).size != angles.size):
        raise ValueError("hazard.fixed_azimuths_deg must be distinct integer angles in [0, 180)")
    return np.geomspace(grid[0], grid[1], int(grid[2])), angles.astype(int)


def empirical_exceedance(values, weights, levels):
    order = np.argsort(values, kind="stable")
    cumulative = np.r_[np.cumsum(weights[order][::-1])[::-1], 0.0]
    return cumulative[np.searchsorted(values[order], levels, side="right")]


def lognormal_exceedance(values, weights, levels):
    if values.size < 2 or np.any(values <= 0):
        return None
    logs = np.log(values)
    mu = weights @ logs
    degrees = 1 - weights @ weights
    if degrees <= 0 or np.ptp(logs) == 0:
        return None
    sigma = np.sqrt((weights @ (logs - mu) ** 2) / degrees)
    if sigma == 0:
        return None
    return ndtr((mu - np.log(levels)) / sigma)


def run(project):
    settings = project.option("hazard")
    levels, azimuths = levels_and_angles(settings)
    mode = settings.get("method", "both")
    if mode not in ("empirical", "lognormal", "both"):
        raise ValueError("hazard.method must be empirical, lognormal, or both")
    spectral_periods = periods(project.option("intensity"))
    im_names = im_labels(spectral_periods)
    variants = [*(f"fixed_{angle:03d}" for angle in azimuths), "RotD00", "RotD50", "RotD100"]
    output = project.output / "06_hazard"
    frame = cases(project)
    columns = [*CASE, "zone_id", "scenario_rate", "magnitude_eligible", "available"]
    write_csv(frame[columns], output / "coverage.csv")
    missing = frame[~frame.available & (frame.scenario_rate > 0)]
    if not missing.empty and not settings.get("allow_missing", False):
        raise ValueError(f"{len(missing)} positive-rate site–scenario pairs have no eligible IM archive. "
                         f"See {output / 'coverage.csv'}")

    shape = (len(im_names), len(variants), len(levels))
    summary = []
    for site_id, group in frame.groupby("site_id", sort=False):
        available = group[group.available & (group.scenario_rate > 0)]
        if available.empty:
            raise ValueError(f"{site_id}: no usable positive-rate scenarios")
        empirical = np.zeros(shape) if mode != "lognormal" else None
        lognormal = np.zeros(shape) if mode != "empirical" else None
        fallback = 0
        for row in available.itertuples(index=False):
            matrix, weights, _ = read_im(row.file, spectral_periods, project.config["acceleration_unit"])
            statistics = np.concatenate([matrix[:, azimuths], matrix.min(axis=1, keepdims=True),
                                         np.median(matrix, axis=1, keepdims=True),
                                         matrix.max(axis=1, keepdims=True)], axis=1)
            for im_index, values in enumerate(statistics):
                for variant, samples in enumerate(values):
                    if empirical is not None:
                        empirical[im_index, variant] += row.scenario_rate * empirical_exceedance(samples, weights, levels)
                    if lognormal is not None:
                        exceedance = lognormal_exceedance(samples, weights, levels)
                        if exceedance is None:
                            fallback += 1
                            exceedance = empirical_exceedance(samples, weights, levels)
                        lognormal[im_index, variant] += row.scenario_rate * exceedance
        for method, curves in (("empirical", empirical), ("lognormal", lognormal)):
            if curves is None:
                continue
            for im_name, values in zip(im_names, curves):
                curve = pd.DataFrame(values.T, columns=variants)
                curve.insert(0, "im_level", levels)
                write_csv(curve, output / site_id / f"{im_name}_{method}.csv")
        summary.append((site_id, len(available), group.scenario_rate.sum(), available.scenario_rate.sum(), fallback))
    write_csv(pd.DataFrame(summary, columns=["site_id", "n_used", "total_rate", "included_rate",
                                             "lognormal_fallback_series"]), output / "coverage_summary.csv")
    print(f"Step 6: hazard curves at {len(summary)} sites")


if __name__ == "__main__":
    raise SystemExit(run_cli(run, __doc__))
