"""Step 7. Return-period maps and RotD100 hazard-direction disaggregation."""

import numpy as np
import pandas as pd
from scipy.special import i0e

from step01_scenarios import coordinate_table, run_cli, write_csv
from step06_hazard import cases, im_labels, read_im
from step05_intensity import periods


def axial_fit(angle_deg, weight):
    vector = np.average(np.exp(2j * np.deg2rad(angle_deg)), weights=weight)
    mean = (np.rad2deg(np.angle(vector)) / 2) % 180
    resultant = min(abs(vector), 0.999999)
    if resultant < 1e-8:
        kappa = 0.0
    elif resultant < 0.53:
        kappa = 2 * resultant + resultant ** 3 + 5 * resultant ** 5 / 6
    elif resultant < 0.85:
        kappa = -0.4 + 1.39 * resultant + 0.43 / (1 - resultant)
    else:
        kappa = 1 / (resultant ** 3 - 4 * resultant ** 2 + 3 * resultant)
    return mean, kappa, resultant


def axial_density(mean, kappa):
    theta = np.arange(181, dtype=float)
    if kappa > 100:
        # Resolve narrow peaks, including peaks split across 0 and 180 degrees.
        width = np.rad2deg(0.5 / np.sqrt(kappa))
        local = (mean + width * np.linspace(-6, 6, 121)) % 180
        theta = np.unique(np.r_[theta, local])
    density = np.exp(kappa * (np.cos(2 * np.deg2rad(theta - mean)) - 1)) / (180 * i0e(kappa))
    return theta, density


def read_targets(project, labels, return_periods, sites, window):
    map_rows, targets = [], {}
    for site in sites.itertuples(index=False):
        for im_name in labels:
            path = project.output / "06_hazard" / site.site_id / f"{im_name}_empirical.csv"
            curve = pd.read_csv(path)
            levels = curve.pop("im_level").to_numpy()
            for rp in return_periods:
                for variant in curve:
                    rate = curve[variant].to_numpy()
                    target = 1 / rp
                    valid = rate[-1] <= target <= rate[0] and rate[0] > rate[-1]
                    index = np.argmin(np.abs(rate - target))
                    im_level = levels[index] if valid else np.nan
                    status = "ok" if valid else "outside_grid"
                    map_rows.append((site.site_id, site.lon, site.lat, im_name, rp, variant, im_level, status))
                    if variant == "RotD100" and valid:
                        low = levels[max(index - window, 0)]
                        high = levels[min(index + window, len(levels) - 1)]
                        targets[(site.site_id, im_name, rp)] = (im_level, low, high)
    return map_rows, targets


def run(project):
    if project.option("hazard").get("method") == "lognormal":
        raise ValueError("Step 7 requires the empirical hazard curves from step 6")
    settings = project.option("postprocess")
    return_periods = np.asarray(settings.get("return_periods_years", [50, 475, 2500]), dtype=float)
    window = int(settings.get("window_indices", 50))
    convention = settings.get("angle_convention", "geographic")
    if (return_periods.ndim != 1 or not return_periods.size or not np.isfinite(return_periods).all()
            or np.any(return_periods <= 0) or np.unique(return_periods).size != return_periods.size
            or window < 1 or convention not in ("geographic", "legacy_pdf")):
        raise ValueError("Invalid postprocess return periods, window or angle convention")
    spectral_periods = periods(project.option("intensity"))
    labels = im_labels(spectral_periods)
    sites = coordinate_table(project.path("sites_csv"), "site_id")
    map_rows, targets = read_targets(project, labels, return_periods, sites, window)
    samples = {key: [] for key in targets}
    frame = cases(project)
    for row in frame[frame.available & (frame.scenario_rate > 0)].itertuples(index=False):
        matrix, weights, raw = read_im(row.file, spectral_periods, project.config["acceleration_unit"])
        maximum = matrix.max(axis=1)
        raw_at_max = raw[matrix.argmax(axis=1)]
        for index, im_name in enumerate(labels):
            for rp in return_periods:
                key = (row.site_id, im_name, rp)
                if key not in targets:
                    continue
                _, low, high = targets[key]
                keep = (low < maximum[index]) & (maximum[index] < high)
                if keep.any():
                    angle = raw_at_max[index, keep]
                    if convention == "legacy_pdf":
                        angle = -np.abs(90 - angle) % 180
                    else:
                        angle = (90 - angle) % 180
                    rate = row.scenario_rate * weights[keep]
                    samples[key].append((row.zone_id, row.event_id, row.mw, angle, rate))
    result, contributions, densities = [], [], []
    for key, entries in samples.items():
        level, low, high = targets[key]
        if not entries:
            result.append((*key, level, low, high, 0.0, np.nan, np.nan, np.nan, convention, "empty_window"))
            continue
        angles = np.concatenate([entry[3] for entry in entries])
        weights = np.concatenate([entry[4] for entry in entries])
        total = weights.sum()
        mean, kappa, resultant = axial_fit(angles, weights)
        theta, pdf = axial_density(mean, kappa)
        densities.extend((*key, angle, density) for angle, density in zip(theta, pdf))
        for zone, event, mw, _, rate in entries:
            contributions.append((*key, zone, event, mw, rate.sum(), rate.sum() / total))
        result.append((*key, level, low, high, total, mean, kappa, resultant, convention, "ok"))
    output = project.output / "07_postprocess"
    keys = ["site_id", "im", "return_period_years"]
    write_csv(pd.DataFrame(map_rows, columns=["site_id", "lon", "lat", "im", "return_period_years",
                                             "variant", "im_level", "status"]), output / "return_levels.csv")
    write_csv(pd.DataFrame(result, columns=[*keys, "im_level", "window_low", "window_high", "window_rate",
                                           "mu_deg", "kappa", "resultant", "angle_convention", "status"]),
              output / "direction_summary.csv")
    write_csv(pd.DataFrame(contributions, columns=[*keys, "zone_id", "event_id", "mw", "annual_rate", "fraction"]),
              output / "disaggregation.csv")
    write_csv(pd.DataFrame(densities, columns=[*keys, "angle_deg", "density_per_deg"]),
              output / "direction_pdf.csv")
    print(f"Step 7: return levels and RotD100 disaggregation for {len(sites)} sites")


if __name__ == "__main__":
    raise SystemExit(run_cli(run, __doc__))
