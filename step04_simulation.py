"""Step 4. Convolve one STF with both EGF components for each realization."""

import h5py
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from step01_scenarios import CASE, PAIR, case_path, h5_output, read_manifest, run_cli, strings, write_csv
from step03_egf import apparent_fc, brune_ratio, fourier_transfer


def simulate(pair, stf, dt, fc, azimuth, settings):
    motion = fftconvolve(pair, stf[None, :], axes=-1, mode="full")
    if settings.get("target_directivity", True):
        directed = apparent_fc(fc, azimuth, settings.get("rupture_azimuth_deg", 60.0),
                               settings.get("velocity_ratio", 0.8))
        grid = np.arange(0.5, 25.01, 0.5)
        motion = fourier_transfer(motion, dt, grid, brune_ratio(grid, directed, fc),
                                  settings.get("band_method", "legacy_bins"))
    return motion


def write_motion(group, index, motion, quantize):
    if not np.isfinite(motion).all():
        raise ValueError("Non-finite simulated acceleration")
    if quantize:
        scale = np.max(np.abs(motion), axis=1) / np.iinfo(np.int16).max
        scale[scale == 0] = 1
        encoded = np.rint(motion / scale[:, None]).astype(np.int16)
        dataset = group.create_dataset(str(index), data=encoded, compression="gzip", shuffle=True)
        dataset.attrs["scale"] = scale
    else:
        group.create_dataset(str(index), data=motion, compression="gzip", shuffle=True)


def run(project):
    stfs = read_manifest(project.output / "02_stf", ["event_id", "mw"], allow_empty=True)
    egfs = read_manifest(project.output / "03_egf", PAIR, allow_empty=True)
    tasks = egfs.rename(columns={"file": "egf_file"}).merge(
        stfs.rename(columns={"file": "stf_file"}), on="event_id")
    settings = project.option("simulation")
    output = project.output / "04_motion"
    manifest = []
    for row in tasks.itertuples(index=False):
        path = case_path(output, row.site_id, row.event_id, row.mw, "motion.h5")
        with h5py.File(row.stf_file) as source, h5py.File(row.egf_file) as egf, h5_output(path) as archive:
            dt = float(egf.attrs["dt_s"])
            if not np.isclose(dt, source.attrs["dt_s"], atol=1e-10, rtol=0):
                raise ValueError(f"STF and EGF sampling intervals differ: {path}")
            azimuth = float(egf.attrs["target_azimuth_deg"])
            archive.attrs.update(dt_s=dt, acceleration_unit=project.config["acceleration_unit"])
            series = archive.create_group("acceleration")
            names, weights = [], []
            donors = egf["donor_id"].asstr()[:]
            samples = source["sample_id"].asstr()[:]
            frequencies = source["fc_hz"][:]
            stf_weights = source["weight"][:]
            pairs = [egf["egf"][str(index)][:] for index in range(len(donors))]
            for i, sample_id in enumerate(samples):
                stf = source["stf"][str(i)][:]
                for donor_id, pair in zip(donors, pairs):
                    motion = simulate(pair, stf, dt, frequencies[i], azimuth, settings)
                    write_motion(series, len(names), motion, settings.get("int16", False))
                    names.append(f"{sample_id}/{donor_id}")
                    weights.append(stf_weights[i] / len(donors))
            strings(archive, "record_names", names)
            archive.create_dataset("weights", data=weights)
        manifest.append((row.site_id, row.event_id, row.mw, path.relative_to(output).as_posix(), len(names)))
    write_csv(pd.DataFrame(manifest, columns=[*CASE, "file", "n_records"]), output / "manifest.csv")
    print(f"Step 4: simulated motion for {len(manifest)} site/scenario pairs")


if __name__ == "__main__":
    raise SystemExit(run_cli(run, __doc__))
