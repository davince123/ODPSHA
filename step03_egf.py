"""Step 3. Select EGF donors and apply a common Fourier transfer to E/N."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
from scipy.interpolate import interp1d

from step01_scenarios import (PAIR, check_dt, check_series, coordinate_table, distance_azimuth,
                     h5_output, identifiers, read_scenarios, run_cli, strings, table, write_csv)


def apparent_fc(fc, azimuth, rupture_azimuth, velocity_ratio):
    if not 0 <= velocity_ratio < 1:
        raise ValueError("velocity_ratio must be in [0, 1)")
    return fc / np.sqrt(1 - velocity_ratio * np.cos(np.deg2rad(azimuth - rupture_azimuth)))


def brune_ratio(freq, target_fc, reference_fc):
    return (1 + (freq / reference_fc) ** 2) / (1 + (freq / target_fc) ** 2)


def band_gain(frequency, grid, ratios, method):
    if method == "legacy_bins":
        index = np.searchsorted(grid, frequency, side="left")
        gain = np.ones_like(frequency)
        valid = index < grid.size
        gain[valid] = ratios[index[valid]]
        return gain
    if method == "linear":
        return np.interp(frequency, grid, ratios, left=1, right=1)
    raise ValueError("band method must be legacy_bins or linear")


def fourier_transfer(pair, dt, grid, ratio, method, shift_s=0):
    if shift_s:
        pad = int(np.ceil(abs(shift_s) / dt)) + 256
        length = next_fast_len(pair.shape[-1] + 2 * pad)
        pair = np.pad(pair, ((0, 0), (pad, length - pad - pair.shape[-1])))
    freq = rfftfreq(pair.shape[-1], dt)
    gain = band_gain(freq, grid, ratio, method)
    transfer = gain * np.exp(-2j * np.pi * freq * shift_s)
    return irfft(rfft(pair, axis=-1) * transfer, n=pair.shape[-1], axis=-1)


def read_attenuation(path):
    with path.open(encoding="utf-8-sig") as handle:
        handle.readline()
        header = handle.readline().split()
        rows = np.loadtxt(handle, ndmin=2)
    frequency = np.asarray(header[1:], dtype=float)
    if (frequency.size < 2 or rows.shape[0] < 2 or rows.shape[1] != len(header) or not np.isfinite(rows).all()
            or not np.isfinite(frequency).all() or np.any(frequency <= 0)
            or np.any(np.diff(rows[:, 0]) <= 0) or np.any(np.diff(frequency) <= 0)):
        raise ValueError(f"{path}: invalid GITeps attenuation grid")
    interp = interp1d(rows[:, 0], rows[:, 1:], axis=0, fill_value="extrapolate")
    return frequency, interp


@lru_cache(maxsize=1024)
def read_amplification(path):
    if path.suffix.lower() == ".site":
        values = np.loadtxt(path, skiprows=1, ndmin=2)
    else:
        values = table(path, ["Frequency_Hz", "Site_Amplification"])[
            ["Frequency_Hz", "Site_Amplification"]].to_numpy(dtype=float)
    if (values.shape[1] != 2 or values.shape[0] < 2 or not np.isfinite(values).all()
            or np.any(values <= 0) or np.any(np.diff(values[:, 0]) <= 0)):
        raise ValueError(f"{path}: invalid site amplification spectrum")
    return values.T


def aligned_spectrum(grid, spectrum, path):
    frequency, amplitude = spectrum
    if grid[0] < frequency[0] - 1e-8 or grid[-1] > frequency[-1] + 1e-8:
        raise ValueError(f"{path}: site spectrum does not cover the transfer frequency grid")
    return np.interp(grid, frequency, amplitude)


def waveform(path, format_name, dt):
    if format_name == "fas":
        data = table(path, ["frequency_Hz", "amplitude", "phase_rad"])
        frequency = data.frequency_Hz.to_numpy(dtype=float)
        if not np.allclose(frequency, np.fft.fftfreq(frequency.size, dt), rtol=1e-6, atol=1e-8):
            raise ValueError(f"{path}: expected a full FFT frequency axis for dt={dt}")
        values = np.fft.ifft(data.amplitude.to_numpy(dtype=float)
                             * np.exp(1j * data.phase_rad.to_numpy(dtype=float)))
        if np.max(np.abs(values.imag)) > 1e-8 * max(1, np.max(np.abs(values.real))):
            raise ValueError(f"{path}: Fourier spectrum does not reconstruct a real waveform")
        values = values.real
    elif format_name == "txt":
        values = np.loadtxt(path, ndmin=1)
    else:
        raise ValueError(f"{path}: record format must be txt or fas")
    check_series(values, path)
    return values


def read_pair(e_path, n_path, format_name, dt):
    east = waveform(e_path, format_name, dt)
    north = waveform(n_path, format_name, dt)
    if east.shape != north.shape:
        raise ValueError(f"Paired records have different lengths: {e_path}, {n_path}")
    return np.stack([east, north])


def legacy_pairs(folder, dt):
    records = []
    for e_path in sorted(folder.glob("*_GM_E.txt")):
        prefix = e_path.name.removesuffix("_GM_E.txt")
        n_path = folder / f"{prefix}_GM_N.txt"
        records.append((prefix.rsplit("_", 1)[-1], read_pair(e_path, n_path, "txt", dt)))
    return records


class Transfer:
    def __init__(self, project, dt):
        self.dt = dt
        self.settings = project.option("egf")
        self.frequency, self.attenuation = read_attenuation(project.path("attenuation_txt"))
        path = project.path("site_spectra_csv")
        spectra = table(path, ["location_id", "file"], ["location_id"], ["location_id"])
        self.spectra = {row.location_id: path.parent / row.file for row in spectra.itertuples(index=False)}

    def apply(self, pair, event, site, donor, target_distance, target_azimuth):
        reference_distance, reference_azimuth = distance_azimuth(event.lon, event.lat, donor.lon, donor.lat)
        velocity_ratio = self.settings.get("velocity_ratio", 0.8)
        direction = self.settings.get("small_event_rupture_azimuth_deg", 0.0)
        fc_target = apparent_fc(event.egf_fc_hz, target_azimuth, direction, velocity_ratio)
        fc_reference = apparent_fc(event.egf_fc_hz, reference_azimuth, direction, velocity_ratio)
        grid = self.frequency
        path_ratio = 10 ** (self.attenuation(target_distance) - self.attenuation(reference_distance))
        target_path, reference_path = self.spectra[site.site_id], self.spectra[donor.station_id]
        site_ratio = (aligned_spectrum(grid, read_amplification(target_path), target_path)
                      / aligned_spectrum(grid, read_amplification(reference_path), reference_path))
        ratio = brune_ratio(grid, fc_target, fc_reference) * path_ratio * site_ratio
        if not np.isfinite(ratio).all() or np.any(ratio <= 0):
            raise ValueError("Invalid EGF Fourier gain")
        shift = 0
        if self.settings.get("travel_time", False):
            velocity = float(self.settings.get("shear_velocity_km_s", 3.5))
            if not np.isfinite(velocity) or velocity <= 0:
                raise ValueError("egf.shear_velocity_km_s must be positive")
            shift = (target_distance - reference_distance) / velocity
        return fourier_transfer(pair, self.dt, grid, ratio, self.settings.get("band_method", "legacy_bins"), shift)


def run(project):
    dt = check_dt(project.config["dt_s"])
    sites = coordinate_table(project.path("sites_csv"), "site_id")
    events = read_scenarios(project).drop_duplicates("event_id")
    settings = project.option("egf")
    legacy_root = project.path("legacy_egf_dir") if "legacy_egf_dir" in project.config else None
    transfer = None
    if legacy_root is None:
        record_path = project.path("records_csv")
        fields = ["event_id", "station_id", "e_file", "n_file"]
        records = table(record_path, fields, ["event_id", "station_id"], ["event_id", "station_id"])
        identifiers(records, "event_id", "station_id")
        if "format" not in records:
            records["format"] = "txt"
        stations = coordinate_table(project.path("stations_csv"), "station_id")
        if not set(records.station_id) <= set(stations.station_id):
            raise ValueError("records.csv references unlisted stations")
        groups = {key: group.merge(stations, on="station_id").reset_index(drop=True)
                  for key, group in records.groupby("event_id")}
        donors = int(settings.get("nearest_donors", 5))
        if donors < 1:
            raise ValueError("egf.nearest_donors must be at least 1")
    output = project.output / "03_egf"
    manifest = []
    for site in sites.itertuples(index=False):
        for event in events.itertuples(index=False):
            target_distance, target_azimuth = distance_azimuth(event.lon, event.lat, site.lon, site.lat)
            if target_distance > settings.get("max_event_distance_km", 250):
                continue
            if legacy_root:
                pairs = legacy_pairs(legacy_root / site.site_id / event.event_id, dt)
            elif event.event_id in groups:
                candidates = groups[event.event_id].copy()
                candidates["distance"], _ = distance_azimuth(
                    candidates.lon.to_numpy(), candidates.lat.to_numpy(),
                    np.full(len(candidates), site.lon), np.full(len(candidates), site.lat),
                )
                direct = candidates[candidates.distance <= settings.get("direct_tolerance_km", 0.001)]
                chosen = direct if not direct.empty else candidates
                chosen = chosen.sort_values(["distance", "station_id"]).head(donors)
                pairs = []
                for donor in chosen.itertuples(index=False):
                    pair = read_pair(record_path.parent / donor.e_file, record_path.parent / donor.n_file,
                                     donor.format, dt)
                    if direct.empty:
                        if transfer is None:
                            transfer = Transfer(project, dt)
                        pair = transfer.apply(pair, event, site, donor, target_distance, target_azimuth)
                    pairs.append((donor.station_id, pair))
            else:
                continue
            if not pairs:
                continue
            path = Path(site.site_id) / f"{event.event_id}.h5"
            with h5_output(output / path) as archive:
                archive.attrs.update(dt_s=dt, target_azimuth_deg=target_azimuth)
                strings(archive, "donor_id", [donor_id for donor_id, _ in pairs])
                waves = archive.create_group("egf")
                for index, (_, pair) in enumerate(pairs):
                    waves.create_dataset(str(index), data=pair)
            manifest.append((site.site_id, event.event_id, path.as_posix(), len(pairs)))
    write_csv(pd.DataFrame(manifest, columns=[*PAIR, "file", "n_donor"]), output / "manifest.csv")
    print(f"Step 3: {len(manifest)} site/event pairs")


if __name__ == "__main__":
    raise SystemExit(run_cli(run, __doc__))
