"""Step 1. Gutenberg–Richter magnitude rates and EGF Voronoi weights."""

import argparse
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from shapely import MultiPoint, Point, STRtree, get_coordinates, voronoi_polygons
from shapely.geometry import shape
from shapely.ops import transform


# Shared configuration, validation and file I/O for the seven scripts.
CASE = ["site_id", "event_id", "mw"]
PAIR = CASE[:2]
GEOD = Geod(ellps="WGS84")


class Project:
    def __init__(self, config_file):
        self.base = Path(config_file).resolve().parent
        self.config = json.loads(Path(config_file).read_text(encoding="utf-8-sig"))
        self.output = self.path("output_dir")

    def path(self, key):
        return self.base / self.config[key]

    def option(self, section):
        return self.config.get(section, {})


def table(path, columns, text=(), key=(), allow_empty=False):
    data = pd.read_csv(path, dtype={name: str for name in text})
    missing = set(columns) - set(data)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if (not allow_empty and data.empty) or data[list(columns)].isna().any().any():
        raise ValueError(f"{path}: empty table or missing required values")
    for name in set(text) & set(data):
        data[name] = data[name].str.strip()
    if key and data.duplicated(list(key)).any():
        raise ValueError(f"{path}: duplicate keys {list(key)}")
    return data


def identifiers(data, *columns):
    for name in columns:
        values = data[name].astype(str)
        invalid = ~values.map(lambda value: re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", value) is not None
                              and not value.endswith("."))
        if invalid.any():
            raise ValueError(f"{name}: unsafe identifier {values[invalid].iloc[0]!r}")


def numbers(data, *columns, positive=False, nonnegative=False):
    for name in columns:
        values = pd.to_numeric(data[name], errors="raise").astype(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: expected finite values")
        if (positive and (values <= 0).any()) or (nonnegative and (values < 0).any()):
            raise ValueError(f"{name}: invalid sign")
        data[name] = values


def coordinate_table(path, id_name):
    result = table(path, [id_name, "lon", "lat"], [id_name], [id_name])
    identifiers(result, id_name)
    numbers(result, "lon", "lat")
    if (result.lon.abs() > 180).any() or (result.lat.abs() > 90).any():
        raise ValueError(f"{path}: longitude or latitude out of range")
    return result


def distance_azimuth(lon1, lat1, lon2, lat2):
    azimuth, _, distance_m = GEOD.inv(lon1, lat1, lon2, lat2)
    return distance_m / 1000, azimuth % 360


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as file:
        temporary = Path(file.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def h5_output(path):
    with atomic_path(path) as temporary, h5py.File(temporary, "w") as archive:
        yield archive


def write_csv(data, path):
    with atomic_path(path) as temporary:
        data.to_csv(temporary, index=False)


def case_path(root, site_id, event_id, mw, filename):
    return root / site_id / f"{mw:.8f}".rstrip("0").rstrip(".") / event_id / filename


def check_dt(dt):
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt_s must be positive and finite")
    return dt


def magnitude_eligible(target_mw, egf_mw):
    """Require a magnitude difference strictly above 0.5, allowing for roundoff."""
    return target_mw - egf_mw > 0.5 + 1e-10


def check_series(series, description):
    if series.ndim != 1 or series.size < 2 or not np.isfinite(series).all():
        raise ValueError(f"{description}: expected a finite one-dimensional record of at least two samples")


def strings(archive, name, values):
    archive.create_dataset(name, data=list(values), dtype=h5py.string_dtype("utf-8"))


def read_manifest(root, keys, allow_empty=False):
    filename = root / "manifest.csv"
    data = table(filename, [*keys, "file"], [k for k in keys if k != "mw"], keys, allow_empty)
    identifiers(data, *(k for k in keys if k != "mw"))
    if "mw" in keys:
        numbers(data, "mw")
        data["mw"] = data.mw.round(8)
        if data.duplicated(keys).any():
            raise ValueError(f"{filename}: duplicate rounded magnitudes")
    data["file"] = data.file.map(lambda value: root / value)
    return data


def run_cli(run, description):
    """Run a calculation stage from a project JSON file."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", help="Path to the project JSON file")
    args = parser.parse_args()
    try:
        run(Project(args.config))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


EVENT = ["event_id", "zone_id", "lon", "lat", "egf_mw", "egf_fc_hz"]


def events(path):
    data = table(path, EVENT, ["event_id", "zone_id"], ["event_id"])
    identifiers(data, "event_id", "zone_id")
    numbers(data, "lon", "lat", "egf_mw")
    numbers(data, "egf_fc_hz", positive=True)
    if (data.lon.abs() > 180).any() or (data.lat.abs() > 90).any():
        raise ValueError("events.csv: longitude or latitude out of range")
    return data


def magnitude_rates(project):
    config = project.config
    choices = {"rates_csv", "rates_xlsx", "recurrence_csv"} & config.keys()
    if len(choices) != 1:
        raise ValueError("Configure exactly one of rates_csv, rates_xlsx or recurrence_csv")
    if "rates_csv" in config:
        rates = table(project.path("rates_csv"), ["zone_id", "mw", "annual_rate"],
                      ["zone_id"], ["zone_id", "mw"])
    elif "rates_xlsx" in config:
        sheets = pd.read_excel(project.path("rates_xlsx"), sheet_name=None, header=None)
        frames = [frame.rename(columns={0: "mw", 1: "annual_rate"}).assign(zone_id=name[4:])
                  for name, frame in sheets.items() if name.startswith("pss_")]
        if not frames:
            raise ValueError("Expected at least one pss_<zone_id> sheet")
        rates = pd.concat(frames, ignore_index=True)[["zone_id", "mw", "annual_rate"]]
    else:
        fields = ["zone_id", "a", "b", "years", "mw_min", "mw_max", "dm"]
        gr = table(project.path("recurrence_csv"), fields, ["zone_id"], ["zone_id"])
        numbers(gr, "a", "mw_min", "mw_max")
        numbers(gr, "b", "years", "dm", positive=True)
        records = []
        for row in gr.itertuples(index=False):
            if row.mw_max <= row.mw_min:
                raise ValueError(f"{row.zone_id}: mw_max must exceed mw_min")
            count = int(np.ceil(round((row.mw_max - row.mw_min) / row.dm, 10)))
            for index in range(count):
                low = row.mw_min + index * row.dm
                high = min(low + row.dm, row.mw_max)
                rate = (10 ** (row.a - row.b * low) - 10 ** (row.a - row.b * high)) / row.years
                records.append((row.zone_id, round((low + high) / 2, 8), rate))
        rates = pd.DataFrame(records, columns=["zone_id", "mw", "annual_rate"])
    numbers(rates, "mw")
    numbers(rates, "annual_rate", nonnegative=True)
    rates["mw"] = rates.mw.round(8)
    if rates.duplicated(["zone_id", "mw"]).any():
        raise ValueError("Duplicate magnitude rates after rounding to 8 decimal places")
    return rates


def voronoi_weights(project, event_data):
    geometry_data = json.loads(project.path("zones_geojson").read_text(encoding="utf-8-sig"))
    zones = {}
    for feature in geometry_data["features"]:
        zone_id = str(feature["properties"]["zone_id"])
        polygon = shape(feature["geometry"])
        if (zone_id in zones or polygon.geom_type not in ("Polygon", "MultiPolygon")
                or not polygon.is_valid or polygon.area <= 0):
            raise ValueError(f"{zone_id}: duplicate or invalid source-zone polygon")
        zones[zone_id] = polygon
    if not set(event_data.zone_id) <= set(zones):
        raise ValueError("Source-zone geometry missing for one or more EGF events")
    crs = project.option("scenarios").get("projected_crs", "local_aeqd")
    if crs == "local_aeqd":
        coordinates = np.vstack([event_data[["lon", "lat"]], *map(get_coordinates, zones.values())])
        lon, lat = coordinates.mean(axis=0)
        crs = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
    projection = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    if not projection.target_crs.is_projected or projection.target_crs.axis_info[0].unit_name != "metre":
        raise ValueError("scenarios.projected_crs must use projected coordinates in metres")

    records = []
    for zone_id, group in event_data.groupby("zone_id", sort=False):
        polygon = transform(projection.transform, zones[zone_id])
        coordinates = list(zip(*projection.transform(group.lon.to_numpy(), group.lat.to_numpy())))
        points = [Point(x, y) for x, y in coordinates]
        if len(set(coordinates)) != len(points) or any(not polygon.covers(point) for point in points):
            raise ValueError(f"{zone_id}: duplicate EGF coordinates or EGF outside its zone")
        cells = [polygon]
        if len(points) > 1:
            diagram = list(voronoi_polygons(MultiPoint(points), extend_to=polygon.envelope).geoms)
            tree = STRtree(diagram)
            cells = [diagram[int(tree.query(point, predicate="within").item())].intersection(polygon)
                     for point in points]
        areas = np.array([cell.area for cell in cells])
        if np.any(areas <= 0) or not np.isclose(areas.sum(), polygon.area, rtol=1e-7, atol=0):
            raise ValueError(f"{zone_id}: clipped cells do not partition the source zone")
        records.extend((event_id, zone_id, area / 1e6, area / areas.sum(), cell.wkt)
                       for event_id, area, cell in zip(group.event_id, areas, cells))
    return pd.DataFrame(records, columns=["event_id", "zone_id", "cell_km2", "weight", "cell_wkt"])


def run(project):
    event_data = events(project.path("events_csv"))
    rates = magnitude_rates(project)
    if "weights_csv" in project.config:
        path = project.path("weights_csv")
        weights = pd.read_csv(path, dtype={"Event": str, "ID_pss": str, "event_id": str, "zone_id": str})
        weights = weights.rename(columns={"Event": "event_id", "ID_pss": "zone_id",
                                          "Weight_Within_PSS": "weight"})
        if weights.duplicated("event_id").any() or set(weights.event_id) != set(event_data.event_id):
            raise ValueError(f"{path}: one weight is required for every listed event")
        numbers(weights, "weight", positive=True)
        expected = event_data.set_index("event_id").zone_id
        if not weights.set_index("event_id").zone_id.eq(expected).all():
            raise ValueError(f"{path}: zone IDs disagree with the event table")
    else:
        weights = voronoi_weights(project, event_data)
    if set(rates.zone_id) != set(weights.zone_id):
        raise ValueError("Magnitude rates and EGF weights must cover the same source zones")
    if not np.allclose(weights.groupby("zone_id").weight.sum(), 1, rtol=0, atol=1e-6):
        raise ValueError("EGF weights must sum to one within each source zone")
    scenarios = event_data.merge(weights[["event_id", "zone_id", "weight"]], on=["event_id", "zone_id"])
    scenarios = scenarios.merge(rates, on="zone_id")
    scenarios["scenario_rate"] = scenarios.annual_rate * scenarios.weight
    output = project.output / "01_scenarios"
    write_csv(weights, output / "weights.csv")
    write_csv(scenarios, output / "scenarios.csv")
    print(f"Step 1: {len(scenarios)} earthquake scenarios")


def read_scenarios(project):
    path = project.output / "01_scenarios" / "scenarios.csv"
    data = table(path, [*EVENT, "mw", "scenario_rate"], ["event_id", "zone_id"], ["event_id", "mw"])
    identifiers(data, "event_id", "zone_id")
    numbers(data, "mw", "lon", "lat", "egf_mw")
    numbers(data, "egf_fc_hz", positive=True)
    numbers(data, "scenario_rate", nonnegative=True)
    return data


if __name__ == "__main__":
    raise SystemExit(run_cli(run, __doc__))
