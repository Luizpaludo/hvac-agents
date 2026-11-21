"""
lbnl.py - Loads the LBNL FDD single-duct AHU dataset (DOI 10.25984/1881324) for the agents.

Area 6: data pipeline (owner: Mary Ann Mastri). Code review: docs/CODE_REVIEW_AREA6.md

What the raw files look like (checked 11/19/2026, see docs/CHALLENGES_LOG.md, section D):
  * 21 CSV files, 1-minute data for one year (2018 labels, Chicago TMY weather), 31 columns.
  * Temperatures in degF; valve, damper and fan signals as 0-1 fractions.
  * The simulation calendar is shifted one day (labelled Mondays are unoccupied), so we use the
    occupancy column SYS_CTL instead of the calendar.
  * Static pressure SA_SP is in Pa in some files and inWC in others; the setpoint column is wrong
    in the fault files (-400). We convert SA_SP to inWC and use the documented setpoint 1.607 inWC.
  * SF_SPD and OA_CFM are constant all year (not usable); fan speed = SF_CS (control signal).
  * The 4 coil-leakage files are byte-identical, and so are the 4 OA-sensor-bias files, so the
    dataset has 15 different files (1 fault-free + 14 faulty), not 21. Duplicates are loaded once.
  * There is no heating-coil signal (reheat is at the zone boxes), so hcv is set to 0.
  * Labels come from the file name; the fault is active the whole year. Some faults have no effect
    in some seasons (e.g. a damper stuck at 10 % when the minimum position is 10 %).

Processing: occupied minutes only -> 15-minute means (complete 15-minute blocks only) -> our column
names in degC / % / inWC. Results are cached in data/lbnl_processed/ (gitignored).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

RAW_DIR = config.DATA_DIR / "LBNL_FDD_Dataset_SDAHU"
CACHE_DIR = config.DATA_DIR / "lbnl_processed"
STATIC_SP_INWC = 1.607          # documented static pressure setpoint
PA_PER_INWC = 249.089
SOURCE = "LBNL FDD dataset (simulated by LBNL, EnergyPlus-Modelica, Chicago weather; building AHU, not cruise ship)"

USECOLS = ["Datetime", "OA_TEMP", "RA_TEMP", "MA_TEMP", "SA_TEMP", "SA_TEMPSPT", "OA_DMPR", "OA_DMPR_DM",
           "CHWC_VLV", "CHWC_VLV_DM", "SF_CS", "SA_SP", "SYS_CTL"]

# file -> (fault_type used by the agents, severity text). Fault types match agents/ranker.py.
SCENARIOS = {
    "AHU_annual.csv": ("none", "fault-free"),
    "coi_bias_-4_annual.csv": ("sat_sensor_bias", "-4 C"),
    "coi_bias_-2_annual.csv": ("sat_sensor_bias", "-2 C"),
    "coi_bias_2_annual.csv": ("sat_sensor_bias", "+2 C"),
    "coi_bias_4_annual.csv": ("sat_sensor_bias", "+4 C"),
    "coi_leakage_010_annual.csv": ("ccv_leakage", "see note: 4 files identical"),
    "coi_leakage_025_annual.csv": ("ccv_leakage", "duplicate"),
    "coi_leakage_040_annual.csv": ("ccv_leakage", "duplicate"),
    "coi_leakage_050_annual.csv": ("ccv_leakage", "duplicate"),
    "coi_stuck_010_annual.csv": ("ccv_stuck", "10 %"),
    "coi_stuck_025_annual.csv": ("ccv_stuck", "25 %"),
    "coi_stuck_050_annual.csv": ("ccv_stuck", "50 %"),
    "coi_stuck_075_annual.csv": ("ccv_stuck", "75 %"),
    "damper_stuck_010_annual.csv": ("stuck_oa_damper", "10 %"),
    "damper_stuck_025_annual.csv": ("stuck_oa_damper", "25 %"),
    "damper_stuck_075_annual.csv": ("stuck_oa_damper", "75 %"),
    "damper_stuck_100_annual_short.csv": ("stuck_oa_damper", "100 % (Apr-Oct)"),
    "oa_bias_-4_annual.csv": ("oat_sensor_bias", "see note: 4 files identical"),
    "oa_bias_-2_annual.csv": ("oat_sensor_bias", "duplicate"),
    "oa_bias_2_annual.csv": ("oat_sensor_bias", "duplicate"),
    "oa_bias_4_annual.csv": ("oat_sensor_bias", "duplicate"),
}


def f_to_c(x):
    return (x - 32.0) * 5.0 / 9.0


def file_hash(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def process_file(path):
    """Raw 1-minute CSV -> 15-minute occupied data with our column names."""
    raw = pd.read_csv(path, usecols=USECOLS, parse_dates=["Datetime"])
    raw = raw.set_index("Datetime").sort_index()
    sp = raw["SA_SP"]
    sp_inwc = sp / PA_PER_INWC if sp.median() > 50 else sp      # Pa in some files, inWC in others
    df = pd.DataFrame({
        "oat": f_to_c(raw["OA_TEMP"]), "rat": f_to_c(raw["RA_TEMP"]), "mat": f_to_c(raw["MA_TEMP"]),
        "sat": f_to_c(raw["SA_TEMP"]), "sat_sp": f_to_c(raw["SA_TEMPSPT"]),
        "oa_damper": raw["OA_DMPR"] * 100, "damper_cmd": raw["OA_DMPR_DM"] * 100,
        "ccv": raw["CHWC_VLV"] * 100, "ccv_cmd": raw["CHWC_VLV_DM"] * 100,
        "fan_speed": raw["SF_CS"] * 100, "static_pressure": sp_inwc, "occ": raw["SYS_CTL"],
    })
    blocks = df.resample("15min")
    out = blocks.mean()
    out["occ_frac"] = blocks["occ"].mean()
    out["n"] = blocks["occ"].count()
    out = out[(out["occ_frac"] >= 0.999) & (out["n"] == 15)].drop(columns=["occ", "occ_frac", "n"])
    out["hcv"] = 0.0
    out["static_sp"] = STATIC_SP_INWC
    for c in ("oa_damper", "damper_cmd", "ccv", "ccv_cmd", "fan_speed"):
        out[c] = out[c].clip(0, 100)
    out = out.reset_index().rename(columns={"Datetime": "timestamp"})
    return out


def load_lbnl(raw_dir=RAW_DIR, cache_dir=CACHE_DIR, verbose=True):
    """Load all scenarios (duplicates once). Returns one DataFrame with columns
    scenario, file, fault_type, severity, fault (0/1) and the AHU signals.
    Processed files are cached as small CSV files (portable between pandas versions)."""
    raw_dir, cache_dir = Path(raw_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_file = cache_dir / "index.json"
    index = json.loads(index_file.read_text()) if index_file.exists() else {}
    if not raw_dir.exists() and not index:
        raise FileNotFoundError(f"{raw_dir} not found: extract LBNL_FDD_Data_Sets_SDAHU.zip into data/")
    seen, frames, duplicates = {}, [], []
    for name, (fault_type, severity) in SCENARIOS.items():
        path = raw_dir / name
        entry = index.get(name)
        if path.exists() and (entry is None or entry.get("size") != path.stat().st_size):
            if verbose:
                print(f"  checking {name} ...")
            entry = {"md5": file_hash(path), "size": path.stat().st_size}
        if entry is None:
            continue
        if entry["md5"] in seen:
            duplicates.append((name, seen[entry["md5"]]))
            entry["duplicate_of"] = seen[entry["md5"]]
            index[name] = entry
            continue
        seen[entry["md5"]] = name
        cached = cache_dir / name.replace(".csv", "_15min.csv")
        if cached.exists():
            part = pd.read_csv(cached, parse_dates=["timestamp"])
        else:
            if verbose:
                print(f"  processing {name} (only the first time) ...")
            part = process_file(path)
            part.to_csv(cached, index=False, float_format="%.4f")
        index[name] = entry
        part["file"], part["fault_type"], part["severity"] = name, fault_type, severity
        part["fault"] = int(fault_type != "none")
        frames.append(part)
    index_file.write_text(json.dumps(index, indent=1))
    df = pd.concat(frames, ignore_index=True)
    df = assign_scenarios(df)
    df.attrs["synthetic"] = False
    df.attrs["source"] = SOURCE
    df.attrs["duplicates"] = duplicates
    return df


def assign_scenarios(df):
    """Fault-free file split by day of month (all seasons in every part):
    days 1-20 -> train, 21-25 -> val, 26-31 -> test_normal. Every fault file -> test_<file>."""
    day = pd.to_datetime(df["timestamp"]).dt.day
    sc = np.where(day <= 20, "train", np.where(day <= 25, "val", "test_normal"))
    faulty = df["fault"] == 1
    df["scenario"] = np.where(faulty, "test_" + df["file"].str.replace("_annual", "").str.replace(".csv", ""), sc)
    return df
