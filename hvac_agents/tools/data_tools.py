"""
data_tools.py - Data loading, demo data and the validate_data tool.

Area 6: data, models and evaluation (owner: Mary Ann Mastri). Code review: docs/CODE_REVIEW_AREA6.md
The agents only need these function names and return formats.

Demo data: the LBNL and BDG2 files are downloaded in Colab. Until then we use
SYNTHETIC data with labeled faults that copies their structure. Every result
made with it says "SYNTHETIC" (proposal rule: label simulated data everywhere).
"""
import numpy as np
import pandas as pd

from .. import config

# Columns the Analyst needs (degC, %, inWC). Map real dataset names to these.
AHU_COLUMNS = ["timestamp", "oat", "rat", "mat", "sat", "sat_sp", "oa_damper",
               "ccv", "hcv", "fan_speed", "static_pressure", "static_sp"]
ENERGY_COLUMNS = ["timestamp", "kwh", "air_temp"]

# The real LBNL dataset is loaded by tools/lbnl.py (it renames, converts units and removes duplicates).
# Detectors trained on LBNL data also need these columns (valve/damper command signals):
LBNL_EXTRA_COLUMNS = ["ccv_cmd", "damper_cmd"]

FAULT_TYPES = ["stuck_oa_damper", "slipping_fan_belt", "ccv_stuck_closed", "mat_sensor_bias"]


# ============================================================ demo data (synthetic)
def make_ahu_data(seed=config.SEED):
    """Synthetic single-duct AHU, 15-minute data, occupied hours only.

    Scenarios (column `scenario`) so evaluation is split BY SCENARIO, never at random:
      train (10 fault-free days), val (3 fault-free days),
      test_normal (2 fault-free days), and one test day per fault type.
    Columns `fault` (0/1) and `fault_type` are the ground-truth labels.
    """
    rng = np.random.default_rng(seed)
    plan = ([("train", None)] * 10 + [("val", None)] * 3 + [("test_normal", None)] * 2
            + [(f"test_{f}", f) for f in FAULT_TYPES])
    frames = []
    day0 = pd.Timestamp("2026-06-01")
    for d, (scenario, fault) in enumerate(plan):
        ts = pd.date_range(day0 + pd.Timedelta(days=d, hours=6), periods=56, freq="15min")  # 06:00-19:45
        hour = ts.hour + ts.minute / 60
        day_shift = rng.normal(0, 2)
        oat = 24 + day_shift + 7 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.3, len(ts))
        # some cooler mornings so the economizer is used
        oat = np.where(hour < 9, oat - 6, oat)
        rat = 23.5 + rng.normal(0, 0.3, len(ts))
        sat_sp = np.full(len(ts), 13.0)
        damper = np.where(oat < 18, np.clip(100 * (rat - 13) / np.maximum(rat - oat, 0.5), 20, 100), 20.0)
        mat = damper / 100 * oat + (1 - damper / 100) * rat
        load = np.clip((oat - 15) / 20, 0, 1)
        fan = 55 + 25 * load + rng.normal(0, 1.5, len(ts))
        static_sp = np.full(len(ts), 1.5)
        static = static_sp + rng.normal(0, 0.04, len(ts))
        sat = sat_sp + rng.normal(0, 0.25, len(ts))
        mat_sensor = mat + rng.normal(0, 0.2, len(ts))
        label = np.zeros(len(ts), dtype=int)

        if fault:
            active = (hour >= 10) & (hour < 16)          # fault active 10:00-16:00
            label[active] = 1
            if fault == "stuck_oa_damper":                # damper stuck at 85 % in hot weather
                damper = np.where(active, 85.0, damper)
                mat_true = damper / 100 * oat + (1 - damper / 100) * rat
                mat_sensor = np.where(active, mat_true + rng.normal(0, 0.2, len(ts)), mat_sensor)
                sat = np.where(active, sat_sp + np.clip(mat_true - mat, 0, None) * 0.6, sat)
                ccv_extra = np.where(active, 100.0, 0.0)
            elif fault == "slipping_fan_belt":           # fan works harder, pressure drops
                fan = np.where(active, np.clip(fan + 30, 0, 100), fan)
                static = np.where(active, static_sp - 0.6 + rng.normal(0, 0.05, len(ts)), static)
                sat = np.where(active, sat + 1.0, sat)
            elif fault == "ccv_stuck_closed":            # valve command 100 % but no cooling
                sat = np.where(active, mat - 1.0 + rng.normal(0, 0.3, len(ts)), sat)
            elif fault == "mat_sensor_bias":             # mixed-air sensor reads 4 degC too low
                mat_sensor = np.where(active, mat_sensor - 4.0, mat_sensor)
        ccv = np.clip((mat - sat_sp) * 10, 0, 100)
        if fault == "ccv_stuck_closed":
            ccv = np.where(label == 1, 100.0, ccv)
        if fault == "stuck_oa_damper":
            ccv = np.maximum(ccv, ccv_extra)
        hcv = np.clip((sat_sp - mat) * 8, 0, 100)
        frames.append(pd.DataFrame({
            "timestamp": ts, "oat": oat, "rat": rat, "mat": mat_sensor, "sat": sat, "sat_sp": sat_sp,
            "oa_damper": damper, "ccv": np.clip(ccv + rng.normal(0, 0.5, len(ts)), 0, 100), "hcv": hcv,
            "fan_speed": np.clip(fan, 0, 100), "static_pressure": static, "static_sp": static_sp,
            "fault": label, "fault_type": fault if fault else "none", "scenario": scenario,
        }))
    df = pd.concat(frames, ignore_index=True)
    df.attrs["synthetic"] = True
    df.attrs["source"] = "SYNTHETIC demo data with labeled faults"
    return df


def lbnl_available():
    from .lbnl import CACHE_DIR, RAW_DIR
    return RAW_DIR.exists() or (CACHE_DIR / "index.json").exists()


def load_dataset(name=None):
    """Returns (dataset_name, ahu_dataframe). name: "lbnl", "demo" or "auto"."""
    name = name or config.DATASET
    if name == "auto":
        name = "lbnl" if lbnl_available() else "demo"
    if name == "lbnl":
        from .lbnl import load_lbnl
        return "lbnl", load_lbnl()
    return "demo", make_ahu_data()


def make_energy_data(seed=config.SEED):
    """Synthetic hourly building meter (2016 = training year, 2017 = test year)
    with outdoor air temperature, like a BDG2 building + weather file.
    A labeled anomaly (+60 % at night: equipment left running) is injected 2017-08-07 to 2017-08-20."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2016-01-01", "2017-12-31 23:00", freq="h")
    doy, hour = ts.dayofyear.values, ts.hour.values
    temp = 21 - 8 * np.cos((doy - 15) / 365 * 2 * np.pi) + 5 * np.sin((hour - 9) / 24 * 2 * np.pi) \
        + rng.normal(0, 1.5, len(ts))
    workday = (ts.dayofweek.values < 5) & (hour >= 7) & (hour < 19)
    kwh = 60 + 45 * workday + 3.5 * np.clip(temp - 18, 0, None) + 2 * np.clip(12 - temp, 0, None) \
        + rng.normal(0, 4, len(ts))
    label = ((ts >= "2017-08-07") & (ts < "2017-08-21")).astype(int)
    night = (hour < 6) | (hour >= 20)
    kwh = np.where((label == 1) & night, kwh * 1.60, kwh)
    df = pd.DataFrame({"timestamp": ts, "kwh": kwh, "air_temp": temp, "anomaly": label})
    df.attrs["synthetic"] = True
    df.attrs["source"] = "SYNTHETIC demo energy data (BDG2 not loaded yet)"
    return df


def load_csv(path, column_map=None):
    """Load a real CSV (e.g. LBNL or BDG2) and rename columns to our names."""
    df = pd.read_csv(path)
    if column_map:
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    df.attrs["synthetic"] = False
    return df


# ============================================================ tool: validate_data
def validate_data(df, kind="ahu", min_rows=20, extra_columns=()):
    """First step on any dataset branch. NEVER fills missing values silently.

    Returns {"ok": bool, "reasons": [...], "warnings": [...], "rows": n, "start", "end"}.
    ok=False stops that branch (the Analyst answers NACK with the reasons).
    """
    reasons, warnings = [], []
    required = (AHU_COLUMNS + list(extra_columns)) if kind == "ahu" else ENERGY_COLUMNS
    if df is None or len(df) == 0:
        return {"ok": False, "reasons": ["dataset is empty"], "warnings": [], "rows": 0}

    missing = [c for c in required if c not in df.columns]
    if missing:
        reasons.append(f"missing required columns: {missing}")
        if set(missing) & set(extra_columns):
            reasons.append("the fault model was trained on the LBNL dataset and needs the valve/damper "
                           "command columns (ccv_cmd, damper_cmd)")

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        bad = int(ts.isna().sum())
        if bad:
            reasons.append(f"{bad} timestamps cannot be read")
        # several files/units in one table (column "scenario"): check time order inside each one
        groups = df["scenario"] if "scenario" in df.columns else pd.Series(0, index=df.index)
        dup = int(pd.DataFrame({"g": groups, "t": ts}).duplicated().sum())
        if dup:
            reasons.append(f"{dup} duplicated timestamps")
        if not all(t.dropna().is_monotonic_increasing for _, t in ts.groupby(groups.values)):
            reasons.append("timestamps are out of order")
        start, end = str(ts.min()), str(ts.max())
    else:
        start = end = None

    present = [c for c in required if c in df.columns and c != "timestamp"]
    for c in present:
        if not pd.api.types.is_numeric_dtype(df[c]):
            reasons.append(f"column {c} is not numeric")
            continue
        frac = df[c].isna().mean()
        if frac > 0.10:
            reasons.append(f"column {c} has {frac:.0%} missing values (limit 10%)")
        elif frac > 0:
            warnings.append(f"column {c} has {frac:.1%} missing values (rows kept as missing, not filled)")

    # unit checks: temperatures must be degC
    if kind == "ahu":
        for c in ["oat", "rat", "mat", "sat", "sat_sp"]:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                med = df[c].median()
                if med > 40:
                    reasons.append(f"column {c} median {med:.1f} looks like Fahrenheit, expected degC")
        for c in ["oa_damper", "ccv", "hcv", "fan_speed"]:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                if df[c].max() > 100.5 or df[c].min() < -0.5:
                    reasons.append(f"column {c} outside 0-100 %")
    else:
        if "kwh" in df.columns and pd.api.types.is_numeric_dtype(df["kwh"]) and (df["kwh"] < 0).any():
            reasons.append("negative energy readings")
        if "air_temp" in df.columns and pd.api.types.is_numeric_dtype(df["air_temp"]) \
                and df["air_temp"].median() > 45:
            reasons.append("air_temp looks like Fahrenheit, expected degC")

    if len(df) < min_rows:
        reasons.append(f"only {len(df)} rows (need at least {min_rows})")

    return {"ok": not reasons, "reasons": reasons, "warnings": warnings, "rows": int(len(df)),
            "start": start, "end": end, "synthetic": bool(df.attrs.get("synthetic", False)),
            "source": df.attrs.get("source", "SYNTHETIC demo data" if df.attrs.get("synthetic") else "user data")}
