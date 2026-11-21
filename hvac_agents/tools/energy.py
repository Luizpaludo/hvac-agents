"""
energy.py - The energy_baseline tool: weather-aware baseline + deviation leads.

Area 6: data, models and evaluation (owner: Mary Ann Mastri). Code review: docs/CODE_REVIEW_AREA6.md

Baseline model (a standard approach for building energy, like ASHRAE Guideline 14):
  kWh ~ hour-of-week schedule + cooling degrees above 18 C + heating degrees below 12 C
fitted by least squares on the training year. Quality metrics on the test year:
  NMBE     = sum(y - yhat) / ((n - 1) * mean(y)) * 100      target |NMBE| <= 10 %
  CV(RMSE) = sqrt(sum((y - yhat)^2) / (n - 1)) / mean(y) * 100   target <= 30 % hourly
Deviation % = (E_actual - E_baseline) / E_baseline * 100, per day.
Days above ENERGY_DEVIATION_PCT for 2+ days in a row become LEADS, not proof.
"""
import numpy as np
import pandas as pd

from .. import config


def _design(df):
    ts = pd.to_datetime(df["timestamp"])
    how = (ts.dt.dayofweek * 24 + ts.dt.hour).to_numpy()
    x = np.zeros((len(df), 168 + 2))
    x[np.arange(len(df)), how] = 1.0
    t = df["air_temp"].to_numpy(dtype=float)
    x[:, 168] = np.clip(t - 18, 0, None)
    x[:, 169] = np.clip(12 - t, 0, None)
    return x


def nmbe(y, yhat):
    return float(np.sum(y - yhat) / ((len(y) - 1) * np.mean(y)) * 100)


def cv_rmse(y, yhat):
    return float(np.sqrt(np.sum((y - yhat) ** 2) / (len(y) - 1)) / np.mean(y) * 100)


def energy_baseline(df, train_end="2016-12-31 23:00", test_start="2017-01-01"):
    """Tool used by the Analyst for energy requests."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    train = df[df["timestamp"] <= train_end].dropna(subset=["kwh", "air_temp"])
    test = df[df["timestamp"] >= test_start].dropna(subset=["kwh", "air_temp"])
    history_days = (train["timestamp"].max() - train["timestamp"].min()).days if len(train) else 0
    if history_days < config.MIN_ENERGY_HISTORY_DAYS:
        return {"ok": False, "reason": f"only {history_days} days of history; need "
                f"{config.MIN_ENERGY_HISTORY_DAYS} -> no baseline reported", "findings": []}
    if len(test) == 0:
        return {"ok": False, "reason": "no test period after the training year", "findings": []}

    coef, *_ = np.linalg.lstsq(_design(train), train["kwh"].to_numpy(), rcond=None)
    test = test.assign(baseline=_design(test) @ coef)
    y, yhat = test["kwh"].to_numpy(), test["baseline"].to_numpy()
    metrics = {"cv_rmse_pct": round(cv_rmse(y, yhat), 2), "nmbe_pct": round(nmbe(y, yhat), 2),
               "train_days": int(history_days), "test_hours": int(len(test))}

    daily = test.set_index("timestamp")[["kwh", "baseline"]].resample("D").sum()
    daily["deviation_pct"] = (daily["kwh"] - daily["baseline"]) / daily["baseline"] * 100
    high = daily["deviation_pct"] > config.ENERGY_DEVIATION_PCT
    findings = []
    run_start = None
    days = list(daily.index) + [None]
    flags = list(high) + [False]
    for i, f in enumerate(flags):
        if f and run_start is None:
            run_start = i
        elif not f and run_start is not None:
            if i - run_start >= 2:
                part = daily.iloc[run_start:i]
                extra = float(part["kwh"].sum() - part["baseline"].sum())
                dev = float(extra / part["baseline"].sum() * 100)
                findings.append({
                    "kind": "energy", "fault_type": "excess_energy_use",
                    "severity": "high" if dev >= 30 else "medium" if dev >= 20 else "low",
                    "start": str(days[run_start].date()), "end": str(days[i - 1].date()),
                    "metrics": {"days": int(i - run_start), "deviation_pct": round(dev, 1),
                                "extra_kwh": round(extra, 0), "cv_rmse_pct": metrics["cv_rmse_pct"],
                                "nmbe_pct": metrics["nmbe_pct"]},
                    "limitations": "Deviation from a weather-aware baseline is a lead, not proof of "
                                   "inefficiency. Check schedules and occupancy first.",
                })
            run_start = None
    if "anomaly" in test.columns:   # ground truth exists only in labeled/synthetic data
        lab = test.set_index("timestamp")["anomaly"].resample("D").max()
        hit_days = set(daily.index[high]) & set(lab.index[lab == 1])
        metrics["labeled_anomaly_days"] = int((lab == 1).sum())
        metrics["labeled_days_flagged"] = len(hit_days)
    return {"ok": True, "metrics": metrics, "findings": findings,
            "meets_targets": metrics["cv_rmse_pct"] <= 30 and abs(metrics["nmbe_pct"]) <= 10}
