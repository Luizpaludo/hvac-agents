"""
anomaly.py - The detect_anomalies tool: Isolation Forest baseline + deep learning autoencoder.

Area 6: data, models and evaluation (owner: Mary Ann Mastri). Version 2 adds the real LBNL dataset.
Code review: docs/CODE_REVIEW_AREA6.md

How it works
  1. Features. Two data types are supported:
     - LBNL dataset (has valve/damper COMMAND and POSITION): supply-temperature error, command-vs-
       position gaps for the cooling valve and the outdoor-air damper, and two LEARNED residuals:
         ccv_resid = cooling-valve command minus what fault-free operation needs in the same conditions
         mat_resid = mixed-air temperature minus what the outdoor/return mix should give
       The residual models are fitted on fault-free TRAINING data only.
     - Demo (synthetic) data: physics-based residuals (as in version 1).
  2. Both detectors learn ONLY fault-free data. tau starts at the 99th percentile of the fault-free
     validation scores (proposal).
  3. Rows above tau are grouped into events (>= 1 hour of consecutive 15-minute rows, no gap in time).
  4. Each event gets a rule-based fault hint. Rule thresholds also come from fault-free validation data
     only (never from the fault files), so the test stays honest.
  5. Events of the same fault type are merged into one finding (count, hours, first/last, worst score).
Scores are ANOMALY scores, not failure probabilities.

Autoencoder: PyTorch if installed (Colab), otherwise the same 3-unit-bottleneck network in scikit-learn.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler

from .. import config

DEMO_FEATURES = ["sat_err", "static_err", "mat_pos", "fan_speed", "ccv", "oa_damper", "oat", "mix_gap"]
LBNL_FEATURES = ["sat_err", "ccv_gap", "damper_gap", "ccv_resid", "mat_resid", "fan_speed", "ccv",
                 "oa_damper", "oat", "mix_gap"]
FEATURES = DEMO_FEATURES   # kept for backward compatibility


def has_commands(df):
    return {"ccv_cmd", "damper_cmd"}.issubset(df.columns)


class FeatureBuilder:
    """Turns raw AHU signals into features that make faults visible."""

    def fit(self, train_df):
        self.mode = "lbnl" if has_commands(train_df) else "demo"
        if self.mode == "lbnl":
            t = train_df.dropna(subset=["mat", "rat", "oat", "fan_speed", "ccv_cmd", "oa_damper"])
            self.ccv_cols = ["mat", "rat", "oat", "fan_speed", "sat_sp"]
            self.ccv_model = HistGradientBoostingRegressor(max_iter=200, random_state=config.SEED).fit(
                t[self.ccv_cols], t["ccv_cmd"])
            self.mat_cols = ["oat", "rat", "oa_damper"]
            self.mat_model = HistGradientBoostingRegressor(max_iter=200, random_state=config.SEED).fit(
                t[self.mat_cols], t["mat"])
        return self

    @property
    def names(self):
        return LBNL_FEATURES if self.mode == "lbnl" else DEMO_FEATURES

    def transform(self, df):
        f = pd.DataFrame(index=df.index)
        f["sat_err"] = df["sat"] - df["sat_sp"]
        f["fan_speed"] = df["fan_speed"]
        f["ccv"] = df["ccv"]
        f["oa_damper"] = df["oa_damper"]
        f["oat"] = df["oat"]
        f["mix_gap"] = df["mat"] - df["sat"]                       # cooling done by the coil
        if self.mode == "lbnl":
            if not has_commands(df):
                raise ValueError("this detector was trained on LBNL data and needs ccv_cmd and damper_cmd")
            f["ccv_gap"] = df["ccv"] - df["ccv_cmd"]                # position minus command
            f["damper_gap"] = df["oa_damper"] - df["damper_cmd"]
            f["ccv_resid"] = df["ccv_cmd"] - self.ccv_model.predict(df[self.ccv_cols])
            f["mat_resid"] = df["mat"] - self.mat_model.predict(df[self.mat_cols])
        else:
            f["static_err"] = df["static_pressure"] - df["static_sp"]
            span = (df["rat"] - df["oat"]).where((df["rat"] - df["oat"]).abs() > 0.5, 0.5)
            f["mat_pos"] = ((df["mat"] - df["oat"]) / span).clip(-3, 4)
        return f[self.names].astype(float)


# ------------------------------------------------------------------ autoencoders
class _TorchAE:
    def __init__(self, n, seed):
        import torch
        torch.manual_seed(seed)
        self.torch = torch
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3), torch.nn.Tanh(),
            torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, n))

    def fit(self, x, epochs=400):
        t = self.torch
        xt = t.tensor(x, dtype=t.float32)
        opt = t.optim.Adam(self.net.parameters(), lr=0.01)
        for _ in range(epochs):
            opt.zero_grad()
            loss = t.nn.functional.mse_loss(self.net(xt), xt)
            loss.backward()
            opt.step()
        return self

    def predict(self, x):
        with self.torch.no_grad():
            return self.net(self.torch.tensor(x, dtype=self.torch.float32)).numpy()


class _SklearnAE:
    def __init__(self, n, seed):
        from sklearn.neural_network import MLPRegressor
        self.net = MLPRegressor(hidden_layer_sizes=(8, 3, 8), activation="tanh", max_iter=2000,
                                random_state=seed, learning_rate_init=0.01)

    def fit(self, x):
        self.net.fit(x, x)
        return self

    def predict(self, x):
        return self.net.predict(x)


def _make_ae(n, seed):
    try:
        import torch  # noqa: F401
        return _TorchAE(n, seed), "pytorch"
    except ImportError:
        return _SklearnAE(n, seed), "sklearn-mlp"


# ------------------------------------------------------------------ detector
class AnomalyDetector:
    """kind = "autoencoder" (deep learning) or "isolation_forest" (baseline)."""

    def __init__(self, kind="autoencoder", percentile=config.AE_PERCENTILE, seed=config.SEED):
        self.kind, self.percentile, self.seed = kind, percentile, seed
        self.scaler = StandardScaler()
        self.model = None
        self.tau = None
        self.backend = None

    def fit(self, train_df, val_df, features=None):
        self.features = features or FeatureBuilder().fit(train_df)
        x = self.scaler.fit_transform(self.features.transform(train_df).dropna())
        if self.kind == "autoencoder":
            self.model, self.backend = _make_ae(x.shape[1], self.seed)
            self.model.fit(x)
        else:
            self.model = IsolationForest(n_estimators=200, random_state=self.seed).fit(x)
            self.backend = "sklearn-isolation-forest"
        self.val_scores = self.score(val_df)
        self.val_scores = self.val_scores[~np.isnan(self.val_scores)]
        self.tau = float(np.percentile(self.val_scores, self.percentile))
        # rule thresholds from fault-free validation data (99.5th percentile of |feature|)
        fv = self.features.transform(val_df).dropna()
        self.thresholds = {c: float(np.percentile(fv[c].abs(), 99.5)) for c in fv.columns}
        for c in ("ccv_gap", "damper_gap"):
            if c in self.thresholds:
                self.thresholds[c] = max(self.thresholds[c], 3.0)    # at least 3 % (sensor noise)
        return self

    def set_percentile(self, p):
        """Used by the reward-based threshold tuning (RL element)."""
        self.percentile = p
        self.tau = float(np.percentile(self.val_scores, p))

    def score(self, df):
        """Anomaly score per row. Rows with missing values get NaN (never filled in) and are not flagged."""
        f = self.features.transform(df)
        ok = f.notna().all(axis=1).to_numpy()
        out = np.full(len(f), np.nan)
        if ok.any():
            x = self.scaler.transform(f[ok])
            if self.kind == "autoencoder":
                out[ok] = np.mean((x - self.model.predict(x)) ** 2, axis=1)   # e_t = mean (x - xhat)^2
            else:
                out[ok] = -self.model.score_samples(x)                         # higher = more abnormal
        return out


def train_detectors(ahu_df):
    """Train both detectors on the fault-free scenarios of the dataset."""
    train = ahu_df[ahu_df["scenario"] == "train"]
    val = ahu_df[ahu_df["scenario"] == "val"]
    if len(train) < 50 or len(val) < 20:
        raise ValueError("not enough fault-free data to train (need scenarios 'train' and 'val')")
    features = FeatureBuilder().fit(train)          # same residual models for both detectors
    return {"autoencoder": AnomalyDetector("autoencoder").fit(train, val, features),
            "isolation_forest": AnomalyDetector("isolation_forest").fit(train, val, features)}


# ------------------------------------------------------------------ diagnosis rules
def diagnose(rows, feats=None, th=None):
    """Rule-based fault hint on the median of the event rows -> (fault_type, rule text)."""
    m = rows.median(numeric_only=True)
    if feats is not None and "ccv_gap" in feats.columns:
        f = feats.median()
        leak = ((rows["ccv_cmd"] < 2) & (rows["ccv"] > 3)).mean()
        if abs(f["damper_gap"]) > th["damper_gap"]:
            return "stuck_oa_damper", (f"outdoor air damper is at {m['oa_damper']:.0f}% but the controller asks "
                                       f"for {m['damper_cmd']:.0f}% (position does not follow the command)")
        # stuck: the position stays still while the command moves; leak: small opening when told to close
        frozen = rows["ccv"].std() < 1.0 and (rows["ccv_cmd"].max() - rows["ccv_cmd"].min()) > 5
        if abs(f["ccv_gap"]) > th["ccv_gap"] and (frozen or m["ccv"] > 20):
            return "ccv_stuck", (f"cooling valve stays at {m['ccv']:.0f}% while the controller asks for "
                                 f"{m['ccv_cmd']:.0f}% (position does not follow the command)")
        if leak > 0.5:
            return "ccv_leakage", (f"cooling valve shows {m['ccv']:.0f}% open while the controller asks for "
                                   f"{m['ccv_cmd']:.0f}% (valve does not close fully: leaking, or stuck slightly open)")
        if abs(f["ccv_gap"]) > th["ccv_gap"]:
            return "ccv_stuck", (f"cooling valve stays at {m['ccv']:.0f}% while the controller asks for "
                                 f"{m['ccv_cmd']:.0f}% (position does not follow the command)")
        if abs(f["ccv_resid"]) > th["ccv_resid"]:
            word = "more" if f["ccv_resid"] > 0 else "less"
            return "sat_sensor_bias", (f"controller opens the cooling valve {abs(f['ccv_resid']):.0f}% {word} than "
                                       f"normal for these conditions while supply air reads {m['sat']:.1f} C at "
                                       f"setpoint; the supply air sensor may be biased")
        if abs(f["mat_resid"]) > th["mat_resid"]:
            return "oat_sensor_bias", (f"mixed air {m['mat']:.1f} C is {f['mat_resid']:+.1f} C off what the outdoor "
                                       f"({m['oat']:.1f} C) / return ({m['rat']:.1f} C) mix should give; an outdoor or "
                                       f"mixed air sensor may be biased")
        return "unknown", "abnormal pattern with no matching rule"
    # demo (synthetic) data rules
    lo, hi = min(m["oat"], m["rat"]) - 1.5, max(m["oat"], m["rat"]) + 1.5
    if m["mat"] < lo or m["mat"] > hi:
        return "mat_sensor_bias", f"mixed air {m['mat']:.1f} C is outside outdoor/return range [{lo:.1f}, {hi:.1f}] C"
    if m["fan_speed"] > 85 and m["static_pressure"] < m["static_sp"] - 0.3:
        return "slipping_fan_belt", (f"fan speed {m['fan_speed']:.0f}% but static pressure "
                                     f"{m['static_pressure']:.2f} below setpoint {m['static_sp']:.2f} inWC")
    if m["oa_damper"] > 60 and m["oat"] > m["rat"]:
        return "stuck_oa_damper", (f"outdoor air damper {m['oa_damper']:.0f}% open while outdoor air "
                                   f"{m['oat']:.1f} C is warmer than return air {m['rat']:.1f} C")
    if m["ccv"] > 90 and m["sat"] > m["sat_sp"] + 2:
        return "ccv_stuck_closed", (f"cooling valve {m['ccv']:.0f}% open but supply air {m['sat']:.1f} C "
                                    f"is above setpoint {m['sat_sp']:.1f} C")
    return "unknown", "abnormal pattern with no matching rule"


# ------------------------------------------------------------------ the tool
def detect_anomalies(df, detectors, kind="autoencoder"):
    """Tool used by the Analyst. Falls back to Isolation Forest if the autoencoder
    is missing or fails; if both fail it raises (-> human review)."""
    used, errors = None, []
    df = df.reset_index(drop=True)
    for k in ([kind, "isolation_forest"] if kind != "isolation_forest" else ["isolation_forest"]):
        det = (detectors or {}).get(k)
        try:
            if det is None:
                raise RuntimeError(f"{k} model not available")
            scores = det.score(df)
            used = det
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{k}: {exc}")
    if used is None:
        raise RuntimeError("both detectors failed -> human review: " + "; ".join(errors))

    skipped = int(np.isnan(scores).sum())
    flags = np.nan_to_num(scores, nan=-np.inf) > used.tau
    feats = used.features.transform(df) if used.features.mode == "lbnl" else None
    events = []
    for start, end in _group_events(flags, df["timestamp"]):
        rows = df.iloc[start:end + 1]
        fault_type, rule = diagnose(rows, feats.iloc[start:end + 1] if feats is not None else None,
                                    getattr(used, "thresholds", None))
        events.append({"fault_type": fault_type, "rule": rule, "start": str(rows["timestamp"].iloc[0]),
                       "source": str(rows["scenario"].iloc[0]) if "scenario" in rows else None,
                       "end": str(rows["timestamp"].iloc[-1]), "rows": int(end - start + 1),
                       "ratio": float(np.nanmean(scores[start:end + 1]) / used.tau)})
    findings = _merge_events(events, used.tau)
    return {"model": used.kind, "backend": used.backend, "tau": round(used.tau, 4),
            "percentile": used.percentile, "flags": flags, "scores": scores, "events": events,
            "findings": findings, "fallback_used": used.kind != kind, "errors": errors,
            "skipped_rows_missing_values": skipped}


def _merge_events(events, tau):
    """One finding per fault type: how often, how long, first/last time, worst event."""
    findings = []
    by_type = {}
    for e in events:
        by_type.setdefault(e["fault_type"], []).append(e)
    for fault_type, evs in sorted(by_type.items(), key=lambda kv: -sum(e["rows"] for e in kv[1])):
        evs = sorted(evs, key=lambda e: e["start"])
        worst = max(evs, key=lambda e: e["ratio"])
        rows = sum(e["rows"] for e in evs)
        ratio = float(np.average([e["ratio"] for e in evs], weights=[e["rows"] for e in evs]))
        severity = "high" if ratio >= 3 else "medium" if ratio >= 1.5 else "low"
        findings.append({
            "kind": "maintenance", "fault_type": fault_type, "severity": severity,
            "start": evs[0]["start"], "end": evs[-1]["end"],
            "metrics": {"events": len(evs), "rows": rows, "hours": round(rows * 0.25, 1),
                        "files": len({e.get("source") for e in evs}),
                        "score_to_tau": round(ratio, 2), "worst_score_to_tau": round(worst["ratio"], 2),
                        "tau": round(tau, 4), "rule": worst["rule"]},
            "limitations": "Anomaly score, not a failure probability. Fault type is a rule-based hint; "
                           "several causes are possible (NIST/ASHRAE).",
        })
    return findings


def _group_events(flags, timestamps=None, min_rows=config.MIN_EVENT_ROWS, max_gap=1, step_min=15):
    """Consecutive flagged rows (gaps of 1 row allowed) that are also consecutive in TIME."""
    flags = np.asarray(flags)
    if timestamps is not None:
        t = pd.to_datetime(pd.Series(timestamps)).to_numpy()
        d = np.diff(t)
        jump = np.r_[False, (d > np.timedelta64(int(step_min * (max_gap + 1)), "m")) | (d <= np.timedelta64(0, "m"))]
    else:
        jump = np.zeros(len(flags), dtype=bool)
    events, start, gap, last = [], None, 0, None
    for i in range(len(flags) + max_gap + 1):
        f = bool(flags[i]) if i < len(flags) else False
        if i < len(flags) and jump[i] and start is not None:      # night / weekend gap ends the event
            if last - start + 1 >= min_rows:
                events.append((start, last))
            start, gap = None, 0
        if f:
            start = i if start is None else start
            gap, last = 0, i
        elif start is not None:
            gap += 1
            if gap > max_gap:
                if last - start + 1 >= min_rows:
                    events.append((start, last))
                start, gap = None, 0
    return events


# ------------------------------------------------------------------ evaluation
def evaluate(df, flags):
    """Row-level precision, recall, F1, FPR, plus per-scenario flag rate and time-to-alert."""
    df = df.reset_index(drop=True)
    y = df["fault"].values.astype(bool)
    p = np.asarray(flags).astype(bool)
    tp, fp = int((p & y).sum()), int((p & ~y).sum())
    fn, tn = int((~p & y).sum()), int((~p & ~y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
           "fpr": round(fp / (fp + tn), 3) if fp + tn else 0.0, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "by_scenario": {}}
    for sc, part in df.groupby("scenario", sort=False):
        idx = part.index.to_numpy()
        yy, pp = y[idx], p[idx]
        entry = {"rows": int(len(idx)), "flag_rate": round(float(pp.mean()), 3),
                 "recall": round(float((pp & yy).sum() / yy.sum()), 3) if yy.sum() else None,
                 "false_alarm_rows": int((pp & ~yy).sum())}
        if yy.any() and (pp & yy).any():
            entry["time_to_alert_min"] = int((np.argmax(pp & yy) - np.argmax(yy)) * 15)
        out["by_scenario"][sc] = entry
    return out
