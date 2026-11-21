"""
rl.py - Reinforcement learning experiments (G6: learning from feedback with a measurable reward).

1. replay_experiment(): 5 rounds of SIMULATED reviewer feedback. The learned
   ranker (Q-values + 10 % exploration) is compared with random ranking.
   Proposal target: mean reward in round 5 at least 0.15 above round 1.
   The simulated reviewer has hidden preferences (below). In the real system
   these rewards come from engineers through HVACSystem.feedback().
2. tune_threshold(): the detector threshold tau is tuned to MAXIMISE average
   reward (+1 detected fault, -1 missed fault, -0.5 false alarm) on a separate
   labeled tuning set (different random seed), then checked on the test set.
All replay feedback is labeled SIMULATED.
"""
import random

import numpy as np
import pandas as pd

from . import config
from .agents.ranker import LIBRARY, Ranker
from .memory import Memory
from .tools.anomaly import detect_anomalies
from .tools.data_tools import make_ahu_data

# Probability that the (simulated) engineer accepts each candidate.
# None = the candidate breaks a comfort/safety limit -> reward -1.
SIMULATED_PREFERENCES = {
    "dmp-1": 0.85, "dmp-2": 0.55, "dmp-3": 0.25,
    "blt-1": 0.85, "blt-2": 0.55, "blt-3": None,
    "ccv-1": 0.45, "ccv-2": 0.85, "ccv-3": 0.30,
    "mat-1": 0.40, "mat-2": 0.25, "mat-3": 0.80,
    "nrg-1": 0.85, "nrg-2": 0.50, "nrg-3": None,
    "unk-1": 0.60, "unk-2": 0.40, "unk-3": 0.30,
    "cst-1": 0.85, "cst-2": 0.60, "cst-3": 0.30,
    "lk-1": 0.50, "lk-2": 0.80, "lk-3": 0.35,
    "sab-1": 0.85, "sab-2": 0.45, "sab-3": 0.40,
    "oab-1": 0.80, "oab-2": 0.40, "oab-3": 0.55,
}


def simulated_reward(candidate_id, rng):
    p = SIMULATED_PREFERENCES[candidate_id]
    if p is None:
        return config.REWARDS["safety_violation"], "safety_violation"
    if rng.random() < p:
        return config.REWARDS["accepted"], "accepted"
    return config.REWARDS["rejected"], "rejected"


def replay_experiment(rounds=5, episodes_per_round=40, seed=config.SEED):
    """Returns per-round mean reward of the learned ranker vs random ranking."""
    rng = random.Random(seed)
    mem = Memory(url="", sqlite_path=":memory:")       # separate memory: does not touch real Q-values
    ranker = Ranker(mem, rng=random.Random(seed + 1))
    cats = list(LIBRARY)
    results = []
    for rnd in range(1, rounds + 1):
        learned, rand = [], []
        for _ in range(episodes_per_round):
            cat = rng.choice(cats)
            top = ranker.rank(cat)[0]                     # what the agent shows first
            r, _ = simulated_reward(top["id"], rng)
            ranker.update(cat, top["id"], r)              # Q(c) <- Q(c) + 0.2 (r - Q(c))
            learned.append(r)
            rc = rng.choice(LIBRARY[cat]["candidates"])  # baseline: random ranking
            rand.append(simulated_reward(rc["id"], rng)[0])
        results.append({"round": rnd, "learned_mean_reward": round(float(np.mean(learned)), 3),
                        "random_mean_reward": round(float(np.mean(rand)), 3)})
    gain = results[-1]["learned_mean_reward"] - results[0]["learned_mean_reward"]
    q_table = {cat: {c: round(q, 2) for c, (q, n) in mem.get_q(cat).items()} for cat in cats}
    return {"rounds": results, "gain_round5_vs_round1": round(gain, 3), "target_met": gain >= 0.15,
            "q_table": q_table, "note": "SIMULATED feedback"}


def replay_summary(seeds=10, **kw):
    """The replay is noisy (few trials per fault type), so it is repeated with 10 random seeds.
    Reports the average gain round 5 vs round 1 and learned vs random reward."""
    runs = [replay_experiment(seed=s, **kw) for s in range(seeds)]
    gains = [r["gain_round5_vs_round1"] for r in runs]
    rounds = []
    for i in range(len(runs[0]["rounds"])):
        rounds.append({"round": i + 1,
                       "learned_mean_reward": round(float(np.mean([r["rounds"][i]["learned_mean_reward"] for r in runs])), 3),
                       "random_mean_reward": round(float(np.mean([r["rounds"][i]["random_mean_reward"] for r in runs])), 3)})
    gain = float(np.mean(gains))
    return {"rounds": rounds, "gain_round5_vs_round1": round(gain, 3), "gain_per_seed": gains,
            "target_met": gain >= 0.15, "seeds": seeds, "q_table": runs[0]["q_table"],
            "single_run_seed42": replay_experiment(**kw)["gain_round5_vs_round1"], "note": "SIMULATED feedback"}


def _event_reward(df, findings):
    """+1 per fault scenario detected, -1 per fault scenario missed, -0.5 per false-alarm event."""
    ts = df["timestamp"].astype(str)
    reward, detected = 0.0, set()
    for f in findings:
        rows = df[(ts >= f["start"]) & (ts <= f["end"])]
        faulty = rows[rows["fault"] == 1]
        if len(faulty):
            detected.add(faulty["scenario"].iloc[0])
        else:
            reward += config.REWARDS["false_alarm"]
    fault_scenarios = set(df.loc[df["fault"] == 1, "scenario"])
    reward += config.REWARDS["accepted"] * len(detected & fault_scenarios)
    reward += config.REWARDS["missed_fault"] * len(fault_scenarios - detected)
    n = max(len(fault_scenarios), 1)
    return reward / n


def tune_threshold(detectors, test_df, kind="autoencoder", percentiles=(95, 97, 98, 99, 99.5), seed=config.SEED + 7):
    tune_all = make_ahu_data(seed=seed)
    tune_df = tune_all[tune_all["scenario"].str.startswith("test")].reset_index(drop=True)
    det = detectors[kind]
    original = det.percentile
    table = []
    for p in percentiles:
        det.set_percentile(p)
        r_tune = _event_reward(tune_df, detect_anomalies(tune_df, detectors, kind)["findings"])
        table.append({"percentile": p, "tau": round(det.tau, 4), "mean_reward_tuning_set": round(r_tune, 3)})
    # highest reward wins; on a tie keep the value closest to the starting threshold
    # (only move tau when the reward really improves)
    best = max(table, key=lambda r: (r["mean_reward_tuning_set"], -abs(r["percentile"] - original)))
    det.set_percentile(original)
    r_before = _event_reward(test_df, detect_anomalies(test_df, detectors, kind)["findings"])
    det.set_percentile(best["percentile"])
    r_after = _event_reward(test_df, detect_anomalies(test_df, detectors, kind)["findings"])
    return {"table": table, "chosen_percentile": best["percentile"], "test_reward_before": round(r_before, 3),
            "test_reward_after": round(r_after, 3), "original_percentile": original}


def tune_threshold_lbnl(detectors, df, kind="autoencoder", percentiles=(95, 97, 98, 99, 99.5)):
    """Reward-based threshold tuning on the LBNL dataset, without touching the test part:
    tuning set = days 1-10 of every fault file + the fault-free validation days;
    test set   = days 11-31 of every fault file + the fault-free test days.
    Reward per fault file: +1 if its fault type is among the findings, -1 if missed;
    -0.5 for every finding on fault-free days (false alarm)."""
    day = pd.to_datetime(df["timestamp"]).dt.day
    tune = df[((df["fault"] == 1) & (day <= 10)) | (df["scenario"] == "val")]
    test = df[((df["fault"] == 1) & (day > 10)) | (df["scenario"] == "test_normal")]
    det = detectors[kind]
    original = det.percentile

    def reward(part):
        total, n = 0.0, 0
        for sc, g in part.groupby("scenario"):
            found = {f["fault_type"] for f in detect_anomalies(g, detectors, kind)["findings"]}
            if g["fault"].iloc[0] == 1:
                total += config.REWARDS["accepted"] if g["fault_type"].iloc[0] in found else config.REWARDS["missed_fault"]
                n += 1
            else:
                total += config.REWARDS["false_alarm"] * len(found)
        return total / max(n, 1)

    table = []
    for p in percentiles:
        det.set_percentile(p)
        table.append({"percentile": p, "tau": round(det.tau, 4), "mean_reward_tuning_set": round(reward(tune), 3)})
    best = max(table, key=lambda r: (r["mean_reward_tuning_set"], -abs(r["percentile"] - original)))
    det.set_percentile(original)
    before = reward(test)
    det.set_percentile(best["percentile"])
    after = reward(test)
    det.set_percentile(original)
    return {"table": table, "chosen_percentile": best["percentile"], "test_reward_before": round(before, 3),
            "test_reward_after": round(after, 3), "original_percentile": original}
