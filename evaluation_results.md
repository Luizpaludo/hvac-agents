# Evaluation results

Date: 2026-11-20 | Dataset: **lbnl** | LBNL FDD dataset (simulated by LBNL, EnergyPlus-Modelica, Chicago weather; building AHU, not cruise ship)

Notes on the LBNL data: one year per file, 15-minute occupied data. The fault is active the whole year, but some faults have no visible effect in some seasons, so row-level recall is a lower bound. Duplicate files loaded once: 6 (4 leakage files identical, 4 OA-bias files identical).

## Maintenance: anomaly detection (fault files never used for training)

| Model | Precision | Recall | F1 | False-alarm rate (fault-free) |
|---|---|---|---|---|
| Autoencoder (sklearn-mlp) | 1.0 | 0.75 | 0.857 | 0.01 |
| Isolation Forest (baseline) | 0.999 | 0.168 | 0.287 | 0.008 |

Per file (autoencoder vs Isolation Forest):

| File | True fault | AE flag rate | AE diagnosis accuracy | AE fault types found | IF flag rate |
|---|---|---|---|---|---|
| normal | none | 0.01 | None | - | 0.008 |
| coi_bias_-4 | sat_sensor_bias | 0.995 | 0.695 | sat_sensor_bias, unknown | 0.07 |
| coi_bias_-2 | sat_sensor_bias | 0.525 | 1.0 | sat_sensor_bias | 0.012 |
| coi_bias_2 | sat_sensor_bias | 0.614 | 0.996 | sat_sensor_bias, unknown | 0.013 |
| coi_bias_4 | sat_sensor_bias | 0.702 | 1.0 | sat_sensor_bias | 0.022 |
| coi_leakage_010 | ccv_leakage | 0.323 | 0.943 | ccv_leakage, ccv_stuck | 0.011 |
| coi_stuck_010 | ccv_stuck | 0.999 | 0.88 | ccv_leakage, ccv_stuck | 0.648 |
| coi_stuck_025 | ccv_stuck | 0.999 | 0.997 | ccv_stuck, unknown | 0.633 |
| coi_stuck_050 | ccv_stuck | 1.0 | 1.0 | ccv_stuck | 0.252 |
| coi_stuck_075 | ccv_stuck | 1.0 | 1.0 | ccv_stuck | 0.091 |
| damper_stuck_010 | stuck_oa_damper | 0.39 | 1.0 | stuck_oa_damper | 0.008 |
| damper_stuck_025 | stuck_oa_damper | 1.0 | 1.0 | stuck_oa_damper | 0.012 |
| damper_stuck_075 | stuck_oa_damper | 1.0 | 1.0 | stuck_oa_damper | 0.3 |
| damper_stuck_100_short | stuck_oa_damper | 0.963 | 1.0 | stuck_oa_damper | 0.328 |
| oa_bias_-4 | oat_sensor_bias | 0.081 | 0.0 | sat_sensor_bias, unknown | 0.011 |

## Energy baseline (SYNTHETIC demo energy data (BDG2 not loaded yet))

- Hourly CV(RMSE): **8.16%** (target <= 30%); NMBE: **0.73%** (target |NMBE| <= 10%)
- Labeled anomaly days flagged: 14 of 14

## Reinforcement learning

### Replay experiment (SIMULATED reviewer feedback, average of 10 random seeds)

| Round | Learned ranker mean reward | Random ranking mean reward |
|---|---|---|
| 1 | 0.516 | 0.226 |
| 2 | 0.549 | 0.235 |
| 3 | 0.589 | 0.306 |
| 4 | 0.609 | 0.269 |
| 5 | 0.665 | 0.235 |

Gain round 5 vs round 1 (average of 10 seeds): **0.149** (target >= 0.15: missed); per seed: [0.338, -0.101, 0.263, 0.175, -0.063, 0.2, 0.45, 0.162, -0.075, 0.137]; single run with seed 42: -0.038. Learned ranking beats random ranking in every round.

### Reward-based threshold tuning

- autoencoder: chosen percentile 99 (start 99.0); test reward 0.857 -> 0.857
- isolation_forest: chosen percentile 98 (start 99.0); test reward 0.286 -> 0.821

## Agents and safety

- Harmful prompts blocked: **10 / 10** (target 10 of 10), rules ['G-04', 'G-02', 'G-04', 'G-04', 'G-04', 'G-02', 'G-03', 'G-05', 'G-04', 'G-03']
- ACK rate: **1.0** (target 100%); p95 latency 0.16 s (target < 90 s)
- LLM mode during evaluation: template | graph engine: built-in loop

## RAG

- Full Hit@4 evaluation is in Cesar's RAG notebook (Part 10): 85% Hit@4, 79/79 valid citations, 10/10 NO_EVIDENCE.
