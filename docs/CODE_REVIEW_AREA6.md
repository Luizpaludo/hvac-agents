# Code review: Area 6 (data, models and evaluation)

**ITAI 2376 | Group 2 (14562)**
Area owner: Mary Ann Mastri · Date: 11/20/2026 · Review carried out with AI assistance (Claude)

Files reviewed: `hvac_agents/tools/lbnl.py`, `hvac_agents/tools/data_tools.py`, `hvac_agents/tools/anomaly.py`,
`hvac_agents/tools/energy.py`, `hvac_agents/rl.py` (threshold tuning), `evaluate.py`.

## How the review was done
1. Read every function against the proposal (sections 3 and 7) and the LBNL dataset documentation.
2. Ran the code with unusual inputs (edge cases), listed below.
3. Checked that no rule threshold or model is fitted on the fault (test) files.
4. Added an automatic test for every problem found, so it cannot come back.

## Edge cases tested
| # | Input | Expected | Result before review | After |
|---|---|---|---|---|
| E1 | Demo-format CSV (no valve/damper command columns) sent to the LBNL-trained model | Clear refusal with the reason | **Crash**: "Analyst failed after 2 retries" | NACK: "missing required columns ccv_cmd, damper_cmd; the model was trained on LBNL data" (test TC-L5) |
| E2 | One week of data with 5 % missing values | Missing rows skipped, never filled; fault still found | **Wrong**: scores distorted, stuck valve downgraded from high to low | Rows with missing values get no score and are reported ("18 rows skipped, not filled"); fault found as high (test TC-L6) |
| E3 | Empty week (date outside the data) | NACK "dataset is empty" | Correct | Correct |
| E4 | Only 3 rows | NACK "need at least 20 rows" | Correct | Correct |
| E5 | Energy data with only 44 days of history | "No baseline" (needs 90 days) | Correct | Correct |
| E6 | Energy data with 2 % missing readings | Baseline still valid | Correct (CV(RMSE) 8.12 %) | Correct |
| E7 | 14 LBNL files combined in one table (same timestamps) | Timestamps checked per file | Rejected (duplicated timestamps) | Fixed earlier (test TC-L4) |
| E8 | Stuck valve at a small opening vs. leaking valve | Different diagnoses | Mixed up | Fixed earlier (log D11) |

## Checks that passed
- **No test leakage:** autoencoder, Isolation Forest, the residual models (`ccv_resid`, `mat_resid`) and all rule thresholds
  are fitted only on fault-free training/validation days. Threshold tuning uses days 1-10 of the fault files and is
  evaluated on days 11-31.
- **Units:** degF converted to degC; 0-1 signals to %; static pressure Pa or inWC detected per file; `validate_data`
  rejects Fahrenheit and out-of-range percentages.
- **Duplicates:** files compared by MD5 hash; each unique file loaded once.
- **Reproducibility:** fixed random seed (42); repeat run gives identical findings (test TC-19).
- **Honest labels:** every result carries its data source (LBNL simulation or synthetic demo data).

## Changes made in the review
1. `data_tools.validate_data`: new `extra_columns` argument; the Analyst asks for the command columns when the model was trained on LBNL data.
2. `anomaly.AnomalyDetector.score`: rows with missing values get NaN instead of a score and are never flagged; training and validation drop incomplete rows.
3. `anomaly.detect_anomalies`: returns `skipped_rows_missing_values`; the Analyst shows it in its observation.
4. `data_tools`: removed an old, wrong guess of the LBNL column names (the real mapping is in `lbnl.py`).

## Remaining notes (not bugs)
- The PyTorch autoencoder path runs only where PyTorch is installed (Colab). Elsewhere the same network is trained with scikit-learn. Both use a 3-unit bottleneck; results may differ slightly.
- The energy baseline is still tested on synthetic data until the BDG2 meter data is loaded.
- Diagnosis rules are hints; several causes can produce the same pattern (NIST IR 7365).

Tests after the review: **26 / 26 pass** (`python tests/run_tests.py`).
