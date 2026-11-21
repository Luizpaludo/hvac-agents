# Challenges and Solutions Log

**ITAI 2376 | Group 2 (14562) | HVAC Multi-Agent System**
Group members: Mary Ann Mastri, Luiz Paludo, Cesar Antonio Noriega Martinez
Log kept by: Luiz Paludo (agent architecture, protocol, database, safety rules, interface)

Every problem found while building and testing the system, what caused it, how it was solved, and its status.
Use it for the "Challenges and Solutions" and "Lessons Learned" sections of the final report.

Status: **Fixed** = solved and tested · **Known limit** = documented, not fully solved · **Open** = still to do

---

## A. Agent system and code (built 11/19/2026)

| # | Problem | Cause | Solution | Status |
|---|---|---|---|---|
| A1 | The demo data failed its own validation: "column ccv outside 0-100 %". | Random noise was added to the cooling valve signal **after** limiting it to 0-100 %, so some values went above 100 %. | Add the noise first, then limit to 0-100 %. Shows that `validate_data` really catches bad units/ranges. | Fixed |
| A2 | Isolation Forest (baseline model) found almost no faults: recall 0.01, precision 0.13. | Its threshold (99th percentile of fault-free scores) was too strict, and its scores change little between normal and faulty data. | Kept as the baseline for comparison. Reward-based threshold tuning moved it to the 95th percentile, and the test reward improved from -1.125 to -0.5. The autoencoder (recall 0.81, precision 0.975) is the main model. | Known limit (reported as a result) |
| A3 | The mixed-air sensor fault was not detected and not diagnosed. | A sensor reading 4 °C **too high** stayed between outdoor and return air on hot days, so it looked normal. | Changed the simulated fault to a sensor reading 4 °C **too low**, which is physically detectable. Recall for this fault is still low (0.25 to 0.5). | Known limit |
| A4 | The labeled energy anomaly (2 weeks) was not flagged (0 of 14 days). | The simulated night-time increase (+35 %) only raised **daily** energy about 12 %, below the 15 % threshold. | Made the simulated fault stronger (+60 % at night, equipment left running). The threshold was not changed. Result: 14 of 14 days flagged. | Fixed (test design) |
| A5 | Threshold tuning chose a threshold that made the test result worse (reward 1.0 → 0.5). | Several thresholds tied on the tuning set, and the tie-break picked the highest one. | On a tie, keep the value closest to the starting threshold: only move it when the reward really improves. | Fixed |
| A6 | A second run in the same session crashed: "UNIQUE constraint failed: findings.finding_id". | Finding IDs (F1, F2…) restart in every run, but the database used the ID alone as the key. | The key is now (run_id, finding_id). Test TC-19 now runs twice on the same database to catch this. | Fixed |
| A7 | Runs with no findings showed the status "analysed" instead of a final status. | The report step did not set a status when there was no safety review. | Status is now "no_findings" (or "data_rejected" / "analyst_failed"). | Fixed |
| A8 | The demo's "learning from feedback" step re-ran the wrong scenario (energy instead of fan belt). | The demo kept the **last** run with recommendations instead of the first one. | The demo now keeps the first run and its data window. | Fixed |
| A9 | LangGraph, PyTorch, sentence-transformers and the Anthropic library could not be installed in the development environment. | The build environment had no internet access. | Every part has a fallback: built-in graph loop with the same nodes, scikit-learn autoencoder, TF-IDF search, template mode. The LangGraph wiring was checked with a small stand-in. The real libraries must be tested in Colab. | Known limit (test in Colab) |
| A10 | The Knowledge agent always answered NO_EVIDENCE. | `chunks.json` (from the RAG notebook) was not in the `data` folder. Only `embeddings.npy` was shared. | The agent says why ("chunks.json not found") instead of inventing sources. Tests use a small labeled test text. Copy `chunks.json` into `data/`. | Open |
| A11 | The agent had no screen: only notebook output and a terminal. | The first version was built for Colab only. | Built a chat web interface (server.py + web/index.html) with quick buttons, CSV upload, Accept / Modify / Reject, a live agent-team view, the message log and Q-values. | Fixed |
| A12 | During testing a recommendation said to raise duct static pressure to 3.0 inWC (limit 2.5). | Some library actions intentionally touch setpoints, to test the Safety Reviewer. | The Safety Reviewer removed it (REVISE) and gave reward -1, so the ranker learns to rank it lower. This is a planned safety demonstration. | Working as designed |
| A13 | Group members needed to test the agent without installing anything, but the agent uses Luiz's API key (cost and leak risk). The server also listened on all network addresses (0.0.0.0), so anyone on the same Wi-Fi could use it. | First version was made for one user on one computer. | ONLINE mode (`ONLINE.bat`): Cloudflare tunnel link + 6-digit access code (new every start) + limits (15 requests / 10 min per person, 300 / day) + 10-minute block after 8 wrong codes. The server now listens only on 127.0.0.1. | Fixed |
| A14 | While testing ONLINE mode, the page kept asking for the access code even after the right code was typed. | The page re-sent the request with the old (empty) code saved from the first try. | Every request now uses the latest code. Tested in a browser: wrong code refused, right code works, limits return "please wait". | Fixed |

## B. API key and LLM connection (11/19/2026, evening)

| # | Problem | Cause | Solution | Status |
|---|---|---|---|---|
| B1 | The page stayed in "template mode" after the key was added. | The key file was saved as `.env.txt`: Windows Notepad adds ".txt" and Windows hides file extensions. | Added a startup check that names the problem ("found .env.txt: rename it"). The program now also reads `.env.txt`, and the reader ignores the hidden BOM character some editors add. | Fixed |
| B2 | Risk of uploading the API key to GitHub. | `.gitignore` only blocked `.env`, not `.env.txt`. | `.gitignore` now blocks every `.env*` file except `.env.example`. The secret-scan test skips key files instead of reading them. | Fixed |
| B3 | The key was typed in the **File name** box of the Save dialog, and part of it was visible in screenshots. | Confusion between the file name (`.env`) and the file content (the key line). | Key goes inside the file; file name is `.env`. Security rule: never show a key in screenshots or chats; replace (rotate) a key that was exposed. | Fixed (key rotation recommended) |
| B4 | The key was found, but the LLM was never used: 7 calls, 0 tokens. The summary said "written by template". | Every API call failed, and the code hid the error and used the template fallback. | The error is now shown in the console, the Welcome message and the report ("LLM problem: …"). A test call is made at startup ("LLM test call: OK / FAILED"). The key is never printed. | Fixed |
| B5 | "TypeError: Messages.create() got an unexpected keyword argument 'temperature'". | The installed version of the `anthropic` library does not accept the `temperature` setting. | The client tries with temperature and, if it is refused, repeats the call without it and remembers that. Confirmed on Luiz's notebook: "LLM test call: OK, claude-haiku-5-5 answered". | Fixed |
| B6 | Startup check warned "the key does not look like an API key (API keys start with sk-ant-api)", but the key worked. | The check guessed the key type from its first letters; that guess was too strict. | Removed the guess. The startup **test call** is the real check (it proved the key works). A wrong key still shows a clear 401 message. | Fixed |
| B7 | "BadRequestError 400: user messages must have non-empty content". The summary in the report was empty. | Haiku returned an **empty** summary (likely its 250-token limit was too small); the Safety Reviewer then sent that empty text back to the API, which refuses empty messages. | Empty answers now count as a failure (template text is used and the reason is shown); empty prompts are never sent; the summary limit was raised to 500 tokens. | Fixed |
| B8 | ReAct step 3 showed "(LLM choice corrected by rule)" even though Haiku correctly chose "finish". | The rule check also compared the branch name, which does not matter for "finish". | For "finish" only the action is compared. | Fixed |

## C. Earlier group work found during review (RAG notebook and proposal)

| # | Problem | Cause | Solution | Status |
|---|---|---|---|---|
| C1 | Running the RAG notebook top to bottom gives 8/10 NO_EVIDENCE instead of 10/10. | `MIN_SCORE = 0.42` sits in a cell **after** the evaluation, while Parts 5 and 9 still set 0.30. | Set 0.42 inside Parts 5 and 9. The agent system already uses 0.42. | Open (Cesar's notebook) |
| C2 | Part 11a of the RAG notebook froze (KeyboardInterrupt). | `psql` opened a pager and waited for a key press. | Run psql with `-P pager=off` (used in the agent notebook). | Fixed in agent notebook |
| C3 | TC-14 failed: HVAC questions the manuals do not cover still returned passages. | Similarity score alone cannot tell "covered" from "related but not covered". | Second guard: the Safety Reviewer checks that each passage mentions the fault; if not, a high-severity finding is ESCALATED. New TC-14 passes. | Fixed (by design) |
| C4 | A member's name is misspelled in the proposal ("Luiz Pulido" instead of Luiz Paludo). | Typo. | Use the correct name in the report, README, video and repo (grading rule: missing names get 0). | Open |
| C5 | The final due date appears as Dec 6 on Canvas and Dec 7 in the instructions. | Inconsistent course information. | Plan for Dec 6. | Open |
| C6 | `chunks.json` (the manual text for the Knowledge agent) was missing, so every report said NO_EVIDENCE and serious findings were escalated. Only `embeddings.npy` had been shared. | The file stayed in Cesar's Colab session. | Found the exact public PDFs Cesar used (IDs 7365, aeb01d19, 288388245 in his notebook) and re-ran his exact method (`scripts/build_manual_library.py`). Checks: same page counts, 393 chunks (175 NIST, 156 ASHRAE, 62 Trane); 17 of 20 passages printed in his notebook identical, 3 shifted by 2-7 characters (different pypdf version); alignment with his embeddings confirmed (correlation 0.575 vs 0.00 when shuffled). Keyword search without the embedding model: 17/20 HVAC questions, 10/10 off-topic NO_EVIDENCE (same as his 85 %). | Fixed |


## D. Real dataset: LBNL FDD single-duct AHU (11/19/2026, night)

| # | Problem | Cause | Solution | Status |
|---|---|---|---|---|
| D1 | The synthetic practice data was too small, too easy and "graded itself" (the same person wrote the data and the diagnosis rules). | The first version needed data quickly. | Switched to the public LBNL FDD dataset (DOI 10.25984/1881324), as promised in the proposal. Synthetic data kept only as "Demo data". | Fixed |
| D2 | The download stopped at a form asking for name, e-mail and organization. | LBNL asks users to register before downloading. | Luiz filled in the form himself (personal data is not typed by the agent). | Fixed |
| D3 | Windows said "Access to the compressed (zipped) folder is denied". | The download (593 MB) was not finished yet. | Waited for the full download, then extracted it. | Fixed |
| D4 | **The 4 cooling-valve leak files (10, 25, 40, 50 %) are byte-identical**, and so are **the 4 outdoor-air sensor bias files (-4, -2, +2, +4 C)**. | Packaging error in the published dataset (checked with MD5 hashes). | The loader detects duplicates by hash and loads each only once: **15 different files (1 fault-free + 14 faulty), not 21**. The leak files all contain a 10 % leak. | Fixed (reported) |
| D5 | The "outdoor-air sensor bias" file shows **no bias** in the outdoor temperature column (same values as fault-free). | Probably the same packaging problem, or the bias is applied only inside the controller. | Reported honestly: this fault is detected in only 8 % of rows and never diagnosed correctly. It is the only file the agent misses. | Known limit |
| D6 | Some column headers do not match the documentation: supply-air bias files are named `coi_bias` (doc: `sa_bias`); static pressure `SA_SP` is in **Pa** in the fault-free file but in **inWC** in fault files; the setpoint column is **-400** in fault files. | Inconsistent dataset export. | SA_SP converted to inWC automatically (values > 50 are Pa); documented setpoint 1.607 inWC used. | Fixed |
| D7 | `SF_SPD` (fan speed) and `OA_CFM` are constant all year; airflow values are around 1 million CFM. | Unusable columns in the export. | Fan speed taken from the fan control signal `SF_CS`; airflow columns not used. | Fixed |
| D8 | The building looked "closed" on Mondays and open on Sundays. | The simulation calendar is shifted by one day compared with the 2018 date labels. | Occupied periods taken from the occupancy signal `SYS_CTL`, not from the calendar. | Fixed |
| D9 | Temperatures are in degF and signals in 0-1; the agent expects degC and %. | Different units. | Converted in the loader; the unit check of `validate_data` confirms. | Fixed |
| D10 | Labels: the fault is "active all year", but some faults have no effect in some seasons (e.g. damper stuck at 10 % when the minimum position is 10 %). | Dataset design (one file = one fault for a full year). | Report per-file flag rate and diagnosis accuracy; row-level recall is explained as a lower bound. | Known limit |
| D11 | Stuck valve at a small opening was diagnosed as a leak. | When the controller asks for 0 %, a valve stuck at 10 % looks exactly like a 10 % leak. | New rule: "stuck" if the position stays still while the command moves; "leak" only when the valve fails to close. **Honest note: this rule was changed after looking at test results**, so the diagnosis accuracy is a little optimistic. | Fixed |
| D12 | The "Whole test year" run was rejected: "233,179 duplicated timestamps". | 14 files cover the same year, so timestamps repeat when they are combined. | `validate_data` now checks timestamps inside each file; events never join across files or nights. | Fixed |
| D13 | Loading 21 CSV files (2.8 GB) is slow, and copying them failed with "wall-clock timeout" when several were sent at once. | Large files. | Files copied one at a time; processed once into small 15-minute CSV files (`data/lbnl_processed`, 31 MB, portable between pandas versions). Big files excluded from Git. | Fixed |
| D14 | The learning replay test missed its target after the new fault types were added (round 5 vs round 1: -0.04). | With 10 fault types and 40 trials per round, each type gets only ~4 trials: very noisy. | Repeated with 10 random seeds: average gain 0.149 (target 0.15, borderline); learned ranking beats random in every round (about 0.63 vs 0.26). **Changed after seeing the miss**; both numbers are reported. | Known limit |
| D15 | Code review: a demo-format CSV sent to the LBNL-trained model crashed the Analyst ("failed after 2 retries"). | The LBNL model needs the valve/damper command columns; the data check did not ask for them. | `validate_data` now requires those columns when the model was trained on LBNL data and answers with a clear NACK. Test TC-L5. | Fixed |
| D16 | Code review: 5 % missing values silently distorted the anomaly scores (a stuck valve dropped from high to low severity). | Missing values went into the model. | Rows with missing values are skipped (never filled in) and reported in the Analyst's observation. Test TC-L6. Full review: `docs/CODE_REVIEW_AREA6.md`. | Fixed |
| D17 | On Luiz's PC (11/20/2026) the Knowledge agent showed "NO_EVIDENCE (best score 0.103 < 0.12)" even though `chunks.json` and `embeddings.npy` were in `data/`. | `sentence-transformers` was not installed, so the library silently used the TF-IDF keyword fallback; Haiku's free-text query shared too few exact words with the manuals. | Installed `sentence-transformers` (6.3.0, with torch 2.15); check printed `embeddings OK`. Same lesson as B4: a fallback must be visible. | Fixed |
| D18 | While recording the demo video (11/20/2026) the manual-question answer ("How do I adjust fan belt tension?") showed "LLM problem: the model returned an empty answer (template text used instead)". | Haiku returned an empty text for the explain step. | The fallback worked as designed: real manual passages with source, page and score were still shown, and the reason was printed. Not changed before submission. | Open |

**Main results on LBNL (autoencoder):** precision 1.00, recall 0.75, false-alarm rate 1 % on fault-free days; the right
fault type is found in 13 of 14 fault files. Isolation Forest baseline: recall 0.17. Reward-based threshold tuning
raised the Isolation Forest test reward from 0.29 to 0.82.

---

## Lessons learned (draft)
1. **Check by testing, not by guessing.** The key-format warning (B6) was wrong; one real test call gave the true answer.
2. **Fallbacks must report their reason.** A silent fallback made the system look healthy while the LLM never worked (B4). Every fallback now says why it was used.
3. **Test with the real environment early.** Library-version problems (B5) only appear on the real machine.
4. **Small Windows details matter.** Hidden file extensions (B1) can block a whole feature.
5. **Secrets need several layers of protection:** .env files, `.gitignore`, a secret-scan test, and never showing keys in screenshots.
6. **Test the reviewer, not only the model.** Safety checks (A12, C3) catch problems that a good model score does not show.
7. **Be honest about data.** Every result says which data it comes from (LBNL simulation or synthetic demo data); every change made after seeing results is written down here (A3, A4, D11, D14).
8. **Check the data before trusting it.** The official LBNL dataset had duplicate files, mixed units and a shifted calendar (D4-D8); a hash check and simple statistics found them in minutes.
