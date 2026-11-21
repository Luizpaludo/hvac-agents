# AI-Powered HVAC Operations Assistant for Cruise Ships: Multi-Agent System

**ITAI 2376 Deep Learning | Houston City College | Group 2 (14562)**

**Group members:** Mary Ann Mastri, Luiz Paludo, Cesar Antonio Noriega Martinez

| Member | Design contribution | Coding | Reflection |
|---|---|---|---|
| Mary Ann Mastri | Data, models, evaluation, reward design | LBNL data pipeline, Isolation Forest, autoencoder, energy baseline, metrics, code review of Area 6 | Data limits and evaluation |
| Luiz Paludo | Agent architecture, message protocol, database, safety rules | LangGraph agents, Input Guard, Safety Reviewer, ranker | Orchestration and safety |
| Cesar Antonio Noriega Martinez | RAG, hosting and cost plan, test plan | Ingestion and retrieval, Colab hosting, tests | Deployment and cost |

> **Advisory only:** the system never controls equipment. A human approves every recommendation.

> Code was written with AI assistance (Claude) and reviewed and tested by the group.

## What it does
Four cooperating agents help HVAC engineers (1) spot abnormal air-handler behavior to prioritize maintenance and
(2) find unusual energy use. Every recommendation is backed by evidence from technical manuals and checked by a Safety Reviewer.

- **Coordinator** (plan-then-execute): plans, sends tasks, ranks recommendations, assembles the report
- **Analyst** (ReAct): `validate_data`, `detect_anomalies` (autoencoder + Isolation Forest), `energy_baseline`
- **Knowledge** (ReAct + RAG): `search_manuals` over the NIST, Trane and ASHRAE manuals
- **Safety Reviewer** (reflection): checks numbers, comfort/safety limits and evidence; APPROVE / REVISE / ESCALATE
- **Learning:** human feedback -> reward -> Q-value update of the recommendation ranking; reward-based threshold tuning

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the flow chart, the step-by-step method and the tool table.

## Chat interface (like a normal app)
**Windows:** double-click `Start HVAC Agent.bat`. It installs the basic libraries and opens the chat page at http://localhost:8000.
**Any computer:** `python server.py`. **Colab:** notebook Part 6b opens the same page in a new tab.

The page has quick buttons for the demo situations, a box to type requests, CSV upload, Accept / Modify / Reject buttons on every
recommendation (this is how the agent learns), a live view of the 4 agents working, the message log and the learned Q-values.

**Better answers with Claude Haiku 5.5:** create a file named `.env` in this folder with one line,
`ANTHROPIC_API_KEY=your-key`, and restart. Without a key the system runs in template mode (same safety checks, no cost).
Never share the key or put it in the code; `.env` is excluded from Git.

## ONLINE mode (group members use your running agent through a link)
1. Double-click `ONLINE.bat` (first time it downloads `cloudflared.exe`, Cloudflare's official tunnel program).
2. A second window shows the **ACCESS CODE** (6 digits, new every start); the first window shows a link like
   `https://something.trycloudflare.com`.
3. Send the link and the code to the group. They open the link in any browser and type the code.
- Your computer must stay on with both windows open; close them to go offline. The link changes every time.
- Visitors use your API key, so there are limits: 15 requests per 10 minutes per person, 300 per day in total;
  8 wrong codes block that visitor for 10 minutes. You (on this computer) never need the code.
- The server only listens on this computer (127.0.0.1); the outside world reaches it only through the tunnel.

## Quick start (Google Colab)
Open `HVAC_Agents_Colab.ipynb` in Colab and run the parts in order. It installs the libraries, loads the manual library
(`chunks.json`, `embeddings.npy` from the RAG notebook), optionally starts PostgreSQL + pgvector, and runs the demo,
the 20 test cases and the evaluation.

## Quick start (your own computer)
```bash
pip install -r requirements.txt
cp .env.example .env          # optional: add ANTHROPIC_API_KEY (without it: template mode, no LLM cost)
# optional, better manual search: python -m pip install sentence-transformers
python demo.py                # 6 demo situations + learning from feedback
python demo.py --chat         # type your own requests
python tests/run_tests.py     # 20-case test matrix -> test_results.md
python evaluate.py            # metrics from the proposal -> evaluation_results.md
```

## Project structure
```
hvac_agents/
  config.py            settings, limits, rewards (no secrets)
  memory.py            database memory: runs, agent_messages, tool_calls, findings, recommendations, feedback, q_values
  protocol.py          TASK / ACK / RESULT / NACK / ERROR / ESCALATION with 60 s timeout and 2 retries
  llm.py               Claude Haiku 5.5 client: cache, token cap, JSON repair retry, template fallback
  system.py            Coordinator + LangGraph graph (built-in loop if LangGraph is missing) + report
  rl.py                replay experiment and reward-based threshold tuning
  agents/guard.py      Input Guard (rules G-01..G-07)
  agents/workers.py    Analyst, Knowledge and Safety Reviewer agents
  agents/ranker.py     recommendation library + epsilon-greedy Q-value ranker
  tools/data_tools.py  validate_data, data loading, synthetic demo data   [Area 6]
  tools/lbnl.py        LBNL FDD dataset loader (units, duplicates, 15-min) [Area 6]
  tools/anomaly.py     autoencoder + Isolation Forest, diagnosis rules     [Area 6]
  tools/energy.py      weather-aware energy baseline, CV(RMSE), NMBE       [Area 6]
  tools/knowledge.py   search_manuals (embeddings / pgvector / TF-IDF fallback)
data/                  chunks.json + embeddings.npy (manual library, see scripts/build_manual_library.py)
scripts/build_manual_library.py  rebuilds chunks.json from the 3 PDF manuals (Cesar's method)
tests/run_tests.py     the 20-case test matrix
evaluate.py            evaluation strategy (section 7 of the proposal)
demo.py                demonstration script for the video
server.py              web server for the chat interface (standard library only)
web/index.html         the chat page
Start HVAC Agent.bat   double-click launcher (Windows)
ONLINE.bat             ONLINE mode: agent + Cloudflare link + access code
HVAC_Agents_Colab.ipynb
```

## Data
**Real fault data: LBNL FDD dataset, single-duct AHU** (Lawrence Berkeley National Laboratory, DOI 10.25984/1881324, CC-BY).
Simulated by LBNL with EnergyPlus-Modelica for a large office in Chicago: 1 fault-free year and 20 faulty years
(supply-air sensor bias, cooling-valve leak, cooling-valve stuck, outdoor-air damper stuck, outdoor-air sensor bias).
- Download `LBNL_FDD_Data_Sets_SDAHU.zip` from https://faultdetection.lbl.gov/dataset/simulated-sd-ahu (form required)
  and extract it into `data/` (folder `data/LBNL_FDD_Dataset_SDAHU`). The first start converts it to 15-minute data
  in `data/lbnl_processed/` (about 31 MB, about 1 minute). Or copy only `data/lbnl_processed/` from a teammate.
- The big files are NOT in Git (`.gitignore`).
- Known problems in the published dataset (duplicate files, mixed units, shifted calendar) are handled in
  `hvac_agents/tools/lbnl.py` and listed in `docs/CHALLENGES_LOG.md`, section D.
- Results are validation on simulated building data, not on cruise ship data.

**Demo data (synthetic)** is still available (choose "Demo data" in the interface, or `HVAC_DATASET=demo`).
**Energy:** the energy branch still uses synthetic data until the BDG2 meter data is loaded.

## Configuration and secrets
Settings are in `hvac_agents/config.py`. Secrets are read only from environment variables, a gitignored `.env` file
or Colab Secrets. Nothing secret is stored in the repository (test TC-20 scans for keys).

## Cost
Claude Haiku 5.5 at $0.10 / $0.50 per million input/output tokens: about $0.003 per run (cost plan in the RAG notebook, Part 13).
Hard cap $15 with alerts at $5, $10 and $15. Template mode costs $0.
