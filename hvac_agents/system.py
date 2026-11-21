"""
system.py - The Coordinator and the agent graph (Area 4, owner: Luiz Paludo).

Step-by-step flow (matches section 6 of the proposal):
  1 Input     -> run ID, request and dataset recorded in memory
  2 Screen    -> Input Guard (rules G-01..G-07); refusals are logged and explained
  3 Plan      -> Coordinator writes a JSON plan (plan-then-execute)
  4 Validate  -> Analyst ACKs and runs validate_data (a failure stops only that branch)
  5 Analyse   -> Analyst ReAct loop with detect_anomalies / energy_baseline
  6 Retrieve  -> Knowledge agent (linked to the Analyst's RESULT) searches the manuals
  7 Recommend -> up to 3 candidates per finding, ranked by learned Q-values
  8 Review    -> Safety Reviewer: APPROVE, REVISE (max 2 loops) or ESCALATE
  9 Deliver   -> report with findings, sources, versions, uncertainty and the advisory banner
 10 Learn     -> human feedback -> reward -> Q-value update (feedback() method)

The graph runs on LangGraph when it is installed (Colab: pip install langgraph).
Without LangGraph the same node functions run in a simple loop with the same
routing rules, so the behaviour is identical.
"""
import json
import re
import time

import pandas as pd
from pathlib import Path
from typing import Any, TypedDict

from . import config
from .agents import guard
from .agents.ranker import Ranker
from .agents.workers import AnalystAgent, KnowledgeAgent, SafetyReviewer, template_summary
from .llm import LLMClient, LLMUnavailable
from .memory import Memory, new_id
from .protocol import AgentNack, MessageBus, TaskFailed
from .tools.anomaly import train_detectors
from .tools.data_tools import load_dataset, make_energy_data
from .tools.knowledge import ManualLibrary

VERSION = "1.0.0"
REPORTS_DIR = config.ROOT / "reports"


class RunState(TypedDict, total=False):
    request: str
    intent: str
    status: str
    refusal: Any
    plan: Any
    analyst: Any
    analyst_msg: str
    findings: Any
    summary: str
    summary_source: str
    evidence: Any
    knowledge_msg: str
    knowledge_trace: Any
    recommendations: Any
    review: Any
    loop: int
    revisions: Any
    errors: Any
    report: str
    answer: Any


class HVACSystem:
    def __init__(self, memory=None, llm=None, ahu_df=None, energy_df=None, library=None,
                 detectors=None, use_langgraph=True, timeout_s=None, dataset=None):
        self.memory = memory or Memory()
        self.llm = llm or LLMClient()
        if ahu_df is None:
            self.dataset, ahu_df = load_dataset(dataset)     # "lbnl" (real) or "demo" (synthetic)
        else:
            self.dataset = "lbnl" if "ccv_cmd" in ahu_df.columns else "demo"
        self.ahu_df = ahu_df
        self.energy_df = energy_df if energy_df is not None else make_energy_data()
        self.detectors = detectors if detectors is not None else train_detectors(self.ahu_df)
        self.library = library or ManualLibrary(memory=self.memory)
        self.ranker = Ranker(self.memory)
        self.timeout_s = timeout_s
        self.run_id = None
        self.graph = self._build_langgraph() if use_langgraph else None
        self.engine = "langgraph" if self.graph is not None else "built-in loop"

    # ================================================================ public API
    def run(self, request, ahu=None, energy=None, file_path=None, detector="autoencoder"):
        """Run the whole agent team for one request. Returns the final state (with 'report')."""
        t0 = time.time()
        self._datasets = {"maintenance": ahu if ahu is not None else self.default_maintenance_window(),
                          "energy": energy if energy is not None else self.energy_df}
        self._detector, self._file = detector, file_path
        try:
            self.memory.ping()
            self.run_id = self.memory.start_run(request, file_path or "demo dataset", self.llm.mode)
        except Exception as exc:                                    # TC-10: database down
            return self._db_failure(request, exc)
        self.bus = MessageBus(self.memory, self.run_id, timeout_s=self.timeout_s)
        self.bus.register("analyst", AnalystAgent(self.memory, self.llm, self.detectors, lambda: self.run_id))
        self.bus.register("knowledge", KnowledgeAgent(self.memory, self.llm, self.library, lambda: self.run_id))
        self.bus.register("reviewer", SafetyReviewer(self.memory, self.llm, lambda: self.run_id))
        calls0, tin0, tout0 = self.llm.calls, self.llm.tokens_in, self.llm.tokens_out
        state = {"request": request, "loop": 0, "errors": [], "revisions": []}
        try:
            if self.graph is not None:
                try:
                    state = self.graph.invoke(state)
                except Exception as exc:
                    if not type(exc).__module__.startswith("langgraph"):
                        raise
                    # a LangGraph version problem must not stop the demo: same nodes, built-in loop
                    self.engine = "built-in loop (LangGraph error)"
                    state = self._run_loop({**state, "errors": [f"LangGraph: {exc}"]})
            else:
                state = self._run_loop(state)
        except Exception as exc:                                    # e.g. database lost mid-run
            state = {**state, "status": "error", "errors": state.get("errors", []) + [f"{type(exc).__name__}: {exc}"]}
            state["report"] = self._error_report(state)
        latency = round(time.time() - t0, 2)
        state["metrics"] = {"latency_s": latency, "llm_calls": self.llm.calls - calls0,
                            "tokens_in": self.llm.tokens_in - tin0, "tokens_out": self.llm.tokens_out - tout0}
        try:
            state["metrics"]["ack_rate"] = round(self.memory.ack_rate(self.run_id), 3)
            self.memory.finish_run(self.run_id, status=state.get("status", "done"), intent=state.get("intent"),
                                   latency_s=latency, llm_calls=state["metrics"]["llm_calls"],
                                   tokens_in=state["metrics"]["tokens_in"], tokens_out=state["metrics"]["tokens_out"],
                                   cost_usd=round(self.llm.cost_usd, 5))
        except Exception as exc:
            state["errors"].append(f"could not save run metrics: {exc}")
        state["report"] = state.get("report", "") + self._footer(state)
        state["run_id"] = self.run_id
        self._save_report_file(state["report"])
        return state

    def feedback(self, rec_id, outcome, simulated=False):
        """Step 10 Learn: human accepts / modifies / rejects a recommendation -> reward -> Q update."""
        reward = config.REWARDS[outcome]
        rows = self.memory.query(
            "SELECT r.run_id, r.candidate_id, f.fault_type FROM recommendations r "
            "JOIN findings f ON f.finding_id = r.finding_id AND f.run_id = r.run_id WHERE r.rec_id = ?", (rec_id,))
        if not rows:
            raise KeyError(f"unknown recommendation {rec_id}")
        r = rows[0]
        q_new = self.ranker.update(r["fault_type"], r["candidate_id"], reward)
        self.memory.save_feedback(r["run_id"], rec_id, outcome, reward, simulated)
        return {"rec_id": rec_id, "outcome": outcome, "reward": reward, "new_q": round(q_new, 3)}

    def default_maintenance_window(self):
        return self.window("all_tests")

    def window(self, scenario=None, start=None, days=None):
        """A slice of the dataset to analyse (keeps the data-source label)."""
        df = self.ahu_df
        if scenario in (None, "all_tests"):
            part = df[df["scenario"].str.startswith("test")]
        else:
            part = df[df["scenario"] == scenario]
        if start is not None:
            t = pd.to_datetime(part["timestamp"])
            t0 = pd.Timestamp(start)
            part = part[(t >= t0) & (t < t0 + pd.Timedelta(days=days or 7))]
        part = part.reset_index(drop=True)
        part.attrs.update(df.attrs)
        return part

    # ================================================================ nodes
    def node_guard(self, state):
        g = guard.screen(state["request"])
        if g["allowed"] and self._file:
            g = {**guard.check_file(self._file), "intent": g["intent"]}
        if not g["allowed"]:
            self.bus.log("input_guard", "user", "NACK", "screen", {"rule_id": g["rule_id"], "reason": g["reason"]})
            return {"status": "refused", "refusal": g, "intent": None}
        self.bus.log("input_guard", "coordinator", "RESULT", "screen", {"intent": g["intent"]})
        return {"intent": g["intent"], "status": "screened"}

    def node_plan(self, state):
        intent = state["intent"]
        expected = {"maintenance": ["maintenance"], "energy": ["energy"], "both": ["maintenance", "energy"],
                    "explain": ["explain"]}[intent]
        plan = {"branches": expected, "reason": f"template plan for intent '{intent}'", "source": "template"}
        if self.llm.available:
            try:
                d = self.llm.json("You are the Coordinator (Chief Engineer) of an HVAC agent team. Make a plan. "
                                  'Reply as JSON: {"branches": [..], "reason": "..."}. Allowed branches: '
                                  "maintenance, energy, explain.",
                                  json.dumps({"request": state["request"], "intent": intent}),
                                  required_keys=("branches", "reason"))
                if sorted(d["branches"]) == sorted(expected):     # plan must match the screened intent
                    plan = {"branches": expected, "reason": str(d["reason"])[:300], "source": "llm"}
            except LLMUnavailable:
                pass
        self.bus.log("coordinator", "coordinator", "RESULT", "plan", plan)
        return {"plan": plan, "status": "planned"}

    def node_analyze(self, state):
        payload = {"request": state["request"], "branches": state["plan"]["branches"],
                   "datasets": self._datasets, "detector": self._detector}
        try:
            res, msg = self.bus.request("coordinator", "analyst", "analyse", payload)
        except AgentNack as nack:
            return {"status": "data_rejected", "findings": [], "analyst": {"failed_branches": nack.reasons},
                    "summary": "The analysis stopped: " + "; ".join(nack.reasons)}
        except TaskFailed as exc:
            return {"status": "analyst_failed", "findings": [], "errors": state["errors"] + [str(exc)],
                    "summary": "The Analyst agent failed after 2 retries. A human must review the data."}
        summary, source = res["summary"], "template"
        if self.llm.available and res["findings"]:
            try:
                summary = self.llm.text("You write a 3-sentence summary for HVAC engineers. Use ONLY numbers that "
                                        "appear in the JSON. Say it is advisory and that scores are anomaly scores.",
                                        json.dumps(res["findings"], default=str)[:3000], max_tokens=500)
                source = "llm"
            except LLMUnavailable:
                pass
        for f in res["findings"]:
            self.memory.save_finding(self.run_id, f)
        return {"analyst": res, "analyst_msg": msg, "findings": res["findings"], "summary": summary,
                "summary_source": source, "status": "analysed"}

    def node_retrieve(self, state):
        if state["intent"] == "explain":
            payload, dep = {"question": state["request"]}, None
        else:
            payload, dep = {"findings": state["findings"]}, state.get("analyst_msg")
        try:
            res, msg = self.bus.request("coordinator", "knowledge", "retrieve", payload, depends_on=dep)
        except TaskFailed as exc:
            return {"evidence": {}, "knowledge_trace": [], "errors": state["errors"] + [str(exc)]}
        return {"evidence": res["evidence"], "knowledge_msg": msg, "knowledge_trace": res["trace"]}

    def node_recommend(self, state):
        recs = []
        for f in state["findings"]:
            for rank, c in enumerate(self.ranker.rank(f["fault_type"])[:3], start=1):
                recs.append({"rec_id": new_id("rec"), "finding_id": f["finding_id"], "candidate_id": c["id"],
                             "text": c["text"], "rank": rank, "q_value": c["q_value"], "explored": c.get("explored", False),
                             "parameter": c.get("parameter"), "value": c.get("value"), "status": "draft"})
        return {"recommendations": recs}

    def node_review(self, state):
        payload = {"summary": state["summary"], "findings": state["findings"],
                   "tool_outputs": state.get("analyst", {}).get("tool_outputs", {}),
                   "recommendations": [r for r in state["recommendations"] if r["status"] != "removed"],
                   "evidence": state.get("evidence", {}), "loop": state["loop"]}
        dep = state.get("knowledge_msg") or state.get("analyst_msg")
        try:
            res, _ = self.bus.request("coordinator", "reviewer", "review", payload, depends_on=dep)
        except TaskFailed as exc:
            res = {"decision": "ESCALATE", "issues": [], "escalations": [f"reviewer unavailable: {exc}"],
                   "rec_checks": {}, "note": "", "findings": state["findings"]}
        if res["decision"] == "ESCALATE":
            self.bus.escalate("reviewer", res["escalations"], depends_on=dep)
        return {"review": res, "findings": res["findings"]}

    def node_revise(self, state):
        """Coordinator fixes what the Reviewer found, then sends the draft back (REVISE loop)."""
        recs, notes = [dict(r) for r in state["recommendations"]], []
        summary = state["summary"]
        for issue in state["review"]["issues"]:
            if issue["type"] == "wrong_number":
                summary = template_summary(state["findings"], state.get("analyst", {}).get("failed_branches", {}))
                notes.append("summary rewritten from tool numbers (wrong number found)")
            elif issue["type"] == "safety_violation":
                for r in recs:
                    if r["rec_id"] == issue["rec_id"]:
                        r["status"] = "removed"
                        # the Reviewer's verdict is also a reward signal: -1 for a safety violation
                        cat = next(f["fault_type"] for f in state["findings"] if f["finding_id"] == r["finding_id"])
                        self.ranker.update(cat, r["candidate_id"], config.REWARDS["safety_violation"])
                notes.append(f"removed {issue['rec_id']}: {issue['detail']} (reward -1 recorded)")
        return {"recommendations": recs, "summary": summary, "summary_source": "template" if summary != state["summary"]
                else state.get("summary_source"), "loop": state["loop"] + 1,
                "revisions": state["revisions"] + [{"loop": state["loop"] + 1, "changes": notes}]}

    def node_report(self, state):
        if state.get("status") == "refused":
            r = state["refusal"]
            report = (f"# Request refused\n\n**Rule {r['rule_id']}:** {r['reason']}\n\n"
                      f"Request: _{state['request']}_\n\n> {config.ADVISORY_BANNER}\n")
            return {"report": report}
        if state.get("intent") == "explain":
            ev = (state.get("evidence") or {}).get("question", {"status": "NO_EVIDENCE", "passages": []})
            state = {**state, "answer": self._explain_answer(state["request"], ev)}
            return {"report": build_report(self, state), "answer": state["answer"],
                    "status": "answered" if ev["status"] == "OK" else "no_evidence"}
        decision = (state.get("review") or {}).get("decision", "N/A")
        ranks = {}
        for r in state.get("recommendations", []):
            if r["status"] != "removed":
                ranks[r["finding_id"]] = ranks.get(r["finding_id"], 0) + 1
                r["rank"] = ranks[r["finding_id"]]          # renumber after removals
                f = next(x for x in state["findings"] if x["finding_id"] == r["finding_id"])
                r["evidence_ids"] = f.get("evidence_ids", [])
                r["review_decision"] = decision
                r["status"] = "pending_human_approval"
                self.memory.save_recommendation(self.run_id, r)
        status = {"APPROVE": "approved_by_reviewer", "ESCALATE": "escalated_to_human"}.get(decision)
        if status is None:
            status = state.get("status") if state.get("status") in ("data_rejected", "analyst_failed") else "no_findings"
        return {"report": build_report(self, state), "status": status}

    def _explain_answer(self, question, ev):
        """Short answer grounded ONLY in the retrieved passages; every citation must be a real passage."""
        if ev["status"] != "OK" or not self.llm.available:
            return None
        ids = [f"[{p['source']} p.{p['page']}]" for p in ev["passages"]]
        ctx = "\n".join(f"{i} {p['text']}" for i, p in zip(ids, ev["passages"]))
        try:
            ans = self.llm.text("Answer the HVAC question in 2-3 sentences using ONLY the passages. Cite each "
                                "fact with the passage tag in square brackets exactly as given. If the passages "
                                "do not answer the question, reply exactly: NO_EVIDENCE.",
                                f"Question: {question}\nPassages:\n{ctx}", max_tokens=250)
        except LLMUnavailable:
            return None
        cited = set(re.findall(r"\[[^\]]+ p\.\d+\]", ans))
        if "NO_EVIDENCE" in ans or not cited or not cited <= set(ids):
            return None            # uncited or invented citation -> show the passages only
        return ans

    # ================================================================ routing
    @staticmethod
    def route_after_guard(state):
        return "report" if state.get("status") == "refused" else "plan"

    @staticmethod
    def route_after_plan(state):
        return "retrieve" if state["intent"] == "explain" else "analyze"

    @staticmethod
    def route_after_analyze(state):
        if not state.get("findings"):
            return "report"
        needs_rag = any(f["severity"] in ("medium", "high") for f in state["findings"])
        return "retrieve" if needs_rag else "recommend"

    @staticmethod
    def route_after_retrieve(state):
        return "report" if state["intent"] == "explain" else "recommend"

    @staticmethod
    def route_after_review(state):
        if state["review"]["decision"] == "REVISE" and state["loop"] < config.MAX_REVIEW_LOOPS:
            return "revise"
        return "report"

    # ================================================================ engines
    def _nodes(self):
        return {"guard": self.node_guard, "plan": self.node_plan, "analyze": self.node_analyze,
                "retrieve": self.node_retrieve, "recommend": self.node_recommend, "review": self.node_review,
                "revise": self.node_revise, "report": self.node_report}

    def _routes(self):
        return {"guard": self.route_after_guard, "plan": self.route_after_plan, "analyze": self.route_after_analyze,
                "retrieve": self.route_after_retrieve, "recommend": lambda s: "review",
                "review": self.route_after_review, "revise": lambda s: "review", "report": lambda s: None}

    def _run_loop(self, state):
        nodes, routes, node = self._nodes(), self._routes(), "guard"
        for _ in range(30):
            state = {**state, **nodes[node](state)}
            node = routes[node](state)
            if node is None:
                return state
        raise RuntimeError("graph did not finish in 30 steps")

    def _build_langgraph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            return None
        g = StateGraph(RunState)
        for name, fn in self._nodes().items():
            g.add_node(name, fn)
        g.set_entry_point("guard")
        g.add_conditional_edges("guard", self.route_after_guard, {"report": "report", "plan": "plan"})
        g.add_conditional_edges("plan", self.route_after_plan, {"retrieve": "retrieve", "analyze": "analyze"})
        g.add_conditional_edges("analyze", self.route_after_analyze,
                                {"report": "report", "retrieve": "retrieve", "recommend": "recommend"})
        g.add_conditional_edges("retrieve", self.route_after_retrieve, {"report": "report", "recommend": "recommend"})
        g.add_edge("recommend", "review")
        g.add_conditional_edges("review", self.route_after_review, {"revise": "revise", "report": "report"})
        g.add_edge("revise", "review")
        g.add_edge("report", END)
        return g.compile()

    # ================================================================ helpers
    def _footer(self, state):
        m = state.get("metrics", {})
        return (f"\n---\n_Run {self.run_id} | engine: {self.engine} | LLM: {self.llm.mode} "
                f"({config.LLM_MODEL if self.llm.mode == 'anthropic' else 'no API calls'}) | latency {m.get('latency_s')} s | "
                f"LLM calls {m.get('llm_calls')} | tokens {m.get('tokens_in')}/{m.get('tokens_out')} | "
                f"ACK rate {m.get('ack_rate')} | system v{VERSION}_\n")

    def _error_report(self, state):
        return (f"# Run stopped safely\n\nThe run could not finish: {'; '.join(state['errors'])}\n\n"
                f"No recommendation was produced. A human should review the data.\n\n> {config.ADVISORY_BANNER}\n")

    def _db_failure(self, request, exc):
        state = {"request": request, "status": "error", "errors": [f"database unavailable: {exc}"]}
        state["report"] = self._error_report(state)
        state["run_id"] = None
        self._save_report_file(state["report"])        # the report is not lost even without the database
        return state

    def _save_report_file(self, text):
        try:
            REPORTS_DIR.mkdir(exist_ok=True)
            name = f"{self.run_id or 'no-db-' + str(int(time.time()))}.md"
            (REPORTS_DIR / name).write_text(text)
        except OSError:
            pass


# ==================================================================== report
def build_report(system, s):
    out = ["# HVAC Operations Assistant: Advisory Report", "", f"> **{config.ADVISORY_BANNER}**", ""]
    sources = (s.get("analyst") or {}).get("sources") or {}
    out += [f"**Request:** {s['request']}  ", f"**Intent:** {s['intent']}  ",
            f"**Plan ({s['plan']['source']}):** {', '.join(s['plan']['branches'])}: {s['plan']['reason']}  "]
    for branch, src in sources.items():
        out.append(f"**Data ({branch}):** {src}  ")
    out.append("")
    if s["intent"] == "explain":
        ev = s.get("evidence", {}).get("question", {"status": "NO_EVIDENCE", "passages": []})
        out.append("## Answer from the technical manuals")
        if s.get("answer"):
            out += [s["answer"], "", "Passages used:"]
        if ev["status"] != "OK":
            out.append(f"NO_EVIDENCE: the manuals in the library do not cover this question ({ev.get('reason', '')}). "
                       "I will not guess.")
        for p in ev.get("passages", []):
            out.append(f"- **{p['source']}, page {p['page']}** (score {p['score']}): {p['text'][:300]}...")
        return "\n".join(out) + "\n"

    out.append("## Summary")
    out.append(f"{s.get('summary', '')} _(written by: {s.get('summary_source', 'template')})_")
    failed = (s.get("analyst") or {}).get("failed_branches") or {}
    if failed:
        out.append("\n**Stopped branches:** " + json.dumps(failed))
    findings = s.get("findings", [])
    if findings:
        out += ["", "## Findings", "| ID | Type | Fault | Severity | Window | Key metric | Evidence |", "|---|---|---|---|---|---|---|"]
        for f in findings:
            m = f["metrics"]
            key = f"score/tau {m['score_to_tau']}" if f["kind"] == "maintenance" else f"{m['deviation_pct']}% vs baseline"
            ev = ", ".join(f.get("evidence_ids", [])) or ("NO_EVIDENCE" if f["severity"] != "low" else "not searched (low)")
            out.append(f"| {f['finding_id']} | {f['kind']} | {f['fault_type']} | {f['severity']} | {f['start']} to {f['end']} | {key} | {ev} |")
        out += ["", "## Evidence from the manuals"]
        for fid, ev in (s.get("evidence") or {}).items():
            if ev["status"] != "OK":
                out.append(f"- {fid}: NO_EVIDENCE ({ev.get('reason', '')})")
            for p in ev.get("passages", [])[:2]:
                out.append(f"- {fid}: **{p['source']} p.{p['page']}** (score {p['score']}): {p['text'][:220]}...")
        recs = [r for r in s.get("recommendations", []) if r["status"] != "removed"]
        out += ["", "## Recommendations (ranked by learned score, pending human approval)"]
        for r in recs:
            tag = " _(exploration)_" if r.get("explored") else ""
            out.append(f"- `{r['rec_id']}` [{r['finding_id']} #{r['rank']}, Q={r['q_value']}]{tag} {r['text']}")
    rv = s.get("review")
    if rv:
        out += ["", f"## Safety review: **{rv['decision']}**"]
        for rev in s.get("revisions", []):
            out.append(f"- Revision loop {rev['loop']}: {'; '.join(rev['changes']) or 'no change'}")
        for e in rv["escalations"]:
            out.append(f"- ESCALATED: {e}")
        out.append(f"- Reviewer note: {rv['note']}")
    trace = (s.get("analyst") or {}).get("trace", [])
    if trace:
        out += ["", "## Reasoning trace (Analyst, ReAct)"]
        for t in trace:
            out.append(f"{t['step']}. Thought: {t['thought']} -> Action: {t['action']} -> Observation: {t.get('observation', '')}")
    tools = (s.get("analyst") or {}).get("tool_outputs", {})
    out += ["", "## Limits and uncertainty",
            "- Anomaly scores are not failure probabilities; fault types are rule-based hints with several possible causes.",
            "- Energy deviations are leads, not proof of inefficiency.",
            "- Validated on building HVAC data (LBNL simulation or synthetic demo data), not on cruise ship data.",
            f"- Models: {tools.get('detect_anomalies', {}).get('backend', 'n/a')} (tau {tools.get('detect_anomalies', {}).get('tau', 'n/a')}), "
            f"RAG backend: {system.library.backend}, embeddings {config.EMBED_MODEL}."]
    if s.get("errors"):
        out.append(f"- Errors handled during the run: {s['errors']}")
    return "\n".join(out) + "\n"
