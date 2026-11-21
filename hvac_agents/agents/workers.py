"""
workers.py - The three specialist agents: Analyst, Knowledge (RAG), Safety Reviewer.

Agent patterns (from the proposal):
  Analyst      -> ReAct  (Thought -> Action(tool) -> Observation, repeated)
  Knowledge    -> ReAct  (build query from the Analyst's findings -> search -> observe)
  Reviewer     -> Reflection (check the draft, then APPROVE / REVISE / ESCALATE)
The LLM chooses actions and writes text; CODE does every calculation and every
safety check. If the LLM is unavailable or answers badly, each agent falls back
to its deterministic policy, so the system still works (and says so).
"""
import json
import re
import time

from .. import config
from ..llm import LLMUnavailable
from ..protocol import AgentNack
from ..tools.anomaly import detect_anomalies
from ..tools.data_tools import validate_data
from ..tools.energy import energy_baseline
from .ranker import FAULT_QUERIES, LIBRARY


# ======================================================================= Analyst
class AnalystAgent:
    """ReAct agent over the data tools. Tools: validate_data, detect_anomalies, energy_baseline."""

    SYSTEM = ("You are the Analyst agent of an HVAC decision-support system. You choose the next tool. "
              "Rules: always run validate_data on a dataset before analysing it; use detect_anomalies only for "
              "the maintenance branch and energy_baseline only for the energy branch; never do math yourself. "
              'Reply as JSON: {"thought": "...", "action": "<tool or finish>", "branch": "maintenance|energy|none"}')

    def __init__(self, memory, llm, detectors, run_id_getter):
        self.memory, self.llm, self.detectors = memory, llm, detectors
        self.run_id = run_id_getter

    def _call(self, tool, fn, *args, **kw):
        t0 = time.time()
        try:
            out = fn(*args, **kw)
            self.memory.log_tool(self.run_id(), "analyst", tool, {"branch": kw.get("kind", "")}, True,
                                 _summary(out), time.time() - t0)
            return out
        except Exception as exc:
            self.memory.log_tool(self.run_id(), "analyst", tool, {}, False, str(exc), time.time() - t0)
            raise

    def _policy(self, todo, state):
        """Deterministic ReAct policy (also the fallback): validate -> analyse -> finish."""
        for b in todo:
            if b not in state["validated"]:
                return {"thought": f"I must check the {b} data before using it.", "action": "validate_data", "branch": b}
            if state["validated"][b] and b not in state["analysed"]:
                tool = "detect_anomalies" if b == "maintenance" else "energy_baseline"
                return {"thought": f"The {b} data passed validation, so I can analyse it.", "action": tool, "branch": b}
        return {"thought": "All branches are validated and analysed.", "action": "finish", "branch": "none"}

    def __call__(self, payload):
        todo = [b for b in payload["branches"] if b in ("maintenance", "energy")]
        datasets = payload["datasets"]
        state = {"validated": {}, "analysed": {}, "reasons": {}}
        trace, findings, outputs = [], [], {}
        for step in range(6):                                 # ReAct loop, max 6 steps
            expected = self._policy(todo, state)
            decision, source = expected, "policy"
            if self.llm.available:
                try:
                    d = self.llm.json(self.SYSTEM, json.dumps({"request": payload["request"], "branches": todo,
                                      "validated": state["validated"], "analysed": list(state["analysed"]),
                                      "allowed_actions": ["validate_data", "detect_anomalies", "energy_baseline", "finish"]}),
                                      required_keys=("thought", "action", "branch"))
                    # guardrail: the LLM may explain in its own words, but an action that breaks
                    # the rules (e.g. analyse before validate) is replaced by the policy action.
                    same = d["action"] == expected["action"] and (d["action"] == "finish" or d["branch"] == expected["branch"])
                    if same:
                        d["branch"] = expected["branch"]
                        decision, source = d, "llm"
                    else:
                        decision = {**expected, "thought": expected["thought"] + " (LLM choice corrected by rule)"}
                except LLMUnavailable:
                    pass
            act, b = decision["action"], decision["branch"]
            obs = ""
            if act == "finish":
                trace.append({"step": step + 1, "thought": decision["thought"], "action": "finish", "source": source})
                break
            if act == "validate_data":
                extra = ()
                if b == "maintenance":
                    det = (self.detectors or {}).get("autoencoder") or (self.detectors or {}).get("isolation_forest")
                    if det is not None and getattr(det.features, "mode", "demo") == "lbnl":
                        extra = ("ccv_cmd", "damper_cmd")
                v = self._call("validate_data", validate_data, datasets[b],
                               kind="ahu" if b == "maintenance" else "energy", extra_columns=extra)
                state["validated"][b] = v["ok"]
                state["reasons"][b] = v["reasons"]
                outputs[f"validate_{b}"] = v
                obs = "PASS" if v["ok"] else f"FAIL: {v['reasons']}"
                if not v["ok"]:
                    state["analysed"][b] = False                 # this branch stops here
            elif act == "detect_anomalies":
                r = self._call("detect_anomalies", detect_anomalies, datasets[b], self.detectors,
                               payload.get("detector", "autoencoder"))
                outputs["detect_anomalies"] = {k: v for k, v in r.items() if k not in ("flags", "scores")}
                findings += r["findings"]
                state["analysed"][b] = True
                obs = f"{len(r['findings'])} findings with {r['model']} (tau {r['tau']})" + \
                      (" [fallback to Isolation Forest]" if r["fallback_used"] else "") + \
                      (f" [{r['skipped_rows_missing_values']} rows with missing values skipped, not filled]"
                       if r.get("skipped_rows_missing_values") else "")
            elif act == "energy_baseline":
                r = self._call("energy_baseline", energy_baseline, datasets[b])
                outputs["energy_baseline"] = r
                findings += r["findings"]
                state["analysed"][b] = True
                obs = (f"CV(RMSE) {r['metrics']['cv_rmse_pct']}%, NMBE {r['metrics']['nmbe_pct']}%, "
                       f"{len(r['findings'])} leads") if r["ok"] else f"no baseline: {r['reason']}"
            trace.append({"step": step + 1, "thought": decision["thought"], "action": f"{act}({b})",
                          "observation": obs, "source": source})

        failed = {b: state["reasons"][b] for b in todo if state["validated"].get(b) is False}
        if todo and len(failed) == len(todo):
            raise AgentNack([f"{b}: {'; '.join(r)}" for b, r in failed.items()])
        for i, f in enumerate(findings):
            f["finding_id"] = f"F{i + 1}"
        synthetic = any(o.get("synthetic") for k, o in outputs.items() if k.startswith("validate_"))
        sources = {k.replace("validate_", ""): o.get("source") for k, o in outputs.items() if k.startswith("validate_")}
        return {"findings": findings, "trace": trace, "tool_outputs": outputs, "failed_branches": failed,
                "synthetic": synthetic, "sources": sources, "summary": template_summary(findings, failed)}


def template_summary(findings, failed):
    """Summary written from tool numbers only (also used when the LLM text fails review)."""
    if not findings:
        lines = ["No abnormal operation or energy deviation was found in the analysed window."]
    else:
        lines = []
        for f in findings:
            m = f["metrics"]
            if f["kind"] == "maintenance":
                extra = f", {m['events']} events, {m['hours']} hours" if "events" in m else ""
                lines.append(f"{f['finding_id']}: {f['fault_type']} ({f['severity']}) from {f['start']} to {f['end']}, "
                             f"score/tau {m['score_to_tau']}, {m['rows']} rows{extra}. Rule: {m['rule']}.")
            else:
                lines.append(f"{f['finding_id']}: energy use {m['deviation_pct']}% above baseline from {f['start']} "
                             f"to {f['end']} ({m['days']} days, {m['extra_kwh']} extra kWh).")
    for b, r in failed.items():
        lines.append(f"The {b} branch stopped: {'; '.join(r)}.")
    return " ".join(lines)


# ======================================================================= Knowledge
class KnowledgeAgent:
    """ReAct over search_manuals. Builds the query from the Analyst's findings."""

    SYSTEM = ("You are the Knowledge agent. Write ONE short search query (max 15 words) to find manual passages "
              'about the HVAC finding. Reply as JSON: {"thought": "...", "query": "..."}')

    def __init__(self, memory, llm, library, run_id_getter):
        self.memory, self.llm, self.library, self.run_id = memory, llm, library, run_id_getter

    def _search(self, query):
        t0 = time.time()
        res = self.library.search_manuals(query)
        self.memory.log_tool(self.run_id(), "knowledge", "search_manuals", {"query": query},
                             res["status"] == "OK", f"{res['status']} {len(res['passages'])} passages", time.time() - t0)
        return res

    def __call__(self, payload):
        out, trace = {}, []
        if payload.get("question"):                       # "explain" intent: answer the user's question
            res = self._search(payload["question"])
            trace.append({"thought": "The user asked a direct question; search the manuals with it.",
                          "action": "search_manuals", "observation": res["status"]})
            return {"evidence": {"question": res}, "trace": trace}
        for f in payload["findings"]:
            if f["severity"] not in ("medium", "high"):
                continue                                   # proposal: retrieve for medium severity or higher
            query = FAULT_QUERIES.get(f["fault_type"], FAULT_QUERIES["unknown"]) + " " + f["metrics"].get("rule", "")[:80]
            thought = "Template query built from the fault type and rule."
            if self.llm.available:
                try:
                    d = self.llm.json(self.SYSTEM, json.dumps({"fault_type": f["fault_type"],
                                      "rule": f["metrics"].get("rule", ""), "kind": f["kind"]}),
                                      required_keys=("thought", "query"))
                    query, thought = str(d["query"])[:200], d["thought"]
                except LLMUnavailable:
                    pass
            res = self._search(query)
            trace.append({"finding": f["finding_id"], "thought": thought, "action": f"search_manuals('{query[:60]}...')",
                          "observation": res["status"]})
            out[f["finding_id"]] = res
        return {"evidence": out, "trace": trace}


# ======================================================================= Safety Reviewer
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def verify_numbers(text, allowed_source):
    """Every number in the draft text must appear in the tool outputs."""
    allowed = {round(float(x), 2) for x in NUM.findall(json.dumps(allowed_source, default=str))}
    bad = []
    for x in NUM.findall(text):
        v = round(float(x), 2)
        if not any(abs(v - a) <= 0.011 + 0.001 * abs(a) for a in allowed):
            bad.append(x)
    return bad


def check_limits(rec):
    """'ok' | 'violation' | 'unknown_limit' for one recommendation."""
    p = rec.get("parameter")
    if not p:
        return "ok", ""
    if p not in config.LIMITS:
        return "unknown_limit", f"no defined limit for '{p}'"
    lo, hi = config.LIMITS[p]
    v = rec.get("value")
    if v is None or not (lo <= v <= hi):
        return "violation", f"{p} = {v} is outside the comfort/safety range [{lo}, {hi}]"
    return "ok", ""


class SafetyReviewer:
    """Reflection agent: checks numbers, evidence and limits. Code decides; the LLM only adds a note."""

    SYSTEM = ("You are the Safety Reviewer of an HVAC decision-support system. In 2 sentences, reflect on whether "
              "the draft report is cautious and clear about uncertainty. Do not add new numbers.")

    def __init__(self, memory, llm, run_id_getter):
        self.memory, self.llm, self.run_id = memory, llm, run_id_getter

    def __call__(self, payload):
        issues, escalations, rec_checks = [], [], {}
        t0 = time.time()
        bad = verify_numbers(payload["summary"], {"findings": payload["findings"], "tools": payload["tool_outputs"]})
        if bad:
            issues.append({"type": "wrong_number", "detail": f"numbers not found in tool outputs: {bad}"})
        for rec in payload["recommendations"]:
            status, why = check_limits(rec)
            rec_checks[rec["rec_id"]] = status
            if status == "violation":
                issues.append({"type": "safety_violation", "rec_id": rec["rec_id"], "detail": why})
            elif status == "unknown_limit":
                escalations.append(f"{rec['rec_id']}: {why}")
        for f in payload["findings"]:
            if f["severity"] not in ("medium", "high"):
                continue
            ev = payload["evidence"].get(f["finding_id"], {"status": "NO_EVIDENCE", "passages": []})
            terms = LIBRARY.get(f["fault_type"], LIBRARY["unknown"])["evidence_terms"]
            supported = [p for p in ev.get("passages", []) if any(t in p["text"].lower() for t in terms)]
            f["evidence_ok"] = bool(supported)
            f["evidence_ids"] = sorted({p["evidence_id"] for p in supported})
            if not supported and f["severity"] == "high":
                escalations.append(f"{f['finding_id']}: high severity without supporting manual evidence")
        decision = "REVISE" if issues else ("ESCALATE" if escalations else "APPROVE")
        if issues and payload.get("loop", 0) >= config.MAX_REVIEW_LOOPS:
            decision = "ESCALATE"
            escalations.append("issues remain after the maximum of 2 revision loops")
        note = "Checks done by code: numbers vs tool outputs, comfort/safety limits, evidence sufficiency."
        if self.llm.available and decision != "REVISE" and payload["summary"].strip():
            try:
                note = self.llm.text(self.SYSTEM, payload["summary"][:1500], max_tokens=300)
                bad_note = verify_numbers(note, {"f": payload["findings"], "t": payload["tool_outputs"]})
                if bad_note:   # reflection must not introduce numbers either
                    note = "Reviewer note removed: it contained numbers not in the tool outputs."
            except LLMUnavailable:
                pass
        self.memory.log_tool(self.run_id(), "reviewer", "verify_numbers+check_limits", {"loop": payload.get("loop", 0)},
                             True, decision, time.time() - t0)
        return {"decision": decision, "issues": issues, "escalations": escalations,
                "rec_checks": rec_checks, "note": note, "findings": payload["findings"]}


def _summary(out):
    if isinstance(out, dict):
        keys = {k: out[k] for k in ("ok", "reasons", "model", "tau", "metrics") if k in out}
        if "findings" in out:
            keys["findings"] = len(out["findings"])
        return json.dumps(keys, default=str)[:900]
    return str(out)[:900]
