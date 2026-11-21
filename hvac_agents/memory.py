"""
memory.py - The agents' memory and log (Area 5, owner: Luiz Paludo).

Stores every run, every message between agents (the ACK log), every tool call,
findings, recommendations, human feedback and the learned Q-values.

Two back-ends with the SAME tables:
  * PostgreSQL (+ pgvector, where Cesar's document_chunks table lives) when
    DATABASE_URL is set, e.g. in Colab after notebook Part 11.
  * SQLite (a single file, part of Python) otherwise, so the system always runs.
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone

from . import config

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY, created_at TEXT, user_request TEXT, intent TEXT,
        dataset TEXT, status TEXT, latency_s REAL, llm_calls INTEGER,
        tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, llm_mode TEXT)""",
    """CREATE TABLE IF NOT EXISTS agent_messages (
        msg_id TEXT PRIMARY KEY, run_id TEXT, created_at TEXT, sender TEXT,
        receiver TEXT, msg_type TEXT, task TEXT, depends_on TEXT, payload TEXT)""",
    """CREATE TABLE IF NOT EXISTS tool_calls (
        call_id TEXT PRIMARY KEY, run_id TEXT, created_at TEXT, agent TEXT,
        tool TEXT, args TEXT, ok INTEGER, summary TEXT, duration_s REAL)""",
    """CREATE TABLE IF NOT EXISTS findings (
        finding_id TEXT, run_id TEXT, kind TEXT, fault_type TEXT,
        severity TEXT, start_ts TEXT, end_ts TEXT, metrics TEXT, limitations TEXT,
        PRIMARY KEY (run_id, finding_id))""",
    """CREATE TABLE IF NOT EXISTS recommendations (
        rec_id TEXT PRIMARY KEY, run_id TEXT, finding_id TEXT, candidate_id TEXT,
        text TEXT, rank INTEGER, q_value REAL, evidence TEXT, review_decision TEXT)""",
    """CREATE TABLE IF NOT EXISTS feedback (
        feedback_id TEXT PRIMARY KEY, run_id TEXT, rec_id TEXT, created_at TEXT,
        outcome TEXT, reward REAL, simulated INTEGER)""",
    """CREATE TABLE IF NOT EXISTS q_values (
        category TEXT, candidate_id TEXT, q REAL, n INTEGER,
        PRIMARY KEY (category, candidate_id))""",
]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


class Memory:
    """Small database wrapper. All SQL uses '?' and is converted for PostgreSQL."""

    def __init__(self, url=None, sqlite_path=None):
        url = config.DATABASE_URL if url is None else url
        self.kind = "postgres" if url.startswith("postgres") else "sqlite"
        if self.kind == "postgres":
            import psycopg2  # only needed in the PostgreSQL set-up
            self.conn = psycopg2.connect(url)
            self.conn.autocommit = True
        else:
            path = sqlite_path or config.SQLITE_PATH
            self.conn = sqlite3.connect(path, check_same_thread=False)
        for stmt in SCHEMA:
            self._exec(stmt)

    # -- low level ----------------------------------------------------------
    def _exec(self, sql, params=()):
        if self.kind == "postgres":
            sql = sql.replace("?", "%s")
        cur = self.conn.cursor()
        cur.execute(sql, params)
        if self.kind == "sqlite":
            self.conn.commit()
        return cur

    def query(self, sql, params=()):
        cur = self._exec(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def ping(self):
        """Raises an error if the database is not reachable (used for TC-10)."""
        self._exec("SELECT 1")

    def close(self):
        self.conn.close()

    # -- writers ------------------------------------------------------------
    def start_run(self, request, dataset, llm_mode):
        run_id = new_id("run")
        self._exec("INSERT INTO runs (run_id, created_at, user_request, dataset, status, llm_mode) "
                   "VALUES (?,?,?,?,?,?)", (run_id, now(), request, dataset, "started", llm_mode))
        return run_id

    def finish_run(self, run_id, **fields):
        for k, v in fields.items():
            self._exec(f"UPDATE runs SET {k} = ? WHERE run_id = ?", (v, run_id))

    def log_message(self, run_id, sender, receiver, msg_type, task, payload, depends_on=None):
        msg_id = new_id("msg")
        self._exec("INSERT INTO agent_messages VALUES (?,?,?,?,?,?,?,?,?)",
                   (msg_id, run_id, now(), sender, receiver, msg_type, task, depends_on,
                    json.dumps(payload, default=str)[:4000]))
        return msg_id

    def log_tool(self, run_id, agent, tool, args, ok, summary, duration):
        self._exec("INSERT INTO tool_calls VALUES (?,?,?,?,?,?,?,?,?)",
                   (new_id("tool"), run_id, now(), agent, tool, json.dumps(args, default=str),
                    int(ok), str(summary)[:1000], round(duration, 3)))

    def save_finding(self, run_id, f):
        self._exec("INSERT INTO findings (finding_id, run_id, kind, fault_type, severity, start_ts, end_ts, "
                   "metrics, limitations) VALUES (?,?,?,?,?,?,?,?,?)",
                   (f["finding_id"], run_id, f["kind"], f.get("fault_type"), f["severity"],
                    f.get("start"), f.get("end"), json.dumps(f.get("metrics", {}), default=str),
                    f.get("limitations", "")))

    def save_recommendation(self, run_id, r):
        self._exec("INSERT INTO recommendations VALUES (?,?,?,?,?,?,?,?,?)",
                   (r["rec_id"], run_id, r["finding_id"], r["candidate_id"], r["text"], r["rank"],
                    r["q_value"], json.dumps(r.get("evidence_ids", [])), r.get("review_decision")))

    def save_feedback(self, run_id, rec_id, outcome, reward, simulated):
        self._exec("INSERT INTO feedback VALUES (?,?,?,?,?,?,?)",
                   (new_id("fb"), run_id, rec_id, now(), outcome, reward, int(simulated)))

    # -- Q-values (the learned policy) -----------------------------------------
    def get_q(self, category):
        rows = self.query("SELECT candidate_id, q, n FROM q_values WHERE category = ?", (category,))
        return {r["candidate_id"]: (r["q"], r["n"]) for r in rows}

    def set_q(self, category, candidate_id, q, n):
        self._exec("DELETE FROM q_values WHERE category = ? AND candidate_id = ?", (category, candidate_id))
        self._exec("INSERT INTO q_values VALUES (?,?,?,?)", (category, candidate_id, q, n))

    def reset_q(self):
        self._exec("DELETE FROM q_values")

    # -- metrics ----------------------------------------------------------------
    def ack_rate(self, run_id):
        """ACK rate = ACKed tasks / all tasks (proposal target 100 %)."""
        rows = self.query("SELECT msg_type, COUNT(*) AS n FROM agent_messages WHERE run_id = ? "
                          "GROUP BY msg_type", (run_id,))
        counts = {r["msg_type"]: r["n"] for r in rows}
        tasks = counts.get("TASK", 0)
        return (counts.get("ACK", 0) / tasks) if tasks else 1.0
