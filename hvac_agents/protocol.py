"""
protocol.py - How the agents talk to each other (Area 4, owner: Luiz Paludo).

Message types (from the proposal): TASK, ACK, RESULT, NACK, ERROR, ESCALATION.

Rules
  1. The Coordinator sends a TASK. The receiver must answer ACK before working.
  2. When done the receiver sends RESULT (or NACK with reasons if it refuses
     the input, e.g. data validation failed).
  3. Each TASK carries `depends_on`: the msg_id of the upstream RESULT it uses,
     so Knowledge works from the Analyst's findings and the Reviewer checks both.
  4. The Coordinator waits up to ACK_TIMEOUT_S (60 s). On a timeout or a crash
     it retries MAX_RETRIES (2) times, then the caller uses its fallback.
Every message is written to the agent_messages table (the ACK log).
"""
import concurrent.futures as cf
import time

from . import config


class AgentNack(Exception):
    """Raised by an agent that refuses its input (e.g. invalid data). Not retried."""

    def __init__(self, reasons):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


class TaskFailed(Exception):
    """Raised when all retries are used. The caller must use its fallback."""


class MessageBus:
    def __init__(self, memory, run_id, timeout_s=None, retries=None):
        self.memory = memory
        self.run_id = run_id
        self.timeout_s = config.ACK_TIMEOUT_S if timeout_s is None else timeout_s
        self.retries = config.MAX_RETRIES if retries is None else retries
        self.agents = {}
        self.pool = cf.ThreadPoolExecutor(max_workers=2)

    def register(self, name, handler):
        """handler(payload) -> result dict."""
        self.agents[name] = handler

    def log(self, sender, receiver, msg_type, task, payload, depends_on=None):
        return self.memory.log_message(self.run_id, sender, receiver, msg_type, task, payload, depends_on)

    def request(self, sender, receiver, task, payload, depends_on=None):
        """Send a TASK and wait for RESULT. Returns (result, result_msg_id)."""
        last_error = None
        for attempt in range(self.retries + 1):
            self.log(sender, receiver, "TASK", task, {"attempt": attempt + 1, **_short(payload)}, depends_on)
            handler = self.agents.get(receiver)
            if handler is None:
                last_error = f"unknown agent {receiver}"
                self.log(receiver, sender, "ERROR", task, {"error": last_error}, depends_on)
                break
            # The receiver acknowledges BEFORE it starts working.
            self.log(receiver, sender, "ACK", task, {"attempt": attempt + 1}, depends_on)
            future = self.pool.submit(handler, payload)
            try:
                result = future.result(timeout=self.timeout_s)
                msg_id = self.log(receiver, sender, "RESULT", task, _short(result), depends_on)
                return result, msg_id
            except AgentNack as nack:
                self.log(receiver, sender, "NACK", task, {"reasons": nack.reasons}, depends_on)
                raise
            except cf.TimeoutError:
                last_error = f"timeout after {self.timeout_s}s"
            except Exception as exc:  # noqa: BLE001 - any agent crash is logged and retried
                last_error = f"{type(exc).__name__}: {exc}"
            self.log(receiver, sender, "ERROR", task, {"attempt": attempt + 1, "error": last_error}, depends_on)
            time.sleep(0.1 * (attempt + 1))
        raise TaskFailed(f"{receiver}/{task} failed after {self.retries + 1} attempts: {last_error}")

    def escalate(self, sender, reason, depends_on=None):
        return self.log(sender, "human", "ESCALATION", "escalate", {"reason": reason}, depends_on)


def _short(obj):
    """Keep the logged payload small (no big data frames in the log)."""
    if not isinstance(obj, dict):
        return {"value": str(obj)[:500]}
    out = {}
    for k, v in obj.items():
        if hasattr(v, "shape"):          # pandas / numpy objects
            out[k] = f"<{type(v).__name__} shape={getattr(v, 'shape', '?')}>"
        elif isinstance(v, (list, tuple)) and len(v) > 10:
            out[k] = f"<list of {len(v)}>"
        else:
            out[k] = v
    return out
