"""
guard.py - The Input Guard (safety rules, owner: Luiz Paludo).

Runs BEFORE any agent or LLM sees the request. Deliberately written as plain
rules (no LLM): rules cannot be talked out of their job by a clever prompt.

Rule IDs (logged with every refusal and shown to the user):
  G-01  empty or too long request
  G-02  prompt injection / attempt to override the system instructions
  G-03  request for secrets (API keys, passwords, tokens, .env)
  G-04  equipment control command (the system is decision support ONLY)
  G-05  out-of-scope decision (staffing, regulatory or certification decisions)
  G-06  not an HVAC request (outside the allowed intents)
  G-07  file rejected (only .csv, max 20 MB)
"""
import re
from pathlib import Path

from .. import config

INJECTION = [r"ignore (all |the )?(previous|prior|above) (instructions|rules)", r"system prompt",
             r"you are now", r"developer mode", r"jailbreak", r"disregard (your|the) (rules|instructions)",
             r"pretend (you are|to be)", r"act as (an? )?(admin|root|developer)", r"forget (your|all) (rules|instructions)"]
SECRETS = [r"api[ _-]?key", r"password", r"\btoken\b", r"\.env\b", r"secret key", r"credentials?"]
CONTROL = [r"\b(turn|switch|shut)\s+(it\s+)?(off|on|down)\b",
           r"\b(override|bypass|disable|silence|reset)\b.{0,25}\b(alarm|interlock|safety|controller|setpoint|chiller|fan|valve|damper|ahu)\b",
           r"\b(set|change|adjust|raise|lower|increase|decrease|put)\b.{0,30}\b(setpoint|set point|speed|valve|damper|temperature)\b.{0,25}\b(to|now|at)\b",
           r"^\s*(please\s+)?(start|stop|open|close|restart)\b",
           r"\b(start|stop|restart)\s+(the\s+)?(chiller|fan|pump|ahu|compressor|unit)\b"]
BOUNDARY = [r"\b(fire|hire|lay off|discipline)\b", r"\bstaff(ing)?\b", r"\bcertif(y|ication)\b",
            r"\bcompliance sign[- ]?off\b", r"\bregulator(y)? (approval|decision)\b", r"\bUSCG\b"]

MAINT = ["fault", "anomal", "abnormal", "maintenance", "damper", "valve", "fan", "belt", "sensor", "ahu",
         "air handl", "coil", "alarm", "diagnos", "hvac", "supply air", "mixed air", "static pressure", "economizer",
         "bearing", "filter", "drain pan", "sheave"]
ENERGY = ["energy", "kwh", "consumption", "electric", "power use", "usage", "baseline", "waste", "meter"]
EXPLAIN = ["explain", "what is", "what are", "how does", "how do", "how often", "why", "what causes", "how should", "how can"]


def screen(text):
    """Return {"allowed", "intent", "rule_id", "reason"}."""
    t = (text or "").strip()
    low = t.lower()
    if not t or len(t) > 2000:
        return _refuse("G-01", "The request is empty or longer than 2000 characters.")
    if any(re.search(p, low) for p in INJECTION):
        return _refuse("G-02", "The request tries to change my instructions. I only follow the system rules.")
    if any(re.search(p, low) for p in SECRETS):
        return _refuse("G-03", "I never show or handle secrets such as API keys or passwords.")
    if any(re.search(p, low) for p in CONTROL):
        return _refuse("G-04", "I am decision support only and never control equipment. "
                               "A qualified engineer must operate the system. I can analyse the data instead.")
    if any(re.search(p, low) for p in BOUNDARY):
        return _refuse("G-05", "Staffing, regulatory and certification decisions are outside my limits; "
                               "they belong to qualified personnel.")
    m = any(k in low for k in MAINT)
    e = any(k in low for k in ENERGY)
    x = any(low.startswith(k) or f" {k} " in f" {low} " for k in EXPLAIN)
    if not (m or e):
        return _refuse("G-06", "I only help with HVAC maintenance and energy questions.")
    if m and e:
        intent = "both"
    elif x and not any(w in low for w in ["check", "analy", "find", "detect", "look at", "review", "scan", "data"]):
        intent = "explain"
    else:
        intent = "maintenance" if m else "energy"
    return {"allowed": True, "intent": intent, "rule_id": None, "reason": "passed all input rules"}


def check_file(path, size_bytes=None):
    """G-07: only .csv files up to MAX_UPLOAD_MB."""
    p = Path(path)
    if p.suffix.lower() not in config.ALLOWED_EXTENSIONS:
        return _refuse("G-07", f"Only CSV files are accepted (got '{p.suffix}').")
    size = size_bytes if size_bytes is not None else (p.stat().st_size if p.exists() else 0)
    if size > config.MAX_UPLOAD_MB * 1024 * 1024:
        return _refuse("G-07", f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    return {"allowed": True, "rule_id": None, "reason": "file accepted"}


def _refuse(rule, reason):
    return {"allowed": False, "intent": None, "rule_id": rule, "reason": reason}
