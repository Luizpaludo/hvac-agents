"""
config.py - All settings of the HVAC multi-agent system in one place.

ITAI 2376 | Group 2 (14562)
Group members: Mary Ann Mastri, Luiz Paludo, Cesar Antonio Noriega Martinez
Agent architecture, message protocol, database and safety rules: Luiz Paludo

Secrets (API keys, database passwords) are NEVER written here.
They are read from environment variables or a gitignored .env file.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("HVAC_DATA_DIR", ROOT / "data"))
CHUNKS_FILE = DATA_DIR / "chunks.json"          # from Cesar's RAG notebook (Part 8)
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"   # from Cesar's RAG notebook (Part 7)


def load_dotenv(path=ROOT / ".env"):
    """Tiny .env reader (KEY=VALUE per line) so we do not need an extra library."""
    p = Path(path)
    if not p.exists():
        return False
    # utf-8-sig removes the invisible BOM some Windows editors add at the start of the file
    for line in p.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = line.strip().lstrip("\ufeff")
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if v and not os.environ.get(k):       # an empty system variable must not hide the .env value
                os.environ[k] = v
    return True


def env_diagnosis():
    """Explain (without showing the key) why the LLM is or is not active."""
    env_file = ENV_FILE_USED or ROOT / ".env"
    similar = [f.name for f in ROOT.glob(".env*") if f.name not in (".env", ".env.txt", ".env.example")]
    try:
        import anthropic  # noqa: F401
        pkg = True
    except ImportError:
        pkg = False
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not env_file.exists() and not key:
        msg = ".env file not found in " + str(ROOT)
        if similar:
            msg += f" (found {similar}: rename it to exactly .env)"
        return msg
    if not key:
        return ".env found but it has no ANTHROPIC_API_KEY=... line"
    if not key.startswith("sk-ant-"):
        return "the key in .env does not start with sk-ant-"
    if not pkg:
        return "key found, but the 'anthropic' library is not installed (run: python -m pip install anthropic)"
    return f"key found in {env_file.name} and anthropic library installed"


# Read .env; also accept ".env.txt" because Windows Notepad often adds ".txt" without showing it.
ENV_FILE_USED = next((f for f in (ROOT / ".env", ROOT / ".env.txt") if load_dotenv(f)), None)

# ---------------------------------------------------------------- LLM
# Assumption agreed for the cost plan: Claude Haiku 5.5. With no key the system
# runs in TEMPLATE mode (deterministic text, no API cost) so it always works.
LLM_MODEL = os.getenv("HVAC_LLM_MODEL", "claude-haiku-5-5")
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 600          # token cap per call (cost control)
LLM_TIMEOUT_S = 30

# ---------------------------------------------------------------- dataset
# "lbnl" = real LBNL FDD dataset (data/LBNL_FDD_Dataset_SDAHU), "demo" = synthetic practice data,
# "auto" = LBNL if it is in the data folder, otherwise demo.
DATASET = os.getenv("HVAC_DATASET", "auto")

# ---------------------------------------------------------------- memory
# DATABASE_URL set (postgresql://...) -> PostgreSQL + pgvector; otherwise SQLite file.
DATABASE_URL = os.getenv("DATABASE_URL", "")
SQLITE_PATH = os.getenv("HVAC_SQLITE_PATH", str(ROOT / "hvac_memory.db"))

# ---------------------------------------------------------------- protocol
ACK_TIMEOUT_S = 60            # proposal: 60 s timeout
MAX_RETRIES = 2               # proposal: 2 retries, then fallback
MAX_REVIEW_LOOPS = 2          # proposal: REVISE at most 2 loops

# ---------------------------------------------------------------- RAG
MIN_SCORE = 0.42              # tuned by Cesar in Part 10 (embedding cosine)
TFIDF_MIN_SCORE = 0.12        # only used if sentence-transformers is not installed
TOP_K = 4
EMBED_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------- input guard
MAX_UPLOAD_MB = 20
ALLOWED_EXTENSIONS = {".csv"}

# ---------------------------------------------------------------- analyst
MIN_ENERGY_HISTORY_DAYS = 90  # under 90 days -> "no baseline"
AE_PERCENTILE = 99.0          # tau starts at 99th pct of fault-free validation error
MIN_EVENT_ROWS = 4            # 4 x 15 min = 1 hour of abnormal data to raise a finding
ENERGY_DEVIATION_PCT = 15.0   # daily deviation that counts as a lead

# Comfort / safety limits the Safety Reviewer checks (degC, inWC, %).
# A recommendation that touches a parameter NOT in this table is ESCALATED.
LIMITS = {
    "supply_air_temp_setpoint": (10.0, 18.0),
    "zone_temp": (20.0, 26.0),
    "duct_static_pressure_setpoint": (0.5, 2.5),
    "min_outdoor_air_damper": (15.0, 100.0),
}

# ---------------------------------------------------------------- reinforcement learning
LEARNING_RATE = 0.2           # Q(c) <- Q(c) + 0.2 (r - Q(c))
EPSILON = 0.10                # 10 % exploration
REWARDS = {
    "accepted": 1.0,          # confirmed fault / accepted recommendation
    "modified": 0.5,          # accepted after the human changed it
    "rejected": -0.5,
    "false_alarm": -0.5,
    "missed_fault": -1.0,
    "safety_violation": -1.0,
}
SEED = 42

ADVISORY_BANNER = "ADVISORY ONLY: human approval required. This system never controls equipment."
