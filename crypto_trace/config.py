"""
config.py
─────────
Central configuration: reads .env and exposes typed settings to all modules.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "data/processed/model.pkl")
FEATURES_CONFIG_PATH = BASE_DIR / os.getenv(
    "FEATURES_CONFIG_PATH", "data/processed/features_15.json"
)

# ─── Etherscan ───────────────────────────────────────────────────────────────
ETHERSCAN_API_KEY: str = os.getenv("ETHERSCAN_API_KEY", "")
ETHERSCAN_BASE_URL: str = "https://api.etherscan.io/v2/api"
ETHERSCAN_MAX_CALLS_PER_SEC: int = int(os.getenv("ETHERSCAN_MAX_CALLS_PER_SEC", "5"))
ETHERSCAN_MAX_TX_PER_WALLET: int = int(os.getenv("ETHERSCAN_MAX_TX_PER_WALLET", "100"))

# ─── BFS Tracer ──────────────────────────────────────────────────────────────
MAX_NODES: int = int(os.getenv("MAX_NODES", "500"))
MAX_DEPTH: int = int(os.getenv("MAX_DEPTH", "10"))
TRACE_TIMEOUT_SECONDS: int = int(os.getenv("TRACE_TIMEOUT_SECONDS", "30"))

# "Supernode" heuristic: if a wallet has more than this many txs, don't recurse
SUPERNODE_TX_THRESHOLD: int = 1_000

# ─── Risk Scoring ────────────────────────────────────────────────────────────
DIRTY_SCORE_HIGH: float = 0.30
DIRTY_SCORE_MED: float = 0.15

# ─── ML ──────────────────────────────────────────────────────────────────────
TOP_K_FEATURES: int = 15
RANDOM_STATE: int = 42
N_ESTIMATORS: int = 200
