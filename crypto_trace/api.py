"""
api.py
──────
FastAPI backend exposing the CryptoTrace engine via HTTP.

Endpoints:
  POST /trace   – BFS trace from a wallet address
  GET  /health  – liveness check

Run:
  uvicorn crypto_trace.api:app --reload --host 0.0.0.0 --port 8000
"""
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from .config import MAX_DEPTH, MAX_NODES
from .graph_builder import trace_wallet
from .model import scorer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model once at startup."""
    log.info("Loading DirtyScorer model…")
    try:
        scorer.load()
        log.info("Model loaded successfully.")
    except FileNotFoundError as exc:
        log.warning(f"Model not found: {exc}. Run the ML pipeline first.")
    yield
    log.info("Shutting down CryptoTrace API.")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CryptoTrace API",
    description="Blockchain Forensic Dashboard — BFS wallet tracing + ML dirty scoring",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response schemas ───────────────────────────────────────────────

class TraceRequest(BaseModel):
    address: str = Field(..., description="Ethereum wallet address (0x…)")
    maxDepth: int = Field(default=5, ge=1, le=MAX_DEPTH)
    maxNodes: int = Field(default=200, ge=1, le=MAX_NODES)

    @field_validator("address")
    @classmethod
    def validate_eth_address(cls, v: str) -> str:
        v = v.strip().lower()
        if not v.startswith("0x") or len(v) != 42:
            raise ValueError("Invalid Ethereum address. Must be 0x followed by 40 hex chars.")
        return v


class HealthResponse(BaseModel):
    status: str
    modelLoaded: bool


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health_check() -> dict:
    return {
        "status": "ok",
        "modelLoaded": scorer._pipeline is not None,
    }


@app.post("/trace", tags=["Forensics"])
async def trace(req: TraceRequest) -> dict[str, Any]:
    """
    Perform a recursive BFS trace from the given Ethereum address.

    Returns a react-force-graph compatible JSON graph with dirty scores.
    """
    if scorer._pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Run the ML pipeline (data_prep → feature_selection → model) first.",
        )
    try:
        graph = await trace_wallet(
            root_address=req.address,
            max_depth=req.maxDepth,
            max_nodes=req.maxNodes,
        )
        return graph
    except Exception as exc:
        log.exception(f"Trace failed for {req.address}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/score", tags=["Forensics"])
async def score_address(address: str) -> dict[str, Any]:
    """
    Quick dirty-score for a single address (without full BFS).
    Useful for UI previews before launching a full trace.
    """
    address = address.strip().lower()
    if not address.startswith("0x") or len(address) != 42:
        raise HTTPException(status_code=400, detail="Invalid Ethereum address.")

    from .etherscan_client import EtherscanClient
    from .graph_builder import _compute_features

    async with EtherscanClient() as client:
        normal_txs, token_txs = await asyncio.gather(
            client.get_normal_txs(address),
            client.get_token_txs(address),
        )

    features = _compute_features(normal_txs, token_txs, address)
    result = scorer.compute_dirty_score(features)
    result["address"] = address
    result["txCount"] = len(normal_txs) + len(token_txs)
    return result


# ─── Missing import in score endpoint ─────────────────────────────────────────
import asyncio  # noqa: E402 — placed after endpoint def to avoid circular
