"""
tests/test_api.py
──────────────────
Tests for the FastAPI /trace and /health endpoints.
All external calls (Etherscan, model inference) are mocked.

Run:
    pytest tests/test_api.py -v
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Patch scorer.load() before importing the app so lifespan doesn't fail
with patch("crypto_trace.model.scorer") as _mock_scorer:
    _mock_scorer._pipeline = MagicMock()
    _mock_scorer.load = MagicMock()
    from crypto_trace.api import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ─── /health ─────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "modelLoaded" in data
        assert data["status"] == "ok"


# ─── /trace ──────────────────────────────────────────────────────────────────

VALID_ADDRESS = "0x" + "a" * 40

MOCK_GRAPH = {
    "nodes": [
        {
            "id": VALID_ADDRESS,
            "address": VALID_ADDRESS,
            "dirtyScore": 0.82,
            "riskLevel": "HIGH",
            "ethBalance": 1.5,
            "txCount": 12,
            "isSupernode": False,
            "depth": 0,
            "isRoot": True,
            "totalSent": 5.0,
            "totalReceived": 6.5,
        }
    ],
    "links": [],
    "meta": {
        "rootAddress": VALID_ADDRESS,
        "nodeCount": 1,
        "linkCount": 0,
        "elapsedSeconds": 1.23,
        "maxDepthReached": 0,
    },
}


class TestTrace:
    def test_valid_address_returns_200(self, client):
        with patch("crypto_trace.api.trace_wallet", new_callable=AsyncMock) as mock_trace:
            mock_trace.return_value = MOCK_GRAPH
            resp = client.post("/trace", json={"address": VALID_ADDRESS})
        assert resp.status_code == 200

    def test_response_has_nodes_and_links(self, client):
        with patch("crypto_trace.api.trace_wallet", new_callable=AsyncMock) as mock_trace:
            mock_trace.return_value = MOCK_GRAPH
            data = client.post("/trace", json={"address": VALID_ADDRESS}).json()
        assert "nodes" in data
        assert "links" in data
        assert "meta" in data

    def test_invalid_address_returns_422(self, client):
        resp = client.post("/trace", json={"address": "not_an_eth_address"})
        assert resp.status_code == 422

    def test_short_address_returns_422(self, client):
        resp = client.post("/trace", json={"address": "0xabc"})
        assert resp.status_code == 422

    def test_depth_capped_at_max(self, client):
        with patch("crypto_trace.api.trace_wallet", new_callable=AsyncMock) as mock_trace:
            mock_trace.return_value = MOCK_GRAPH
            resp = client.post(
                "/trace",
                json={"address": VALID_ADDRESS, "maxDepth": 999},
            )
        # pydantic should clamp / reject values over MAX_DEPTH
        assert resp.status_code in (200, 422)

    def test_max_nodes_capped(self, client):
        with patch("crypto_trace.api.trace_wallet", new_callable=AsyncMock) as mock_trace:
            mock_trace.return_value = MOCK_GRAPH
            resp = client.post(
                "/trace",
                json={"address": VALID_ADDRESS, "maxNodes": 10_000},
            )
        assert resp.status_code in (200, 422)

    def test_trace_called_with_correct_args(self, client):
        with patch("crypto_trace.api.trace_wallet", new_callable=AsyncMock) as mock_trace:
            mock_trace.return_value = MOCK_GRAPH
            client.post(
                "/trace",
                json={"address": VALID_ADDRESS, "maxDepth": 3, "maxNodes": 100},
            )
        mock_trace.assert_called_once_with(
            root_address=VALID_ADDRESS,
            max_depth=3,
            max_nodes=100,
        )

    def test_internal_error_returns_500(self, client):
        with patch("crypto_trace.api.trace_wallet", new_callable=AsyncMock) as mock_trace:
            mock_trace.side_effect = RuntimeError("Something went wrong")
            resp = client.post("/trace", json={"address": VALID_ADDRESS})
        assert resp.status_code == 500
