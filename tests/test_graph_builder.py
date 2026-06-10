"""
tests/test_graph_builder.py
────────────────────────────
Unit tests for the BFS tracer and feature computation.
Etherscan calls are fully mocked — no real API key needed.

Run:
    pytest tests/test_graph_builder.py -v
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crypto_trace.graph_builder import _compute_features, _wei


# ─── _wei helper ─────────────────────────────────────────────────────────────

class TestWeiConverter:
    def test_basic_conversion(self):
        assert _wei("1000000000000000000") == pytest.approx(1.0)
        assert _wei("500000000000000000") == pytest.approx(0.5)

    def test_zero(self):
        assert _wei("0") == 0.0

    def test_empty_string(self):
        assert _wei("") == 0.0

    def test_none(self):
        assert _wei(None) == 0.0

    def test_invalid(self):
        assert _wei("not_a_number") == 0.0


# ─── _compute_features ───────────────────────────────────────────────────────

ADDR = "0xabc123"

def _make_tx(from_addr, to_addr, value_eth, timestamp="1700000000"):
    """Helper to build a mock Etherscan transaction dict."""
    return {
        "from": from_addr,
        "to": to_addr,
        "value": str(int(value_eth * 1e18)),
        "timeStamp": timestamp,
        "hash": "0xdeadbeef",
        "gasUsed": "21000",
    }

def _make_token_tx(from_addr, to_addr, value_eth):
    return {
        "from": from_addr,
        "to": to_addr,
        "value": str(int(value_eth * 1e18)),
        "timeStamp": "1700000000",
        "hash": "0xeeeeeeee",
    }


class TestComputeFeatures:
    def test_returns_dict(self):
        features = _compute_features([], [], ADDR)
        assert isinstance(features, dict)

    def test_all_zeros_on_empty_txs(self):
        features = _compute_features([], [], ADDR)
        for k, v in features.items():
            assert v == 0.0, f"Feature '{k}' should be 0.0 on empty txs"

    def test_sent_count(self):
        txs = [
            _make_tx(ADDR, "0xrecipient1", 1.0, "1700000000"),
            _make_tx(ADDR, "0xrecipient2", 2.0, "1700001000"),
            _make_tx("0xother", ADDR, 0.5, "1700002000"),  # received, not sent
        ]
        features = _compute_features(txs, [], ADDR)
        assert features["Sent tnx"] == 2.0
        assert features["Received Tnx"] == 1.0

    def test_total_ether_sent(self):
        txs = [
            _make_tx(ADDR, "0xrec1", 1.5, "1700000000"),
            _make_tx(ADDR, "0xrec2", 2.5, "1700001000"),
        ]
        features = _compute_features(txs, [], ADDR)
        assert features["total Ether sent"] == pytest.approx(4.0, rel=1e-4)

    def test_total_ether_received(self):
        txs = [
            _make_tx("0xsender1", ADDR, 3.0, "1700000000"),
            _make_tx("0xsender2", ADDR, 1.0, "1700001000"),
        ]
        features = _compute_features(txs, [], ADDR)
        assert features["total ether received"] == pytest.approx(4.0, rel=1e-4)

    def test_avg_val_sent(self):
        txs = [
            _make_tx(ADDR, "0xrec1", 2.0, "1700000000"),
            _make_tx(ADDR, "0xrec2", 4.0, "1700001000"),
        ]
        features = _compute_features(txs, [], ADDR)
        assert features["avg val sent"] == pytest.approx(3.0, rel=1e-4)

    def test_time_diff_between_first_and_last(self):
        txs = [
            _make_tx(ADDR, "0xrec", 1.0, "1700000000"),   # ts = 0
            _make_tx(ADDR, "0xrec", 1.0, "1700003600"),   # ts = 60 min later
        ]
        features = _compute_features(txs, [], ADDR)
        assert features["Time Diff between first and last (Mins)"] == pytest.approx(60.0, rel=1e-3)

    def test_unique_sent_addresses(self):
        txs = [
            _make_tx(ADDR, "0xrec1", 1.0, "1700000000"),
            _make_tx(ADDR, "0xrec1", 1.0, "1700001000"),  # same recipient
            _make_tx(ADDR, "0xrec2", 1.0, "1700002000"),  # different recipient
        ]
        features = _compute_features(txs, [], ADDR)
        assert features["Unique Sent To Addresses"] == 2.0

    def test_erc20_sent_count(self):
        token_txs = [
            _make_token_tx(ADDR, "0xrec1", 100.0),
            _make_token_tx(ADDR, "0xrec2", 200.0),
        ]
        features = _compute_features([], token_txs, ADDR)
        assert features[" Total ERC20 tnxs"] == 2.0
        assert features[" ERC20 total ether sent"] == pytest.approx(300.0, rel=1e-4)

    def test_total_tx_count_in_features(self):
        txs = [_make_tx(ADDR, "0xrec", 1.0)] * 5
        features = _compute_features(txs, [], ADDR)
        assert features["total transactions (including tnx to create contract"] == 5.0


# ─── Etherscan Client (unit, mocked) ─────────────────────────────────────────

class TestEtherscanClientRateLimiting:
    """Verify the semaphore is initialized correctly."""

    def test_semaphore_value(self):
        from crypto_trace.etherscan_client import get_semaphore
        from crypto_trace.config import ETHERSCAN_MAX_CALLS_PER_SEC
        # Reset global semaphore for clean test
        import crypto_trace.etherscan_client as ec
        ec._semaphore = None
        sem = get_semaphore()
        assert sem._value == ETHERSCAN_MAX_CALLS_PER_SEC

    def test_supernode_detection(self):
        from crypto_trace.etherscan_client import EtherscanClient
        from crypto_trace.config import SUPERNODE_TX_THRESHOLD
        client = EtherscanClient()
        assert client.is_supernode(SUPERNODE_TX_THRESHOLD) is True
        assert client.is_supernode(SUPERNODE_TX_THRESHOLD - 1) is False
        assert client.is_supernode(0) is False


# ─── BFS Integration (fully mocked) ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_trace_wallet_returns_correct_structure():
    """
    Full BFS trace with mocked Etherscan responses and mocked scorer.
    Verifies the JSON output schema expected by react-force-graph.
    """
    root = "0x" + "a" * 40
    child = "0x" + "b" * 40

    mock_normal_txs = [
        {
            "from": root,
            "to": child,
            "value": str(int(1.0 * 1e18)),
            "timeStamp": "1700000000",
            "hash": "0xdeadbeef",
            "gasUsed": "21000",
        }
    ]

    with (
        patch("crypto_trace.graph_builder.EtherscanClient") as MockClient,
        patch("crypto_trace.graph_builder.scorer") as mock_scorer,
    ):
        # Set up mock client as async context manager
        mock_instance = AsyncMock()
        mock_instance.get_normal_txs = AsyncMock(return_value=mock_normal_txs)
        mock_instance.get_token_txs = AsyncMock(return_value=[])
        mock_instance.get_eth_balance = AsyncMock(return_value=1.5)
        mock_instance.is_supernode = MagicMock(return_value=False)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=None)

        # Set up mock scorer
        mock_scorer.compute_dirty_score = MagicMock(
            return_value={"dirtyScore": 0.8, "riskLevel": "HIGH"}
        )
        mock_scorer._pipeline = MagicMock()  # mark as loaded

        from crypto_trace.graph_builder import trace_wallet
        result = await trace_wallet(root_address=root, max_depth=1, max_nodes=10)

    # Verify output schema
    assert "nodes" in result
    assert "links" in result
    assert "meta" in result
    assert isinstance(result["nodes"], list)
    assert isinstance(result["links"], list)

    # Root node must be present
    node_ids = {n["id"] for n in result["nodes"]}
    assert root.lower() in node_ids

    # Nodes must have required fields
    for node in result["nodes"]:
        assert "dirtyScore" in node
        assert "riskLevel" in node
        assert "depth" in node
        assert "address" in node

    # Links must have required fields
    for link in result["links"]:
        assert "source" in link
        assert "target" in link
        assert "value" in link
