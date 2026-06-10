"""
tests/test_model.py
────────────────────
Unit tests for model.py — risk scoring, label thresholds, schema alignment.

These tests run WITHOUT a trained model file by mocking the pipeline.
Integration tests (requiring model.pkl) are marked @pytest.mark.integration.

Run:
    pytest tests/test_model.py -v
    pytest tests/test_model.py -v -m integration   # needs model.pkl
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crypto_trace.model import DirtyScorer, risk_label
from crypto_trace.config import DIRTY_SCORE_HIGH, DIRTY_SCORE_MED


# ─── risk_label ───────────────────────────────────────────────────────────────

class TestRiskLabel:
    def test_high_risk(self):
        assert risk_label(DIRTY_SCORE_HIGH) == "HIGH"
        assert risk_label(1.0) == "HIGH"
        assert risk_label(0.99) == "HIGH"

    def test_medium_risk(self):
        assert risk_label(DIRTY_SCORE_MED) == "MEDIUM"
        assert risk_label(0.5) == "MEDIUM"
        assert risk_label(DIRTY_SCORE_HIGH - 0.001) == "MEDIUM"

    def test_low_risk(self):
        assert risk_label(0.0) == "LOW"
        assert risk_label(0.1) == "LOW"
        assert risk_label(DIRTY_SCORE_MED - 0.001) == "LOW"


# ─── DirtyScorer ─────────────────────────────────────────────────────────────

MOCK_FEATURES = [
    "Avg min between sent tnx",
    "Avg min between received tnx",
    "Time Diff between first and last (Mins)",
    "Sent tnx",
    "Received Tnx",
    "total Ether sent",
    "total ether received",
    "total ether balance",
    "avg val sent",
    "avg val received",
    "Unique Sent To Addresses",
    "Unique Received From Addresses",
    " ERC20 total Ether received",
    " ERC20 total ether sent",
    " Total ERC20 tnxs",
]


def _make_scorer(fraud_prob: float = 0.8) -> DirtyScorer:
    """Build a DirtyScorer with a mocked pipeline."""
    scorer = DirtyScorer()
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array([[1 - fraud_prob, fraud_prob]])
    scorer._pipeline = mock_pipeline
    scorer._features = MOCK_FEATURES
    return scorer


class TestDirtyScorerComputeScore:
    def test_returns_expected_keys(self):
        scorer = _make_scorer(0.8)
        result = scorer.compute_dirty_score({f: 0.0 for f in MOCK_FEATURES})
        assert "dirtyScore" in result
        assert "riskLevel" in result

    def test_score_range(self):
        for prob in [0.0, 0.3, 0.5, 0.7, 1.0]:
            scorer = _make_scorer(prob)
            result = scorer.compute_dirty_score({f: 0.0 for f in MOCK_FEATURES})
            assert 0.0 <= result["dirtyScore"] <= 1.0

    def test_high_risk_label(self):
        scorer = _make_scorer(0.9)
        result = scorer.compute_dirty_score({f: 0.0 for f in MOCK_FEATURES})
        assert result["riskLevel"] == "HIGH"

    def test_low_risk_label(self):
        scorer = _make_scorer(0.1)
        result = scorer.compute_dirty_score({f: 0.0 for f in MOCK_FEATURES})
        assert result["riskLevel"] == "LOW"

    def test_missing_features_default_to_zero(self):
        """Schema alignment: missing keys should default to 0, not raise."""
        scorer = _make_scorer(0.5)
        # Pass only a subset of features — rest should default to 0.0
        partial = {"Sent tnx": 100.0, "total Ether sent": 50.0}
        result = scorer.compute_dirty_score(partial)
        assert "dirtyScore" in result  # should not raise

    def test_pipeline_called_with_correct_shape(self):
        scorer = _make_scorer(0.7)
        scorer.compute_dirty_score({f: 1.0 for f in MOCK_FEATURES})
        call_args = scorer._pipeline.predict_proba.call_args
        X_arg = call_args[0][0]
        assert X_arg.shape == (1, len(MOCK_FEATURES))

    def test_raises_if_not_loaded(self):
        scorer = DirtyScorer()
        # _pipeline is None → should raise RuntimeError
        with pytest.raises((RuntimeError, AttributeError)):
            scorer.compute_dirty_score({})


# ─── Integration (requires model.pkl on disk) ─────────────────────────────────

@pytest.mark.integration
class TestDirtyScorerIntegration:
    def test_load_from_disk(self):
        scorer = DirtyScorer()
        scorer.load()
        assert scorer._pipeline is not None
        assert len(scorer.features) == 15

    def test_score_real_feature_vector(self):
        scorer = DirtyScorer()
        scorer.load()
        # All-zero vector should return a valid score (not crash)
        features = {f: 0.0 for f in scorer.features}
        result = scorer.compute_dirty_score(features)
        assert 0.0 <= result["dirtyScore"] <= 1.0
        assert result["riskLevel"] in ("LOW", "MEDIUM", "HIGH")
