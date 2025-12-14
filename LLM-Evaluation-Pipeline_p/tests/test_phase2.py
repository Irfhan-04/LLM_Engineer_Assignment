"""Phase 2 metric evaluators tests."""

import pytest
from src.metrics.relevance import RelevanceEvaluator
from src.metrics.hallucination import HallucinationEvaluator
from src.metrics.performance import PerformanceEvaluator


class TestRelevanceEvaluator:
    """Test relevance evaluation."""

    def test_empty_response(self):
        eval = RelevanceEvaluator()
        score, details = eval.evaluate("query", "", [])
        assert score == 0.0

    def test_high_relevance(self):
        eval = RelevanceEvaluator()
        query = "What is machine learning?"
        response = "Machine learning is artificial intelligence. " * 5
        score, _ = eval.evaluate(query, response, [])
        assert score > 0.5


class TestHallucinationEvaluator:
    """Test hallucination detection."""

    def test_balanced_response(self):
        eval = HallucinationEvaluator()
        response = "This might be true based on research."
        score, details = eval.evaluate(response, [])
        assert 0 <= score <= 1
        assert details["certainty"] == "appropriately_qualified"


class TestPerformanceEvaluator:
    """Test performance metrics."""

    def test_fast_response(self):
        eval = PerformanceEvaluator("gpt-3.5-turbo")
        score, details = eval.evaluate("test", 250)
        assert details["latency_score"] == 1.0

    def test_cost_calculation(self):
        eval = PerformanceEvaluator("gpt-3.5-turbo")
        _, details = eval.evaluate("test", 100)
        assert details["cost_usd"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
