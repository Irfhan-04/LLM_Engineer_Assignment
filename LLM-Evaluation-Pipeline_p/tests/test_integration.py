"""Integration tests for complete pipeline."""

import pytest
import json
import tempfile
from pathlib import Path

from src.evaluator import LLMEvaluator
from src.utils.json_loader import JSONLoader


class TestEndToEnd:
    """End-to-end pipeline tests."""

    def test_complete_evaluation_flow(self):
        """Test complete evaluation pipeline."""
        evaluator = LLMEvaluator(enable_cache=True, model_name="gpt-3.5-turbo")

        query = "What is artificial intelligence?"
        response = "AI is the simulation of human intelligence by computers. " * 5
        context = [{"summary": "AI definition and applications"}]

        result = evaluator.evaluate(
            conversation_id="test_001",
            message_id="test_msg_001",
            user_query=query,
            llm_response=response,
            context_vectors=context,
            latency_ms=150,
        )

        # Verify structure
        assert "scores" in result
        assert "overall" in result["scores"]
        assert all(
            0 <= result["scores"][m]["score"] <= 1
            for m in ["relevance", "hallucination", "performance"]
        )

    def test_batch_evaluation(self):
        """Test batch processing."""
        evaluator = LLMEvaluator()

        batch = [
            {
                "conversation_id": f"conv_{i}",
                "message_id": f"msg_{i}",
                "user_query": f"Question {i}",
                "llm_response": f"Answer {i}" * 5,
                "context_vectors": [],
                "latency_ms": 100 + i,
            }
            for i in range(10)
        ]

        results = evaluator.evaluate_batch(batch)

        assert len(results) == 10
        assert all("overall" in r["scores"] for r in results)

    def test_caching_effectiveness(self):
        """Test cache hit rate."""
        evaluator = LLMEvaluator(enable_cache=True)

        for _ in range(5):
            evaluator.evaluate(
                conversation_id="cache_test",
                message_id="same_msg",
                user_query="test",
                llm_response="response",
                context_vectors=[],
                latency_ms=100,
            )

        stats = evaluator.get_statistics()
        assert stats["total_evaluations"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
