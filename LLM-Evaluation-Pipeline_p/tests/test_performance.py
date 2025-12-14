"""Performance benchmarking tests."""

import time
import pytest
from src.evaluator import LLMEvaluator


class TestPerformance:
    """Performance benchmarks."""

    def test_single_eval_latency(self):
        """Test single evaluation latency."""
        evaluator = LLMEvaluator(verbose=False)

        start = time.time()
        evaluator.evaluate(
            conversation_id="perf_1",
            message_id="msg_1",
            user_query="test",
            llm_response="response " * 20,
            context_vectors=[],
            latency_ms=100,
        )
        elapsed = (time.time() - start) * 1000

        # Should complete in <100ms
        assert elapsed < 100
        print(f"\nSingle eval latency: {elapsed:.2f}ms")

    def test_batch_throughput(self):
        """Test batch processing throughput."""
        evaluator = LLMEvaluator(verbose=False)

        batch = [
            {
                "conversation_id": f"perf_{i}",
                "message_id": f"msg_{i}",
                "user_query": "test",
                "llm_response": "response " * 10,
                "context_vectors": [],
                "latency_ms": 100,
            }
            for i in range(100)
        ]

        start = time.time()
        results = evaluator.evaluate_batch(batch)
        elapsed = time.time() - start

        throughput = len(results) / elapsed
        print(f"\nThroughput: {throughput:.0f} evals/sec")

        assert throughput > 100  # At least 100 evals/sec


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
