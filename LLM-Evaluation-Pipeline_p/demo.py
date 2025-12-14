#!/usr/bin/env python3
"""Demo script showing evaluator usage."""
import json
from src.evaluator import LLMEvaluator


def demo_single():
    """Demo single evaluation."""
    print("\n" + "=" * 70)
    print("DEMO 1: Single Response Evaluation")
    print("=" * 70 + "\n")

    evaluator = LLMEvaluator(model_name="gpt-3.5-turbo")

    query = "What are the benefits of machine learning?"
    response = """Machine learning provides numerous benefits across industries.
    First, it enables predictive analytics for data-driven decisions.
    Second, it automates complex tasks and improves efficiency.
    Third, it discovers patterns humans might miss.
    Fourth, it personalizes user experiences at scale.
    Finally, it reduces operational costs through automation.
    These advantages make ML invaluable for modern organizations."""

    context = [
        {"summary": "ML enables predictive analytics"},
        {"summary": "ML improves operational efficiency"},
    ]

    result = evaluator.evaluate(
        conversation_id="demo_001",
        message_id="demo_msg_001",
        user_query=query,
        llm_response=response,
        context_vectors=context,
        latency_ms=275,
    )

    print("Result:")
    print(json.dumps(result, indent=2))


def demo_batch():
    """Demo batch evaluation."""
    print("\n" + "=" * 70)
    print("DEMO 2: Batch Evaluation")
    print("=" * 70 + "\n")

    evaluator = LLMEvaluator(model_name="gpt-4")

    batch = [
        {
            "conversation_id": f"batch_{i:02d}",
            "message_id": f"msg_{i:02d}",
            "user_query": f"Question {i}?",
            "llm_response": f"Answer to question {i}. " * 5,
            "context_vectors": [],
            "latency_ms": 200 + (i * 10),
        }
        for i in range(5)
    ]

    results = evaluator.evaluate_batch(batch)

    print(f"Processed {len(results)} evaluations\n")

    for result in results:
        print(
            f"Conversation {result['conversation_id']}: "
            f"Overall={result['scores']['overall']}"
        )


def demo_stats():
    """Demo statistics."""
    print("\n" + "=" * 70)
    print("DEMO 3: Statistics")
    print("=" * 70 + "\n")

    evaluator = LLMEvaluator(enable_cache=True)

    for i in range(10):
        evaluator.evaluate(
            conversation_id=f"stat_{i}",
            message_id=f"msg_{i}",
            user_query="test",
            llm_response="response " * 10,
            context_vectors=[],
            latency_ms=100 + i,
        )

    stats = evaluator.get_statistics()
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    print("\n🚀 LLM EVALUATION PIPELINE - DEMO\n")

    demo_single()
    demo_batch()
    demo_stats()

    print("\n" + "=" * 70)
    print("✓ All demos completed!")
    print("=" * 70 + "\n")
