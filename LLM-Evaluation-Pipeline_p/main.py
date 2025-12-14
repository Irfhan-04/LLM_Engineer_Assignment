#!/usr/bin/env python3
"""CLI interface for LLM Evaluation Pipeline."""
import argparse
import json
import sys
import logging

from src.evaluator import LLMEvaluator
from src.utils.json_loader import JSONLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLM Response Evaluation Pipeline")

    parser.add_argument("--conversation", type=str, help="Path to conversation JSON")
    parser.add_argument("--context", type=str, help="Path to context vectors JSON")
    parser.add_argument("--batch", type=str, help="Path to batch input JSON")
    parser.add_argument(
        "--output", type=str, default="results.json", help="Output file for results"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo", help="LLM model being evaluated"
    )
    parser.add_argument("--latency", type=float, help="Latency in milliseconds")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)

    print("\n" + "=" * 70)
    print("LLM Response Evaluation Pipeline")
    print("=" * 70 + "\n")

    try:
        # Initialize evaluator
        evaluator = LLMEvaluator(model_name=args.model, verbose=args.verbose)

        # Single evaluation
        if args.conversation and args.context:
            print(f"Loading conversation from: {args.conversation}")
            print(f"Loading context from: {args.context}\n")

            conv, context = JSONLoader.load_conversation_and_context(
                args.conversation, args.context
            )

            # Extract message and query (handle both sample formats)
            messages = conv["messages"]
            if len(messages) >= 2:
                user_query = messages[0]["content"]
                llm_response = messages[-1]["content"]
                message_id = messages[-1].get("message_id", "msg_final")
            else:
                # Fallback for single message format
                user_query = "Sample query"
                llm_response = messages[0]["content"]
                message_id = messages[0].get("message_id", "msg_final")

            latency = args.latency or 250.0

            result = evaluator.evaluate(
                conversation_id=conv["conversation_id"],
                message_id=message_id,
                user_query=user_query,
                llm_response=llm_response,
                context_vectors=context["vectors"],
                latency_ms=latency,
            )

            print("\nEvaluation Result:")
            print(json.dumps(result, indent=2))

            # Save result
            JSONLoader.save_json(result, args.output)
            print(f"\nResults saved to: {args.output}")

        # Batch evaluation
        elif args.batch:
            print(f"Loading batch from: {args.batch}\n")

            with open(args.batch, "r") as f:
                batch_data = json.load(f)

            results = evaluator.evaluate_batch(batch_data["evaluations"])

            output_data = {"total_evaluations": len(results), "results": results}

            JSONLoader.save_json(output_data, args.output)
            print(f"\nProcessed {len(results)} evaluations")
            print(f"Results saved to: {args.output}")

        # Show statistics
        stats = evaluator.get_statistics()
        print("\nEvaluator Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\n✓ Evaluation completed successfully\n")
        return 0

    except Exception as e:
        print(f"\n✗ Error: {str(e)}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
