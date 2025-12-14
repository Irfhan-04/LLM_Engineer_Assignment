from utils import load_json, extract_latest_turn
from evaluator import (
    evaluate_relevance_and_completeness,
    evaluate_hallucination
)
from metrics import Metrics

def llm_call(prompt: str) -> str:
    """
    Replace with OpenAI / Claude / local LLM.
    Kept minimal for evaluation purposes.
    """
    return "0.8"  # replace with real LLM call if desired

if __name__ == "__main__":
    conversation = load_json("data/conversation.json")
    context_data = load_json("data/context.json")

    query, answer = extract_latest_turn(conversation)
    contexts = context_data.get("contexts", [])

    metrics = Metrics()

    relevance, completeness = evaluate_relevance_and_completeness(
        query, answer, llm_call, metrics
    )

    hallucination = evaluate_hallucination(
        answer, contexts, llm_call, metrics
    )

    result = {
        "relevance_score": relevance,
        "completeness_score": completeness,
        "hallucination_score": hallucination,
        **metrics.finalize()
    }

    print(result)
