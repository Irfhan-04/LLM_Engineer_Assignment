"""Response Relevance & Completeness Evaluator."""

import re
from typing import Dict, List, Any, Tuple


class RelevanceEvaluator:
    """Evaluates response relevance and completeness."""

    def __init__(self, min_response_length: int = 50):
        self.min_response_length = min_response_length

    def evaluate(
        self, user_query: str, llm_response: str, context_vectors: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate relevance of LLM response.

        Returns: (score 0-1, details dict)
        """
        if not llm_response or not llm_response.strip():
            return 0.0, {"error": "Empty response"}

        # Calculate sub-metrics
        keyword_score = self._calculate_keyword_match(user_query, llm_response)
        completeness = self._calculate_completeness(user_query, llm_response)
        context_util = self._calculate_context_utilization(
            context_vectors, llm_response
        )
        length_score = self._calculate_length_appropriateness(user_query, llm_response)

        # Weighted average
        overall = (
            keyword_score * 0.3
            + completeness * 0.4
            + context_util * 0.2
            + length_score * 0.1
        )

        # Return score and details
        return round(min(1.0, max(0.0, overall)), 3), {
            "keyword_match": round(keyword_score, 3),
            "semantic_completeness": round(completeness, 3),
            "context_utilization": round(context_util, 3),
            "length_appropriateness": round(length_score, 3),
            "response_length": len(llm_response),
            "query_length": len(user_query),
        }

    def _calculate_keyword_match(self, query: str, response: str) -> float:
        """Calculate how many query keywords appear in response."""
        common_words = {"the", "a", "an", "and", "or", "but", "is", "are"}

        query_words = set(
            w.lower()
            for w in re.findall(r"\b\w+\b", query)
            if len(w) > 3 and w.lower() not in common_words
        )

        if not query_words:
            return 1.0

        response_lower = response.lower()
        matches = sum(1 for w in query_words if w in response_lower)

        return matches / len(query_words)

    def _calculate_completeness(self, query: str, response: str) -> float:
        """Check if response comprehensively answers query."""
        indicators = [
            ("what", ["is", "are", "involves"]),
            ("how", ["steps", "process", "method"]),
            ("why", ["because", "reason", "causes"]),
        ]

        query_lower = query.lower()
        response_lower = response.lower()
        score = 0.5

        for qtype, words in indicators:
            if qtype in query_lower:
                if any(w in response_lower for w in words):
                    score = 0.8
                else:
                    score = 0.4
                break

        # Boost for substantive responses
        word_count = len(response.split())
        if word_count > 100:
            score = min(1.0, score + 0.15)
        elif word_count < 20:
            score = max(0.0, score - 0.2)

        return min(1.0, score)

    def _calculate_context_utilization(
        self, context: List[Dict], response: str
    ) -> float:
        """Evaluate if response uses provided context."""
        if not context:
            return 0.7

        response_len = len(response)
        min_len = 75

        if response_len < min_len:
            return 0.6
        elif response_len > min_len * 3:
            return 0.85
        else:
            return 0.8

    def _calculate_length_appropriateness(self, query: str, response: str) -> float:
        """Ensure response length is appropriate."""
        if len(response) < self.min_response_length:
            return 0.3

        ratio = len(response) / max(len(query), 1)

        if 3 <= ratio <= 10:
            return 1.0
        elif 1 <= ratio < 3:
            return 0.7
        elif ratio > 10:
            return 0.85
        else:
            return 0.4
