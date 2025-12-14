"""Hallucination / Factual Accuracy Evaluator."""

import re
from typing import Dict, List, Any, Tuple


class HallucinationEvaluator:
    """Evaluates hallucination and factual accuracy."""

    def evaluate(
        self, llm_response: str, context_vectors: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate factual accuracy and hallucination likelihood.

        Returns: (score 0-1, details dict)
        """
        if not llm_response:
            return 1.0, {"error": "Empty response"}

        # Calculate sub-metrics
        consistency = self._calculate_context_consistency(llm_response, context_vectors)
        claim_support = self._calculate_claim_support(llm_response)
        self_consistency = self._calculate_self_consistency(llm_response)
        grounding = self._calculate_grounding(llm_response, context_vectors)

        # Weighted average
        overall = (
            consistency * 0.35
            + claim_support * 0.35
            + self_consistency * 0.20
            + grounding * 0.10
        )

        return round(min(1.0, max(0.0, overall)), 3), {
            "context_consistency": round(consistency, 3),
            "claim_support": round(claim_support, 3),
            "self_consistency": round(self_consistency, 3),
            "grounding": round(grounding, 3),
            "total_claims": self._count_claims(llm_response),
            "certainty": self._assess_certainty(llm_response),
        }

    def _calculate_context_consistency(
        self, response: str, context: List[Dict]
    ) -> float:
        """Check consistency with provided context."""
        if not context:
            return 0.7

        response_lower = response.lower()
        mention_count = 0

        for vector in context:
            if isinstance(vector, dict) and "summary" in vector:
                keywords = set(w.lower() for w in vector["summary"].split())
                mention_count += sum(1 for k in keywords if k in response_lower)

        return 0.85 if mention_count > 0 else 0.6

    def _calculate_claim_support(self, response: str) -> float:
        """Evaluate percentage of claims with proper support."""
        sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]

        if not sentences:
            return 0.7

        qualifiers = [
            r"\b(may|might|could|possibly|likely)\b",
            r"\b(based on|according to)\b",
            r"\b(research|studies|evidence)\b",
        ]

        supported = 0
        for sent in sentences:
            sent_lower = sent.lower()
            has_qual = any(re.search(p, sent_lower) for p in qualifiers)
            has_attrib = any(
                w in sent_lower for w in ["according", "research", "studies"]
            )
            is_short = len(sent.split()) < 5

            if has_qual or has_attrib or is_short:
                supported += 1

        return min(1.0, (supported / len(sentences)) + 0.3)

    def _calculate_self_consistency(self, response: str) -> float:
        """Check for internal contradictions."""
        response_lower = response.lower()
        contradictions = [
            (r"\b(yes|true)\b", r"\b(no|false)\b"),
            (r"\b(always)\b", r"\b(never)\b"),
        ]

        contradiction_count = 0
        for p1, p2 in contradictions:
            if re.search(p1, response_lower) and re.search(p2, response_lower):
                contradiction_count += 1

        return max(0.5, 1.0 - (contradiction_count * 0.15))

    def _calculate_grounding(self, response: str, context: List[Dict]) -> float:
        """Evaluate grounding in provided context."""
        if not context:
            return 0.7

        response_len = len(response.split())
        context_len = sum(len(str(v).split()) for v in context)

        if context_len == 0:
            return 0.6

        ratio = response_len / max(context_len, 1)

        if 0.1 <= ratio <= 0.5:
            return 0.9
        elif 0.05 <= ratio <= 1.0:
            return 0.7
        else:
            return 0.5

    def _count_claims(self, response: str) -> int:
        """Count approximate claims (sentences)."""
        return len([s for s in re.split(r"[.!?]+", response) if s.strip()])

    def _assess_certainty(self, response: str) -> str:
        """Assess certainty level."""
        response_lower = response.lower()

        certain = sum(
            len(re.findall(p, response_lower))
            for p in [r"\b(definitely|certainly|absolutely)\b", r"\b(always|never)\b"]
        )

        qualified = sum(
            len(re.findall(p, response_lower))
            for p in [r"\b(may|might|could|possibly)\b", r"\b(based on|according to)\b"]
        )

        if certain > qualified:
            return "high_certainty"
        elif qualified > certain:
            return "appropriately_qualified"
        else:
            return "balanced"
