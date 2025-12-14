"""Performance Metrics Evaluator (Latency & Costs)."""

from typing import Dict, Any, Optional, Tuple


class PerformanceEvaluator:
    """Evaluates performance: latency and costs."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.inference_times = []

        # Pricing models (per 1K tokens)
        self.costs = {
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "claude-2": {"input": 0.008, "output": 0.024},
            "default": {"input": 0.0005, "output": 0.0015},
        }

    def evaluate(
        self,
        response_text: str,
        latency_ms: float,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate performance metrics.

        Returns: (score 0-1, details dict)
        """
        if input_tokens is None:
            input_tokens = self._estimate_tokens(response_text, "input")

        if output_tokens is None:
            output_tokens = self._estimate_tokens(response_text, "output")

        # Calculate costs
        cost = self._calculate_cost(input_tokens, output_tokens)
        latency_score = self._calculate_latency_score(latency_ms)
        cost_score = self._calculate_cost_score(cost)
        efficiency = self._calculate_efficiency(input_tokens, output_tokens)

        # Weighted average
        overall = latency_score * 0.5 + cost_score * 0.4 + efficiency * 0.1

        self.inference_times.append(latency_ms)

        return round(min(1.0, max(0.0, overall)), 3), {
            "latency_ms": latency_ms,
            "latency_score": round(latency_score, 3),
            "cost_usd": round(cost, 6),
            "cost_score": round(cost_score, 3),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "efficiency": round(efficiency, 3),
            "cost_per_token": round(cost / (input_tokens + output_tokens), 8),
            "model": self.model_name,
        }

    def _estimate_tokens(self, text: str, token_type: str = "output") -> int:
        """Estimate token count (1 token ≈ 4 chars or 0.75 words)."""
        word_count = len(text.split())
        tokens = int(word_count / 0.75)

        if token_type == "input":
            tokens = int(tokens * 0.8)

        return max(1, tokens)

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost."""
        pricing = self.costs.get(self.model_name, self.costs["default"])

        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

        return input_cost + output_cost

    def _calculate_latency_score(self, latency_ms: float) -> float:
        """Score latency (lower is better)."""
        if latency_ms < 500:
            return 1.0
        elif latency_ms < 1000:
            return 0.9
        elif latency_ms < 2000:
            return 0.7
        elif latency_ms < 5000:
            return 0.5
        else:
            return 0.3

    def _calculate_cost_score(self, cost_usd: float) -> float:
        """Score cost efficiency."""
        if cost_usd < 0.0001:
            return 1.0
        elif cost_usd < 0.001:
            return 0.9
        elif cost_usd < 0.01:
            return 0.7
        elif cost_usd < 0.1:
            return 0.5
        else:
            return max(0.1, 1.0 - (cost_usd / 10))

    def _calculate_efficiency(self, input_tokens: int, output_tokens: int) -> float:
        """Score token efficiency."""
        if input_tokens == 0:
            return 0.5

        ratio = output_tokens / input_tokens

        if 1.5 <= ratio <= 5:
            return 1.0
        elif 0.5 <= ratio < 1.5:
            return 0.8
        elif 5 < ratio <= 10:
            return 0.7
        else:
            return 0.5
