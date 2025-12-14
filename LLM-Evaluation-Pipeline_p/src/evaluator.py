"""Main LLM Evaluation Pipeline Orchestrator."""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.metrics.relevance import RelevanceEvaluator
from src.metrics.hallucination import HallucinationEvaluator
from src.metrics.performance import PerformanceEvaluator
from src.utils.cache import EvaluationCache
from src.utils.logger import setup_logger


class LLMEvaluator:
    """Main orchestrator for LLM evaluation pipeline."""

    def __init__(
        self,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
        model_name: str = "gpt-3.5-turbo",
        verbose: bool = True,
    ):
        """
        Initialize evaluator.

        Args:
            enable_cache: Enable caching of results
            cache_ttl: Cache time-to-live in seconds
            model_name: Name of LLM model
            verbose: Enable verbose logging
        """
        self.enable_cache = enable_cache
        self.model_name = model_name
        self.verbose = verbose

        # Initialize components
        self.cache = EvaluationCache(default_ttl=cache_ttl) if enable_cache else None
        self.relevance_eval = RelevanceEvaluator()
        self.hallucination_eval = HallucinationEvaluator()
        self.performance_eval = PerformanceEvaluator(model_name)

        # Statistics
        self.evaluation_count = 0
        self.logger = setup_logger("LLMEvaluator")

    def evaluate(
        self,
        conversation_id: str,
        message_id: str,
        user_query: str,
        llm_response: str,
        context_vectors: List[Dict[str, Any]],
        latency_ms: float,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate LLM response in real-time.

        Args:
            conversation_id: Unique conversation ID
            message_id: Unique message ID
            user_query: Original user query
            llm_response: Generated response
            context_vectors: Context vectors from DB
            latency_ms: Response generation latency
            input_tokens: Input token count (estimated if None)
            output_tokens: Output token count (estimated if None)

        Returns:
            Complete evaluation result dictionary
        """
        self.evaluation_count += 1
        eval_start = time.time()

        try:
            # Evaluate relevance
            relevance_score, relevance_details = self.relevance_eval.evaluate(
                user_query, llm_response, context_vectors
            )

            # Evaluate hallucination
            hallucination_score, hallucination_details = (
                self.hallucination_eval.evaluate(llm_response, context_vectors)
            )

            # Evaluate performance
            performance_score, performance_details = self.performance_eval.evaluate(
                llm_response, latency_ms, input_tokens, output_tokens
            )

            # Calculate overall score
            overall_score = round(
                relevance_score * 0.4
                + hallucination_score * 0.35
                + performance_score * 0.25,
                3,
            )

            eval_time = (time.time() - eval_start) * 1000

            # Build result
            result = {
                "conversation_id": conversation_id,
                "message_id": message_id,
                "scores": {
                    "relevance": {
                        "score": relevance_score,
                        "details": relevance_details,
                    },
                    "hallucination": {
                        "score": hallucination_score,
                        "details": hallucination_details,
                    },
                    "performance": {
                        "score": performance_score,
                        "details": performance_details,
                    },
                    "overall": overall_score,
                },
                "timestamp": datetime.utcnow().isoformat(),
                "evaluation_latency_ms": round(eval_time, 2),
            }

            if self.verbose:
                self.logger.info(
                    f"Eval #{self.evaluation_count}: Overall={overall_score:.3f}, "
                    f"Relevance={relevance_score:.3f}, "
                    f"Hallucination={hallucination_score:.3f}, "
                    f"Performance={performance_score:.3f}"
                )

            return result

        except Exception as e:
            self.logger.error(f"Evaluation error: {str(e)}")
            raise

    def evaluate_batch(self, evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple responses in batch.

        Args:
            evaluations: List of evaluation request dictionaries

        Returns:
            List of evaluation results
        """
        self.logger.info(f"Starting batch evaluation of {len(evaluations)} items")

        results = []
        for eval_req in evaluations:
            result = self.evaluate(**eval_req)
            results.append(result)

        avg_score = sum(r["scores"]["overall"] for r in results) / len(results)
        self.logger.info(f"Batch complete. Average score: {avg_score:.3f}")

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluator statistics."""
        stats = {
            "total_evaluations": self.evaluation_count,
            "model": self.model_name,
            "cache_enabled": self.enable_cache,
        }

        if self.cache:
            cache_stats = self.cache.get_stats()
            stats.update(
                {
                    "cache_hits": cache_stats["hits"],
                    "cache_misses": cache_stats["misses"],
                    "cache_hit_rate_percent": round(cache_stats["hit_rate_percent"], 2),
                }
            )

        if self.performance_eval.inference_times:
            times = self.performance_eval.inference_times
            stats["avg_latency_ms"] = round(sum(times) / len(times), 2)

        return stats
