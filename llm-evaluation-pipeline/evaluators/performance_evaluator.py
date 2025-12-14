"""
Performance Evaluator (Latency & Costs)

Tracks execution time and estimates costs:
- Real-time latency measurement
- Token counting and cost estimation
- Performance benchmarking
"""

import time
import logging
from typing import Dict, Any, List
import tiktoken


class PerformanceEvaluator:
    """Evaluates latency and cost metrics."""
    
    # Cost per 1M tokens (as of Dec 2024)
    COST_PER_1M_TOKENS = {
        'gpt-4-turbo': {'input': 10.0, 'output': 30.0},
        'gpt-3.5-turbo': {'input': 0.5, 'output': 1.5},
        'claude-3-sonnet': {'input': 3.0, 'output': 15.0},
        'claude-3-haiku': {'input': 0.25, 'output': 1.25},
        'gemini-pro': {'input': 0.5, 'output': 1.5},
    }
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.logger = logging.getLogger("PerformanceEvaluator")
        
        # Initialize tokenizer for cost estimation
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            # Fallback to cl100k_base (used by GPT-4)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.logger.info("Performance evaluator initialized")
    
    def evaluate(
        self, 
        response: str, 
        start_time: float,
        context_texts: List[str] = None,
        model_name: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate performance metrics.
        
        Args:
            response: AI-generated response
            start_time: Evaluation start timestamp
            context_texts: Context used (for cost estimation)
            model_name: LLM model used
            metadata: Response metadata (may contain generation latency)
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # 1. Calculate evaluation latency
            eval_latency_ms = (time.time() - start_time) * 1000
            
            # 2. Extract generation latency if available
            generation_latency_ms = None
            if metadata:
                generation_latency_ms = (
                    metadata.get('latency_ms') or
                    metadata.get('generation_time_ms') or
                    metadata.get('response_time_ms')
                )
            
            # 2. Count tokens
            response_tokens = self._count_tokens(response)
            
            # Estimate context tokens if provided
            context_tokens = 0
            if context_texts:
                context_text = " ".join(context_texts)
                context_tokens = self._count_tokens(context_text)
            
            total_tokens = context_tokens + response_tokens
            
            # 3. Estimate costs (for AI response generation)
            model = model_name or self.config.DEFAULT_MODEL
            generation_cost = self._estimate_cost(
                input_tokens=context_tokens,
                output_tokens=response_tokens,
                model=model
            )
            
            # 4. Calculate throughput
            tokens_per_second = (
                response_tokens / (eval_latency_ms / 1000) 
                if eval_latency_ms > 0 else 0
            )
            
            return {
                # AI Response Performance (what they asked for)
                "generation_latency_ms": generation_latency_ms,
                "estimated_generation_cost_usd": round(generation_cost, 6),
                "response_tokens": response_tokens,
                "context_tokens": context_tokens,
                "total_tokens": total_tokens,
                
                # Evaluation Pipeline Performance (for monitoring)
                "evaluation_latency_ms": round(eval_latency_ms, 2),
                "tokens_per_second": round(tokens_per_second, 2),
                
                "model": model,
                "assessment": self._get_assessment(generation_latency_ms, generation_cost)
            }
            
        except Exception as e:
            self.logger.error(f"Performance evaluation failed: {str(e)}")
            return self._get_error_result()
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        if not text:
            return 0
        
        try:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            self.logger.warning(f"Token counting failed: {str(e)}")
            # Fallback: rough estimate (1 token ≈ 4 chars)
            return len(text) // 4
    
    def _estimate_cost(
        self, 
        input_tokens: int, 
        output_tokens: int, 
        model: str
    ) -> float:
        """
        Estimate API cost based on token usage.
        
        Args:
            input_tokens: Number of input tokens (context + query)
            output_tokens: Number of output tokens (response)
            model: Model identifier
            
        Returns:
            Estimated cost in USD
        """
        # Normalize model name
        model_lower = model.lower()
        
        # Find matching cost profile
        cost_profile = None
        for model_key, costs in self.COST_PER_1M_TOKENS.items():
            if model_key in model_lower:
                cost_profile = costs
                break
        
        # Default to GPT-3.5 costs if model unknown
        if not cost_profile:
            cost_profile = self.COST_PER_1M_TOKENS['gpt-3.5-turbo']
            self.logger.warning(
                f"Unknown model '{model}', using default costs"
            )
        
        # Calculate cost
        input_cost = (input_tokens / 1_000_000) * cost_profile['input']
        output_cost = (output_tokens / 1_000_000) * cost_profile['output']
        
        total_cost = input_cost + output_cost
        return total_cost
    
    def _get_assessment(
        self, 
        latency_ms: float = None, 
        cost_usd: float = None
    ) -> str:
        """Convert latency and cost to qualitative assessment."""
        assessments = []
        
        if latency_ms is not None:
            if latency_ms < 500:
                assessments.append("Fast response")
            elif latency_ms < 2000:
                assessments.append("Acceptable latency")
            elif latency_ms < 5000:
                assessments.append("Slow response")
            else:
                assessments.append("Very slow response")
        
        if cost_usd is not None:
            if cost_usd < 0.001:
                assessments.append("Low cost")
            elif cost_usd < 0.01:
                assessments.append("Moderate cost")
            else:
                assessments.append("High cost")
        
        return " - ".join(assessments) if assessments else "Unknown"
    
    def _get_error_result(self) -> Dict[str, Any]:
        """Return error result structure."""
        return {
            "generation_latency_ms": None,
            "estimated_generation_cost_usd": 0.0,
            "response_tokens": 0,
            "context_tokens": 0,
            "total_tokens": 0,
            "evaluation_latency_ms": 0.0,
            "tokens_per_second": 0.0,
            "model": "unknown",
            "assessment": "Error - Evaluation failed"
        }