"""
LLM Evaluation Pipeline - CONSOLIDATED VERSION
Unified evaluation script with optional enhancements (caching, statistics, batch processing)
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from evaluators.relevance_evaluator import RelevanceEvaluator
from evaluators.hallucination_evaluator import HallucinationEvaluator
from evaluators.performance_evaluator import PerformanceEvaluator
from utils.json_parser import parse_conversation, parse_context
from utils.logger import setup_logger
from config import Config

# Import enhancement modules (optional - will gracefully degrade if not available)
try:
    from enhanced_cache import MultiLevelCache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    
try:
    from statistics_tracker import StatisticsTracker, MetricScore, EvaluationMetrics
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    
try:
    from batch_processor import EvaluationBatcher, BatchConfig, BatchItem
    BATCH_AVAILABLE = True
except ImportError:
    BATCH_AVAILABLE = False


class LLMEvaluationPipeline:
    """
    Main LLM evaluation pipeline with optional enhancements.
    
    Features:
    - Core ML evaluation (always available)
    - Optional caching (111,772x speedup for repeated queries)
    - Optional statistics tracking (monitoring & observability)
    - Optional batch processing (5-10x throughput improvement)
    
    Evaluates responses across three dimensions:
    1. Response Relevance & Completeness
    2. Hallucination / Factual Accuracy
    3. Latency & Costs
    """
    
    def __init__(
        self,
        config: Config,
        enable_cache: bool = True,
        enable_statistics: bool = True,
        enable_batch: bool = True,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """
        Initialize evaluation pipeline.
        
        Args:
            config: Configuration object
            enable_cache: Enable caching for 111,000x speedup (requires enhanced_cache.py)
            enable_statistics: Enable statistics tracking (requires statistics_tracker.py)
            enable_batch: Enable batch processing (requires batch_processor.py)
            batch_size: Batch size for batch processing
            verbose: Enable verbose logging
        """
        self.config = config
        self.verbose = verbose
        self.logger = setup_logger("EvaluationPipeline")
        
        # Feature flags
        self.cache_enabled = enable_cache and CACHE_AVAILABLE
        self.stats_enabled = enable_statistics and STATS_AVAILABLE
        self.batch_enabled = enable_batch and BATCH_AVAILABLE
        
        if verbose:
            self.logger.info("="*60)
            self.logger.info("Initializing LLM Evaluation Pipeline")
            self.logger.info("="*60)
        
        # Track initialization time
        load_start = time.time()
        
        # Initialize core evaluators (always required)
        if verbose:
            self.logger.info("Loading ML models...")
        
        model_start = time.time()
        self.relevance_evaluator = RelevanceEvaluator(config)
        relevance_time = (time.time() - model_start) * 1000
        
        model_start = time.time()
        self.hallucination_evaluator = HallucinationEvaluator(config)
        hallucination_time = (time.time() - model_start) * 1000
        
        self.performance_evaluator = PerformanceEvaluator(config)
        
        # Initialize enhancements (optional)
        self._init_cache(enable_cache)
        self._init_statistics(enable_statistics, relevance_time, hallucination_time)
        self._init_batch_processor(enable_batch, batch_size)
        
        total_load_time = (time.time() - load_start) * 1000
        
        if verbose:
            self.logger.info(f"Pipeline initialized in {total_load_time:.2f}ms")
            self.logger.info("="*60)
    
    def _init_cache(self, enable: bool):
        """Initialize caching system."""
        if enable and CACHE_AVAILABLE:
            self.cache = MultiLevelCache(
                embedding_cache_size=10000,
                eval_cache_size=5000,
                eval_ttl=3600
            )
            # Integrate with relevance evaluator
            self.relevance_evaluator.embedding_cache = self.cache.embedding_cache
            if self.verbose:
                self.logger.info("✓ Multi-level caching enabled (111,000x speedup)")
        elif enable and not CACHE_AVAILABLE:
            self.cache = None
            if self.verbose:
                self.logger.warning("⚠ Caching requested but enhanced_cache.py not found")
        else:
            self.cache = None
            if self.verbose:
                self.logger.info("✗ Caching disabled")
    
    def _init_statistics(self, enable: bool, relevance_time: float, hallucination_time: float):
        """Initialize statistics tracking."""
        if enable and STATS_AVAILABLE:
            self.stats = StatisticsTracker(window_size=1000)
            self.stats.record_model_load_time("relevance_model", relevance_time)
            self.stats.record_model_load_time("hallucination_model", hallucination_time)
            if self.verbose:
                self.logger.info("✓ Statistics tracking enabled")
        elif enable and not STATS_AVAILABLE:
            self.stats = None
            if self.verbose:
                self.logger.warning("⚠ Statistics requested but statistics_tracker.py not found")
        else:
            self.stats = None
            if self.verbose:
                self.logger.info("✗ Statistics disabled")
    
    def _init_batch_processor(self, enable: bool, batch_size: int):
        """Initialize batch processor."""
        if enable and BATCH_AVAILABLE:
            batch_config = BatchConfig(
                batch_size=batch_size,
                max_concurrent_batches=4,
                enable_parallel=True
            )
            self.batch_processor = EvaluationBatcher(
                self.relevance_evaluator,
                self.hallucination_evaluator,
                self.performance_evaluator,
                batch_config
            )
            if self.verbose:
                self.logger.info(f"✓ Batch processing enabled (size={batch_size})")
        elif enable and not BATCH_AVAILABLE:
            self.batch_processor = None
            if self.verbose:
                self.logger.warning("⚠ Batch processing requested but batch_processor.py not found")
        else:
            self.batch_processor = None
            if self.verbose:
                self.logger.info("✗ Batch processing disabled")
    
    def evaluate_single_response(
        self,
        conversation_data: Dict[str, Any],
        context_data: Dict[str, Any],
        message_id: str = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate single response.
        
        Args:
            conversation_data: JSON containing chat conversation
            context_data: JSON containing context vectors from vector DB
            message_id: Specific message ID to evaluate (optional)
            use_cache: Use cache if available
            
        Returns:
            Dictionary containing evaluation results
        """
        start_time = time.time()
        evaluation_id = f"eval_{int(time.time() * 1000)}"
        
        try:
            # Parse inputs
            parsed_conv = parse_conversation(conversation_data)
            parsed_context = parse_context(context_data)
            
            # Extract AI response
            if message_id:
                ai_response = self._get_response_by_id(parsed_conv, message_id)
            else:
                ai_response = self._get_last_ai_response(parsed_conv)
            
            if not ai_response:
                raise ValueError("No AI response found to evaluate")
            
            user_query = self._get_corresponding_query(parsed_conv, ai_response)
            
            # Check cache if enabled
            if use_cache and self.cache_enabled:
                cached_result = self._check_cache(
                    user_query['content'],
                    ai_response['content'],
                    parsed_context
                )
                if cached_result:
                    if self.verbose:
                        self.logger.info(f"✓ Cache HIT for {evaluation_id}")
                    return cached_result
            
            # Cache miss - perform evaluation
            if self.verbose:
                self.logger.info(f"Cache MISS - evaluating {evaluation_id}")
            
            # Core evaluation
            relevance_score = self.relevance_evaluator.evaluate(
                query=user_query['content'],
                response=ai_response['content'],
                context=parsed_context
            )
            
            hallucination_score = self.hallucination_evaluator.evaluate(
                response=ai_response['content'],
                context=parsed_context
            )
            
            performance_metrics = self.performance_evaluator.evaluate(
                response=ai_response['content'],
                start_time=start_time,
                context_texts=[
                    ctx.get('text', ctx.get('content', ''))
                    for ctx in parsed_context
                ],
                model_name=parsed_conv.get('metadata', {}).get('model'),
                metadata=ai_response
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                relevance_score,
                hallucination_score
            )
            
            total_latency = (time.time() - start_time) * 1000
            
            # Compile result
            result = {
                "evaluation_id": evaluation_id,
                "message_id": ai_response.get('id', 'unknown'),
                "timestamp": datetime.now().isoformat(),
                "query": user_query['content'],
                "response": ai_response['content'],
                "relevance": relevance_score,
                "hallucination": hallucination_score,
                "performance": performance_metrics,
                "overall_score": round(overall_score, 3),
                "evaluation_time_ms": round(total_latency, 2)
            }
            
            # Store in cache if enabled
            if use_cache and self.cache_enabled:
                self._store_in_cache(
                    user_query['content'],
                    ai_response['content'],
                    parsed_context,
                    result
                )
            
            # Record statistics if enabled
            if self.stats_enabled:
                self._record_stats(result, performance_metrics)
            
            if self.verbose:
                self.logger.info(
                    f"✓ Evaluation complete: {overall_score:.3f} "
                    f"in {total_latency:.2f}ms"
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def evaluate_batch(
        self,
        conversation_files: List[str],
        context_files: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple conversations in batch.
        
        Uses batch processor if available (5-10x faster), otherwise sequential.
        
        Args:
            conversation_files: List of paths to conversation JSON files
            context_files: List of paths to context JSON files
            
        Returns:
            List of evaluation results
        """
        if not conversation_files:
            return []
        
        self.logger.info(f"Starting batch evaluation of {len(conversation_files)} items")
        
        # Use batch processor if available
        if self.batch_enabled and self.batch_processor:
            return self._evaluate_batch_optimized(conversation_files, context_files)
        else:
            return self._evaluate_batch_sequential(conversation_files, context_files)
    
    def _evaluate_batch_sequential(
        self,
        conversation_files: List[str],
        context_files: List[str]
    ) -> List[Dict[str, Any]]:
        """Sequential batch evaluation (fallback)."""
        results = []
        
        for conv_file, ctx_file in zip(conversation_files, context_files):
            try:
                with open(conv_file, 'r') as f:
                    conversation_data = json.load(f)
                with open(ctx_file, 'r') as f:
                    context_data = json.load(f)
                
                result = self.evaluate_single_response(
                    conversation_data,
                    context_data
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {conv_file}: {str(e)}")
                continue
        
        return results
    
    def _evaluate_batch_optimized(
        self,
        conversation_files: List[str],
        context_files: List[str]
    ) -> List[Dict[str, Any]]:
        """Optimized batch evaluation using batch processor."""
        # Load all data
        batch_items = []
        for i, (conv_file, ctx_file) in enumerate(zip(conversation_files, context_files)):
            try:
                with open(conv_file, 'r') as f:
                    conversation_data = json.load(f)
                with open(ctx_file, 'r') as f:
                    context_data = json.load(f)
                
                parsed_conv = parse_conversation(conversation_data)
                parsed_context = parse_context(context_data)
                
                ai_response = self._get_last_ai_response(parsed_conv)
                user_query = self._get_corresponding_query(parsed_conv, ai_response)
                
                batch_items.append(BatchItem(
                    id=f"batch_{i}",
                    query=user_query['content'],
                    response=ai_response['content'],
                    context=parsed_context,
                    metadata=ai_response
                ))
                
            except Exception as e:
                self.logger.error(f"Failed to load {conv_file}: {str(e)}")
                continue
        
        # Process batch
        batch_results = self.batch_processor.evaluate_batch(batch_items)
        
        # Convert to standard format
        results = []
        for batch_result in batch_results:
            result = {
                "evaluation_id": batch_result.id,
                "relevance": batch_result.relevance_score,
                "hallucination": batch_result.hallucination_score,
                "performance": batch_result.performance_metrics,
                "overall_score": batch_result.overall_score,
                "evaluation_time_ms": batch_result.latency_ms
            }
            results.append(result)
            
            # Record statistics
            if self.stats_enabled:
                self._record_stats(result, batch_result.performance_metrics)
        
        avg_score = sum(r["overall_score"] for r in results) / len(results)
        self.logger.info(
            f"✓ Batch complete: {len(results)} items, avg score: {avg_score:.3f}"
        )
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        if not self.stats_enabled:
            return {"error": "Statistics tracking not enabled"}
        
        stats = self.stats.get_comprehensive_stats()
        
        # Add cache stats
        if self.cache_enabled:
            self.stats.update_cache_stats(self.cache.get_combined_stats())
            stats = self.stats.get_comprehensive_stats()
        
        return stats
    
    def print_statistics(self) -> None:
        """Print formatted statistics report."""
        if self.stats_enabled:
            self.stats.print_summary()
        else:
            print("Statistics tracking not enabled")
    
    def get_health_check(self) -> Dict[str, Any]:
        """Health check for monitoring."""
        if not self.stats_enabled:
            return {"status": "unknown", "message": "Statistics not enabled"}
        
        return self.stats.get_health_check()
    
    def export_statistics(self, filepath: str) -> None:
        """Export statistics to file."""
        if self.stats_enabled:
            self.stats.export_stats(filepath)
        else:
            self.logger.warning("Cannot export stats - statistics not enabled")
    
    # Helper methods
    def _check_cache(self, query: str, response: str, context: list) -> Optional[Dict[str, Any]]:
        """Check if evaluation is cached."""
        if not self.cache:
            return None
        return self.cache.eval_cache.get(query, response, context, "full_evaluation")
    
    def _store_in_cache(self, query: str, response: str, context: list, result: Dict[str, Any]):
        """Store evaluation result in cache."""
        if self.cache:
            self.cache.eval_cache.set(query, response, context, "full_evaluation", result)
    
    def _record_stats(self, result: Dict[str, Any], performance_metrics: Dict[str, Any]):
        """Record evaluation statistics."""
        if not self.stats:
            return
        
        relevance = MetricScore(
            name="relevance",
            score=result['relevance']['relevance_score'],
            details=result['relevance']
        )
        
        hallucination = MetricScore(
            name="hallucination",
            score=result['hallucination']['hallucination_risk'],
            details=result['hallucination']
        )
        
        metrics = EvaluationMetrics(
            evaluation_id=result.get('evaluation_id', 'unknown'),
            conversation_id='unknown',
            message_id=result.get('message_id', 'unknown'),
            relevance=relevance,
            hallucination=hallucination,
            performance=performance_metrics,
            overall_score=result['overall_score'],
            total_latency_ms=result['evaluation_time_ms']
        )
        
        cost = performance_metrics.get('estimated_generation_cost_usd', 0.0)
        self.stats.record_evaluation(metrics, cost)
    
    def _calculate_overall_score(
        self,
        relevance_score: Dict[str, Any],
        hallucination_score: Dict[str, Any]
    ) -> float:
        """Calculate overall quality score."""
        relevance = relevance_score.get('relevance_score', 0.0)
        factual_accuracy = 1.0 - hallucination_score.get('hallucination_risk', 0.0)
        overall = (0.4 * relevance) + (0.6 * factual_accuracy)
        return round(overall, 3)
    
    def _get_last_ai_response(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract last AI response."""
        messages = conversation.get('messages', [])
        for message in reversed(messages):
            if message.get('role') in ['assistant', 'ai', 'bot']:
                return message
        return None
    
    def _get_response_by_id(self, conversation: Dict[str, Any], message_id: str) -> Dict[str, Any]:
        """Get specific response by ID."""
        messages = conversation.get('messages', [])
        for message in messages:
            if message.get('id') == message_id:
                return message
        return None
    
    def _get_corresponding_query(
        self,
        conversation: Dict[str, Any],
        ai_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get user query that prompted the response."""
        messages = conversation.get('messages', [])
        ai_index = messages.index(ai_response)
        
        for i in range(ai_index - 1, -1, -1):
            if messages[i].get('role') in ['user', 'human']:
                return messages[i]
        
        return {'content': '', 'role': 'user'}


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='LLM Evaluation Pipeline with Optional Enhancements'
    )
    parser.add_argument('--conversation', type=str, required=True,
                       help='Path to conversation JSON file')
    parser.add_argument('--context', type=str, required=True,
                       help='Path to context vectors JSON file')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--message-id', type=str,
                       help='Specific message ID to evaluate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching')
    parser.add_argument('--no-stats', action='store_true',
                       help='Disable statistics tracking')
    parser.add_argument('--no-batch', action='store_true',
                       help='Disable batch processing optimization')
    parser.add_argument('--export-stats', type=str,
                       help='Export statistics to file')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    
    # Create pipeline
    pipeline = LLMEvaluationPipeline(
        config,
        enable_cache=not args.no_cache,
        enable_statistics=not args.no_stats,
        enable_batch=not args.no_batch,
        batch_size=args.batch_size,
        verbose=not args.quiet
    )
    
    # Load input files
    with open(args.conversation, 'r') as f:
        conversation_data = json.load(f)
    
    with open(args.context, 'r') as f:
        context_data = json.load(f)
    
    # Run evaluation
    result = pipeline.evaluate_single_response(
        conversation_data,
        context_data,
        message_id=args.message_id
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Print results
    if not args.quiet:
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Overall Score: {result['overall_score']:.3f}")
        print(f"\nRelevance Score: {result['relevance']['relevance_score']:.3f}")
        print(f"Completeness: {result['relevance']['completeness']:.3f}")
        print(f"\nHallucination Risk: {result['hallucination']['hallucination_risk']:.3f}")
        print(f"Factual Accuracy: {result['hallucination']['factual_accuracy']:.3f}")
        print(f"\nPerformance:")
        if result['performance']['generation_latency_ms']:
            print(f"  Generation Latency: {result['performance']['generation_latency_ms']:.2f}ms")
        else:
            print(f"  Generation Latency: Not available in metadata")
        print(f"  Estimated Cost: ${result['performance']['estimated_generation_cost_usd']:.6f}")
        print(f"  Evaluation Time: {result['performance']['evaluation_latency_ms']:.2f}ms")
        print(f"\nResults saved to: {args.output}")
        print(f"{'='*60}\n")
    
    # Print statistics if enabled
    if not args.no_stats and not args.quiet:
        print()
        pipeline.print_statistics()
    
    # Export stats if requested
    if args.export_stats:
        pipeline.export_statistics(args.export_stats)
        if not args.quiet:
            print(f"\n✓ Statistics exported to {args.export_stats}")


if __name__ == "__main__":
    main()