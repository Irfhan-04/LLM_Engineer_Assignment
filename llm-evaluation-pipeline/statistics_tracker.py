"""
Statistics Tracking & Monitoring System
Production-grade observability for the evaluation pipeline

Priority 2: Statistics Tracking - Monitor performance, costs, quality
"""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import deque
import json
import logging

logger = logging.getLogger("Statistics")


@dataclass
class MetricScore:
    """Standardized metric score with metadata."""
    name: str
    score: float  # 0-1 scale
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    latency_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "score": self.score,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms
        }
    
    def __post_init__(self):
        """Validate score is in valid range."""
        if not 0 <= self.score <= 1:
            logger.warning(
                f"Score {self.score} for {self.name} outside [0,1] range"
            )


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics for a single response."""
    evaluation_id: str
    conversation_id: str
    message_id: str
    
    # Core metrics
    relevance: MetricScore
    hallucination: MetricScore
    performance: Dict[str, Any]
    
    # Overall
    overall_score: float
    
    # Timing
    total_latency_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evaluation_id": self.evaluation_id,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "relevance": self.relevance.to_dict(),
            "hallucination": self.hallucination.to_dict(),
            "performance": self.performance,
            "overall_score": self.overall_score,
            "total_latency_ms": self.total_latency_ms,
            "timestamp": self.timestamp.isoformat()
        }


class PerformanceTracker:
    """Tracks performance metrics over time."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.scores = deque(maxlen=window_size)
        self.costs = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
    
    def record(
        self,
        latency_ms: float,
        overall_score: float,
        cost_usd: float
    ) -> None:
        """Record a single evaluation's metrics."""
        self.latencies.append(latency_ms)
        self.scores.append(overall_score)
        self.costs.append(cost_usd)
        self.timestamps.append(datetime.utcnow())
    
    def get_stats(self) -> Dict[str, Any]:
        """Calculate statistics over the window."""
        if not self.latencies:
            return self._empty_stats()
        
        latencies_list = list(self.latencies)
        scores_list = list(self.scores)
        costs_list = list(self.costs)
        
        # Latency stats
        latencies_sorted = sorted(latencies_list)
        n = len(latencies_sorted)
        
        # Percentiles
        p50_idx = int(n * 0.50)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)
        
        # Throughput (evaluations per second)
        if len(self.timestamps) >= 2:
            time_span = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
            throughput = len(self.timestamps) / time_span if time_span > 0 else 0
        else:
            throughput = 0
        
        return {
            "window_size": len(self.latencies),
            "latency": {
                "mean_ms": round(sum(latencies_list) / n, 2),
                "min_ms": round(min(latencies_list), 2),
                "max_ms": round(max(latencies_list), 2),
                "p50_ms": round(latencies_sorted[p50_idx], 2),
                "p95_ms": round(latencies_sorted[p95_idx], 2),
                "p99_ms": round(latencies_sorted[p99_idx], 2)
            },
            "quality": {
                "mean_score": round(sum(scores_list) / n, 3),
                "min_score": round(min(scores_list), 3),
                "max_score": round(max(scores_list), 3)
            },
            "cost": {
                "total_usd": round(sum(costs_list), 6),
                "mean_usd": round(sum(costs_list) / n, 6),
                "estimated_daily_usd": round(sum(costs_list) / n * 86400, 2)
            },
            "throughput": {
                "evaluations_per_second": round(throughput, 2),
                "estimated_daily_capacity": int(throughput * 86400)
            }
        }
    
    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty stats structure."""
        return {
            "window_size": 0,
            "latency": {"mean_ms": 0, "min_ms": 0, "max_ms": 0, 
                       "p50_ms": 0, "p95_ms": 0, "p99_ms": 0},
            "quality": {"mean_score": 0, "min_score": 0, "max_score": 0},
            "cost": {"total_usd": 0, "mean_usd": 0, "estimated_daily_usd": 0},
            "throughput": {"evaluations_per_second": 0, "estimated_daily_capacity": 0}
        }


class MetricAggregator:
    """Aggregates metrics by type (relevance, hallucination, etc.)."""
    
    def __init__(self):
        self.relevance_scores = []
        self.hallucination_scores = []
        self.overall_scores = []
        self.evaluation_count = 0
    
    def add_evaluation(
        self,
        relevance_score: float,
        hallucination_score: float,
        overall_score: float
    ) -> None:
        """Add evaluation result."""
        self.relevance_scores.append(relevance_score)
        self.hallucination_scores.append(hallucination_score)
        self.overall_scores.append(overall_score)
        self.evaluation_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated summary statistics."""
        if not self.relevance_scores:
            return self._empty_summary()
        
        return {
            "total_evaluations": self.evaluation_count,
            "relevance": {
                "mean": round(sum(self.relevance_scores) / len(self.relevance_scores), 3),
                "min": round(min(self.relevance_scores), 3),
                "max": round(max(self.relevance_scores), 3)
            },
            "hallucination": {
                "mean_risk": round(sum(self.hallucination_scores) / len(self.hallucination_scores), 3),
                "min_risk": round(min(self.hallucination_scores), 3),
                "max_risk": round(max(self.hallucination_scores), 3)
            },
            "overall": {
                "mean": round(sum(self.overall_scores) / len(self.overall_scores), 3),
                "min": round(min(self.overall_scores), 3),
                "max": round(max(self.overall_scores), 3)
            }
        }
    
    def _empty_summary(self) -> Dict[str, Any]:
        """Return empty summary."""
        return {
            "total_evaluations": 0,
            "relevance": {"mean": 0, "min": 0, "max": 0},
            "hallucination": {"mean_risk": 0, "min_risk": 0, "max_risk": 0},
            "overall": {"mean": 0, "min": 0, "max": 0}
        }


class StatisticsTracker:
    """
    Main statistics tracking system.
    
    Combines performance tracking, metric aggregation, and reporting.
    """
    
    def __init__(self, window_size: int = 1000):
        self.performance_tracker = PerformanceTracker(window_size)
        self.metric_aggregator = MetricAggregator()
        
        # Component-specific stats
        self.model_load_times = {}
        self.cache_stats = {}
        
        # Session info
        self.session_start = datetime.utcnow()
        
        logger.info("Statistics tracker initialized")
    
    def record_evaluation(
        self,
        metrics: EvaluationMetrics,
        cost_usd: float
    ) -> None:
        """Record a complete evaluation."""
        # Track performance
        self.performance_tracker.record(
            latency_ms=metrics.total_latency_ms,
            overall_score=metrics.overall_score,
            cost_usd=cost_usd
        )
        
        # Aggregate metrics
        self.metric_aggregator.add_evaluation(
            relevance_score=metrics.relevance.score,
            hallucination_score=metrics.hallucination.score,
            overall_score=metrics.overall_score
        )
    
    def record_model_load_time(self, model_name: str, load_time_ms: float) -> None:
        """Record model initialization time."""
        self.model_load_times[model_name] = load_time_ms
        logger.info(f"Model {model_name} loaded in {load_time_ms:.2f}ms")
    
    def update_cache_stats(self, cache_stats: Dict[str, Any]) -> None:
        """Update cache statistics."""
        self.cache_stats = cache_stats
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get all statistics in one report."""
        session_duration = (datetime.utcnow() - self.session_start).total_seconds()
        
        return {
            "session": {
                "start_time": self.session_start.isoformat(),
                "duration_seconds": round(session_duration, 2),
                "uptime_hours": round(session_duration / 3600, 2)
            },
            "performance": self.performance_tracker.get_stats(),
            "metrics": self.metric_aggregator.get_summary(),
            "cache": self.cache_stats,
            "models": {
                "load_times_ms": self.model_load_times
            }
        }
    
    def get_health_check(self) -> Dict[str, Any]:
        """Quick health check status."""
        perf_stats = self.performance_tracker.get_stats()
        
        # Define health thresholds
        latency_healthy = perf_stats['latency']['p95_ms'] < 500 if perf_stats['window_size'] > 0 else True
        quality_healthy = perf_stats['quality']['mean_score'] > 0.6 if perf_stats['window_size'] > 0 else True
        cache_healthy = (
            self.cache_stats.get('overall_hit_rate_percent', 0) > 30
            if self.cache_stats else True
        )
        
        overall_healthy = latency_healthy and quality_healthy and cache_healthy
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "checks": {
                "latency": "ok" if latency_healthy else "slow",
                "quality": "ok" if quality_healthy else "poor",
                "cache": "ok" if cache_healthy else "low_hit_rate"
            },
            "metrics": {
                "p95_latency_ms": perf_stats['latency']['p95_ms'],
                "mean_quality_score": perf_stats['quality']['mean_score'],
                "cache_hit_rate_percent": self.cache_stats.get('overall_hit_rate_percent', 0)
            }
        }
    
    def export_stats(self, filepath: str) -> None:
        """Export statistics to JSON file."""
        stats = self.get_comprehensive_stats()
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics exported to {filepath}")
    
    def print_summary(self) -> None:
        """Print formatted statistics summary."""
        stats = self.get_comprehensive_stats()
        
        print("\n" + "="*60)
        print("EVALUATION PIPELINE STATISTICS")
        print("="*60)
        
        print(f"\nSession Uptime: {stats['session']['uptime_hours']:.2f} hours")
        
        if stats['performance']['window_size'] > 0:
            print(f"\nPerformance (last {stats['performance']['window_size']} evaluations):")
            print(f"  Latency P50: {stats['performance']['latency']['p50_ms']:.2f}ms")
            print(f"  Latency P95: {stats['performance']['latency']['p95_ms']:.2f}ms")
            print(f"  Throughput: {stats['performance']['throughput']['evaluations_per_second']:.2f} eval/sec")
            
            print(f"\nQuality Metrics:")
            print(f"  Mean Score: {stats['metrics']['overall']['mean']:.3f}")
            print(f"  Mean Relevance: {stats['metrics']['relevance']['mean']:.3f}")
            print(f"  Mean Hallucination Risk: {stats['metrics']['hallucination']['mean_risk']:.3f}")
            
            print(f"\nCost Estimates:")
            print(f"  Mean Cost per Eval: ${stats['performance']['cost']['mean_usd']:.6f}")
            print(f"  Estimated Daily Cost: ${stats['performance']['cost']['estimated_daily_usd']:.2f}")
        
        if stats['cache']:
            print(f"\nCache Performance:")
            print(f"  Overall Hit Rate: {stats['cache'].get('overall_hit_rate_percent', 0):.2f}%")
        
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = StatisticsTracker(window_size=100)
    
    # Simulate some evaluations
    for i in range(50):
        # Create mock metrics
        relevance = MetricScore("relevance", 0.7 + (i % 10) * 0.02)
        hallucination = MetricScore("hallucination", 0.2 + (i % 8) * 0.01)
        
        metrics = EvaluationMetrics(
            evaluation_id=f"eval_{i}",
            conversation_id=f"conv_{i}",
            message_id=f"msg_{i}",
            relevance=relevance,
            hallucination=hallucination,
            performance={"latency_ms": 150 + i},
            overall_score=0.75,
            total_latency_ms=150 + i
        )
        
        tracker.record_evaluation(metrics, cost_usd=0.0001)
    
    # Print summary
    tracker.print_summary()
    
    # Health check
    health = tracker.get_health_check()
    print(f"Health Status: {health}")