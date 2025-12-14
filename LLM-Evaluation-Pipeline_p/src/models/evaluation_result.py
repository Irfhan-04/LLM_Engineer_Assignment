"""Data models for evaluation results."""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class MetricScore:
    """Represents a single metric score."""

    name: str
    score: float  # 0-1 scale
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if not 0 <= self.score <= 1:
            raise ValueError(f"Score must be 0-1, got {self.score}")


@dataclass
class PerformanceMetrics:
    """Performance-related metrics."""

    latency_ms: float
    estimated_cost_usd: float
    tokens_generated: int
    model_name: Optional[str] = None


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    conversation_id: str
    message_id: str
    relevance_score: MetricScore
    hallucination_score: MetricScore
    performance_metrics: PerformanceMetrics
    overall_score: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "relevance_score": {
                "name": self.relevance_score.name,
                "score": self.relevance_score.score,
                "details": self.relevance_score.details,
            },
            "hallucination_score": {
                "name": self.hallucination_score.name,
                "score": self.hallucination_score.score,
                "details": self.hallucination_score.details,
            },
            "performance_metrics": asdict(self.performance_metrics),
            "overall_score": self.overall_score,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
