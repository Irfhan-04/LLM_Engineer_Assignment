"""
Configuration Module

Centralized configuration for the evaluation pipeline.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration for LLM evaluation pipeline."""
    
    # Model configurations
    RELEVANCE_MODEL: str = "all-MiniLM-L6-v2"  # Fast and efficient
    HALLUCINATION_MODEL: str = "microsoft/deberta-v3-base"  # NLI model
    DEFAULT_MODEL: str = "gpt-3.5-turbo"  # For cost estimation
    
    # Performance thresholds
    MAX_LATENCY_MS: float = 1000.0
    WARNING_LATENCY_MS: float = 500.0
    
    # Scoring thresholds
    MIN_RELEVANCE_SCORE: float = 0.6
    MAX_HALLUCINATION_RISK: float = 0.4
    
    # Batch processing
    BATCH_SIZE: int = 32
    MAX_WORKERS: int = 4
    
    # Caching (for scale optimization)
    ENABLE_EMBEDDING_CACHE: bool = True
    CACHE_SIZE: int = 10000
    
    # Enhancement settings (optional - used if enhancement modules available)
    ENABLE_CACHE_BY_DEFAULT = True
    ENABLE_STATISTICS_BY_DEFAULT = True
    ENABLE_BATCH_BY_DEFAULT = True
    
    # Cache configuration
    EMBEDDING_CACHE_SIZE = 10000
    EVALUATION_CACHE_SIZE = 5000
    CACHE_TTL_SECONDS = 3600
    
    # Statistics configuration
    STATISTICS_WINDOW_SIZE = 1000
    
    # Batch configuration
    DEFAULT_BATCH_SIZE = 32
    MAX_CONCURRENT_BATCHES = 4
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "evaluation_pipeline.log"
    
    # Context handling
    MAX_CONTEXT_LENGTH: int = 2048  # Max tokens per context
    CONTEXT_OVERLAP: int = 128
    
    # Device configuration
    DEVICE: str = "cpu"  # or "cuda" for GPU
    
    # API keys (from environment)
    OPENAI_API_KEY: Optional[str] = None
    
    def __post_init__(self):
        """Load environment variables."""
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        # Allow environment overrides
        self.RELEVANCE_MODEL = os.getenv(
            "RELEVANCE_MODEL", 
            self.RELEVANCE_MODEL
        )
        self.HALLUCINATION_MODEL = os.getenv(
            "HALLUCINATION_MODEL", 
            self.HALLUCINATION_MODEL
        )
        self.DEVICE = os.getenv("DEVICE", self.DEVICE)
        
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }