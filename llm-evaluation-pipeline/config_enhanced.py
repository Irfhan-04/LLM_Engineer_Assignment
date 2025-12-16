"""
Enhanced configuration with cache and batch settings.
"""
from config import Config as BaseConfig


class EnhancedConfig(BaseConfig):
    """Enhanced configuration with additional settings."""
    
    # Cache settings
    ENABLE_CACHE = True
    EMBEDDING_CACHE_SIZE = 10000
    EVALUATION_CACHE_SIZE = 5000
    CACHE_TTL_SECONDS = 3600  # 1 hour
    
    # Batch processing
    BATCH_SIZE = 32
    MAX_CONCURRENT_BATCHES = 4
    
    # Statistics
    ENABLE_STATISTICS = True
    STATISTICS_WINDOW_SIZE = 1000
    
    # Export settings
    AUTO_EXPORT_STATS = False
    STATS_EXPORT_INTERVAL = 3600  # 1 hour