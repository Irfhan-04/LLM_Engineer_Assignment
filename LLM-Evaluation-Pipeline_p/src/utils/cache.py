"""Caching utility for evaluation results."""

import hashlib
from typing import Any, Dict, Optional
from datetime import datetime, timedelta


class CacheEntry:
    """Single cache entry with TTL."""

    def __init__(self, value: Any, ttl_seconds: int = 3600):
        self.value = value
        self.created_at = datetime.utcnow()
        self.ttl_seconds = ttl_seconds

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds

    def get(self) -> Optional[Any]:
        """Get value if not expired."""
        if not self.is_expired():
            return self.value
        return None


class EvaluationCache:
    """In-memory cache for evaluation results."""

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _generate_key(conversation_id: str, message_id: str, metric_type: str) -> str:
        """Generate a cache key from evaluation parameters."""
        key_str = f"{conversation_id}:{message_id}:{metric_type}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(
        self, conversation_id: str, message_id: str, metric_type: str
    ) -> Optional[Any]:
        """Retrieve cached evaluation result."""
        key = self._generate_key(conversation_id, message_id, metric_type)

        if key not in self.cache:
            self.misses += 1
            return None

        value = self.cache[key].get()
        if value is None:
            del self.cache[key]
            self.misses += 1
            return None

        self.hits += 1
        return value

    def set(
        self,
        conversation_id: str,
        message_id: str,
        metric_type: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Store evaluation result in cache."""
        key = self._generate_key(conversation_id, message_id, metric_type)
        ttl = ttl_seconds or self.default_ttl

        if len(self.cache) >= self.max_size:
            self._evict_expired()

        self.cache[key] = CacheEntry(value, ttl)

    def _evict_expired(self) -> None:
        """Remove expired entries."""
        expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
        for key in expired_keys:
            del self.cache[key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": hit_rate,
            "cache_size": len(self.cache),
        }
