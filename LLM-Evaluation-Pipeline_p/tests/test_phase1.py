"""Phase 1 foundation tests."""

import pytest
import tempfile
import json
from pathlib import Path
from src.models.evaluation_result import MetricScore, EvaluationResult
from src.utils.cache import EvaluationCache
from src.utils.json_loader import JSONLoader, JSONValidator


class TestCache:
    """Test caching system."""

    def test_cache_store_retrieve(self):
        cache = EvaluationCache()
        cache.set("conv_1", "msg_1", "relevance", 0.85)
        result = cache.get("conv_1", "msg_1", "relevance")
        assert result == 0.85

    def test_cache_miss(self):
        cache = EvaluationCache()
        result = cache.get("missing", "missing", "relevance")
        assert result is None
        assert cache.misses == 1


class TestJSONLoader:
    """Test JSON loading."""

    def test_load_and_validate_conversation(self):
        data = {
            "conversation_id": "conv_1",
            "messages": [
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "response"},
            ],
        }

        assert JSONValidator.validate_conversation(data) == True

    def test_invalid_conversation(self):
        data = {"messages": []}  # Missing conversation_id

        with pytest.raises(ValueError):
            JSONValidator.validate_conversation(data)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
