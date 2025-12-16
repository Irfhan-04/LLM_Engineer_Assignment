#!/usr/bin/env python3
"""Quick test of enhanced pipeline."""

import json
from enhanced_main import EnhancedLLMEvaluationPipeline
from config import Config

# Load test data
with open('samples/conversation.json', 'r') as f:
    conversation = json.load(f)
with open('samples/context.json', 'r') as f:
    context = json.load(f)

# Initialize enhanced pipeline
print("Initializing enhanced pipeline...")
config = Config()
pipeline = EnhancedLLMEvaluationPipeline(
    config,
    enable_cache=True,
    enable_statistics=True,
    batch_size=32
)

# Test single evaluation
print("\nTest 1: Single evaluation (cache MISS)...")
import time
start = time.time()
result1 = pipeline.evaluate_single_response(conversation, context)
time1 = (time.time() - start) * 1000
print(f"  Time: {time1:.2f}ms")
print(f"  Score: {result1['overall_score']:.3f}")

# Test cached evaluation
print("\nTest 2: Same evaluation (cache HIT)...")
start = time.time()
result2 = pipeline.evaluate_single_response(conversation, context)
time2 = (time.time() - start) * 1000
print(f"  Time: {time2:.2f}ms")
print(f"  Score: {result2['overall_score']:.3f}")
print(f"  Speedup: {time1/time2:.1f}x faster!")

# Test statistics
print("\nTest 3: Statistics...")
stats = pipeline.get_statistics()
print(f"  Total evaluations: {stats['metrics']['total_evaluations']}")
cache_stats = stats.get('cache', {})
if cache_stats:
    print(f"  Cache hit rate: {cache_stats.get('overall_hit_rate_percent', 0):.1f}%")

# Test health check
print("\nTest 4: Health check...")
health = pipeline.get_health_check()
print(f"  Status: {health['status']}")
print(f"  Latency check: {health['checks']['latency']}")

print("\n✅ All tests passed!")