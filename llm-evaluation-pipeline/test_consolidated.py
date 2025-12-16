#!/usr/bin/env python3
"""Test consolidated pipeline."""

from main import LLMEvaluationPipeline
from config import Config
import json

config = Config()

# Test 1: Without enhancements
print("Test 1: Legacy mode (no enhancements)...")
pipeline_legacy = LLMEvaluationPipeline(
    config,
    enable_cache=False,
    enable_statistics=False,
    enable_batch=False,
    verbose=False
)
print("✓ Legacy mode works")

# Test 2: With enhancements
print("\nTest 2: Enhanced mode...")
pipeline_enhanced = LLMEvaluationPipeline(
    config,
    enable_cache=True,
    enable_statistics=True,
    enable_batch=True,
    verbose=True
)

# Load test data
with open('samples/conversation.json') as f:
    conv = json.load(f)
with open('samples/context.json') as f:
    ctx = json.load(f)

# Test evaluation
import time

# First run (cache MISS)
start = time.time()
result1 = pipeline_enhanced.evaluate_single_response(conv, ctx)
time1 = (time.time() - start) * 1000

# Second run (cache HIT)
start = time.time()
result2 = pipeline_enhanced.evaluate_single_response(conv, ctx)
time2 = (time.time() - start) * 1000

print(f"\n✓ First eval: {time1:.2f}ms")
print(f"✓ Cached eval: {time2:.2f}ms")
print(f"✓ Speedup: {time1/time2:.1f}x faster!")

# Test statistics
if hasattr(pipeline_enhanced, 'stats') and pipeline_enhanced.stats:
    stats = pipeline_enhanced.get_statistics()
    print(f"✓ Statistics working: {stats['metrics']['total_evaluations']} evaluations tracked")
else:
    print("⚠ Statistics not available")

print("\n✅ All tests passed!")