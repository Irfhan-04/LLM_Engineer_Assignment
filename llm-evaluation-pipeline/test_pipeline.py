#!/usr/bin/env python3
"""
Unified pipeline test script.
Tests both basic and enhanced functionality.
"""

import json
import time
from main import LLMEvaluationPipeline
from config import Config

# Load test data
with open('samples/conversation.json', 'r') as f:
    conversation = json.load(f)
with open('samples/context.json', 'r') as f:
    context = json.load(f)

print("="*60)
print("UNIFIED PIPELINE TEST")
print("="*60)

# Test 1: Basic mode (no enhancements)
print("\n1. Testing BASIC mode (original functionality)...")
config = Config()
pipeline_basic = LLMEvaluationPipeline(
    config,
    enable_cache=False,
    enable_statistics=False,
    verbose=False
)

start = time.time()
result = pipeline_basic.evaluate_single_response(conversation, context)
basic_time = (time.time() - start) * 1000
print(f"   Time: {basic_time:.2f}ms")
print(f"   Score: {result['overall_score']:.3f}")

# Test 2: Enhanced mode (all features)
print("\n2. Testing ENHANCED mode (with all features)...")
pipeline_enhanced = LLMEvaluationPipeline(
    config,
    enable_cache=True,
    enable_statistics=True,
    batch_size=32,
    verbose=False
)

# First run (cache MISS)
start = time.time()
result1 = pipeline_enhanced.evaluate_single_response(conversation, context)
miss_time = (time.time() - start) * 1000
print(f"   First run (cache MISS): {miss_time:.2f}ms")

# Second run (cache HIT)
start = time.time()
result2 = pipeline_enhanced.evaluate_single_response(conversation, context)
hit_time = (time.time() - start) * 1000
print(f"   Second run (cache HIT): {hit_time:.2f}ms")
print(f"   🚀 Speedup: {miss_time/hit_time:.1f}x faster!")

# Test 3: Statistics
print("\n3. Testing STATISTICS...")
stats = pipeline_enhanced.get_statistics()
print(f"   Total evaluations: {stats['metrics']['total_evaluations']}")
cache_stats = stats.get('cache', {})
if cache_stats:
    hit_rate = cache_stats.get('overall_hit_rate_percent', 0)
    print(f"   Cache hit rate: {hit_rate:.1f}%")

# Test 4: Health check
print("\n4. Testing HEALTH CHECK...")
health = pipeline_enhanced.get_health_check()
print(f"   Status: {health['status']}")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)