#!/usr/bin/env python3
"""Benchmark original vs enhanced pipeline."""

import json
import time
from main import LLMEvaluationPipeline as OriginalPipeline
from enhanced_main import EnhancedLLMEvaluationPipeline
from config import Config

# Load test data
with open('samples/conversation.json', 'r') as f:
    conversation = json.load(f)
with open('samples/context.json', 'r') as f:
    context = json.load(f)

config = Config()

print("="*60)
print("PERFORMANCE BENCHMARK: Original vs Enhanced")
print("="*60)

# Benchmark original
print("\n1. Testing ORIGINAL pipeline...")
original_times = []
pipeline_orig = OriginalPipeline(config)

for i in range(3):
    start = time.time()
    result = pipeline_orig.evaluate_single_response(conversation, context)
    elapsed = (time.time() - start) * 1000
    original_times.append(elapsed)
    print(f"  Run {i+1}: {elapsed:.2f}ms")

avg_original = sum(original_times) / len(original_times)
print(f"  Average: {avg_original:.2f}ms")

# Benchmark enhanced (first run - cache miss)
print("\n2. Testing ENHANCED pipeline (cache MISS)...")
enhanced_times_miss = []
pipeline_enh = EnhancedLLMEvaluationPipeline(config, enable_cache=True)

start = time.time()
result = pipeline_enh.evaluate_single_response(conversation, context)
elapsed = (time.time() - start) * 1000
enhanced_times_miss.append(elapsed)
print(f"  First run: {elapsed:.2f}ms")

# Benchmark enhanced (cached runs)
print("\n3. Testing ENHANCED pipeline (cache HIT)...")
enhanced_times_hit = []

for i in range(3):
    start = time.time()
    result = pipeline_enh.evaluate_single_response(conversation, context)
    elapsed = (time.time() - start) * 1000
    enhanced_times_hit.append(elapsed)
    print(f"  Run {i+1}: {elapsed:.2f}ms")

avg_hit = sum(enhanced_times_hit) / len(enhanced_times_hit)
print(f"  Average (cached): {avg_hit:.2f}ms")

# Results
print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"Original pipeline:     {avg_original:.2f}ms")
print(f"Enhanced (first run):  {enhanced_times_miss[0]:.2f}ms")
print(f"Enhanced (cached):     {avg_hit:.2f}ms")
print(f"\n🚀 Speedup (cached):    {avg_original/avg_hit:.1f}x faster!")
print("="*60)

# Statistics
stats = pipeline_enh.get_statistics()
print("\nCache Statistics:")
cache_stats = stats.get('cache', {})
if 'embedding_cache' in cache_stats:
    print(f"  Embedding cache hits: {cache_stats['embedding_cache']['hits']}")
    print(f"  Embedding cache misses: {cache_stats['embedding_cache']['misses']}")
    print(f"  Hit rate: {cache_stats['embedding_cache']['hit_rate_percent']:.1f}%")

# Create multiple test cases
test_cases = [
    (conversation, context) for _ in range(100)
]

print("\n4. Testing BATCH processing...")
start = time.time()

# Create batch items
from batch_processor import BatchItem
batch_items = [
    BatchItem(
        id=f"item_{i}",
        query="Test query",
        response="Test response",
        context=context['retrieved_documents']
    )
    for i in range(100)
]

# Process batch
results = pipeline_enh.batch_processor.evaluate_batch(batch_items)
batch_time = time.time() - start

print(f"  100 items in {batch_time:.2f}s")
print(f"  Per-item: {(batch_time/100)*1000:.2f}ms")
print(f"  Throughput: {100/batch_time:.1f} items/sec")