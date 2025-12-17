"""
Performance Benchmark: Original vs Enhanced Mode
Updated to work with unified main.py
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

config = Config()

print("="*60)
print("PERFORMANCE BENCHMARK: Basic vs Enhanced Mode")
print("="*60)

# Benchmark 1: Basic mode (no enhancements)
print("\n1. Testing BASIC mode (original functionality)...")
basic_times = []
pipeline_basic = LLMEvaluationPipeline(
    config,
    enable_cache=False,
    enable_statistics=False,
    verbose=False
)

for i in range(3):
    start = time.time()
    result = pipeline_basic.evaluate_single_response(conversation, context)
    elapsed = (time.time() - start) * 1000
    basic_times.append(elapsed)
    print(f"  Run {i+1}: {elapsed:.2f}ms")

avg_basic = sum(basic_times) / len(basic_times)
print(f"  Average: {avg_basic:.2f}ms")

# Benchmark 2: Enhanced mode (first run - cache miss)
print("\n2. Testing ENHANCED mode (cache MISS)...")
enhanced_miss_times = []
pipeline_enhanced = LLMEvaluationPipeline(
    config,
    enable_cache=True,
    enable_statistics=True,
    verbose=False
)

start = time.time()
result = pipeline_enhanced.evaluate_single_response(conversation, context)
elapsed = (time.time() - start) * 1000
enhanced_miss_times.append(elapsed)
print(f"  First run: {elapsed:.2f}ms")

# Benchmark 3: Enhanced mode (cached runs)
print("\n3. Testing ENHANCED mode (cache HIT)...")
enhanced_hit_times = []

for i in range(3):
    start = time.time()
    result = pipeline_enhanced.evaluate_single_response(conversation, context)
    elapsed = (time.time() - start) * 1000
    enhanced_hit_times.append(elapsed)
    print(f"  Run {i+1}: {elapsed:.2f}ms")

avg_hit = sum(enhanced_hit_times) / len(enhanced_hit_times)
print(f"  Average (cached): {avg_hit:.2f}ms")

# Results summary
print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"Basic mode (no cache):    {avg_basic:.2f}ms")
print(f"Enhanced (first run):     {enhanced_miss_times[0]:.2f}ms")
print(f"Enhanced (cached):        {avg_hit:.2f}ms")
print(f"\n🚀 Cache Speedup:         {avg_basic/avg_hit:.1f}x faster!")
print("="*60)

# Cache statistics
print("\nCache Statistics:")
stats = pipeline_enhanced.get_statistics()
cache_stats = stats.get('cache', {})

if cache_stats:
    emb_cache = cache_stats.get('embedding_cache', {})
    eval_cache = cache_stats.get('evaluation_cache', {})
    
    print(f"  Embedding cache:")
    print(f"    Hits: {emb_cache.get('hits', 0)}")
    print(f"    Misses: {emb_cache.get('misses', 0)}")
    print(f"    Hit rate: {emb_cache.get('hit_rate_percent', 0):.1f}%")
    
    print(f"  Evaluation cache:")
    print(f"    Hits: {eval_cache.get('hits', 0)}")
    print(f"    Misses: {eval_cache.get('misses', 0)}")
    print(f"    Hit rate: {eval_cache.get('hit_rate_percent', 0):.1f}%")
    
    print(f"  Overall hit rate: {cache_stats.get('overall_hit_rate_percent', 0):.1f}%")

# Batch processing benchmark
print("\n" + "="*60)
print("BATCH PROCESSING BENCHMARK")
print("="*60)

# Check if batch processor available
if pipeline_enhanced.batch_processor:
    print("\n4. Testing BATCH processing (100 items)...")
    
    # Create batch items
    from batch_processor import BatchItem
    batch_items = [
        BatchItem(
            id=f"item_{i}",
            query="What are the key features of Python 3.11?",
            response="Python 3.11 introduces several significant improvements...",
            context=context['retrieved_documents'] if 'retrieved_documents' in context else [],
            metadata={}
        )
        for i in range(100)
    ]
    
    # Process batch
    start = time.time()
    results = pipeline_enhanced.batch_processor.evaluate_batch(batch_items)
    batch_time = time.time() - start
    
    per_item_ms = (batch_time / 100) * 1000
    throughput = 100 / batch_time
    
    print(f"  100 items in {batch_time:.2f}s")
    print(f"  Per-item: {per_item_ms:.2f}ms")
    print(f"  Throughput: {throughput:.1f} items/sec")
    
    # Compare to sequential
    sequential_estimate = (avg_basic / 1000) * 100
    speedup = sequential_estimate / batch_time
    print(f"\n  Sequential estimate: {sequential_estimate:.2f}s")
    print(f"  🚀 Batch speedup: {speedup:.1f}x faster!")
else:
    print("\n⚠️  Batch processing not available")
    print("   (batch_processor.py not found)")

# Final summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"✅ Basic mode works: {avg_basic:.0f}ms per evaluation")
print(f"✅ Cache works: {avg_basic/avg_hit:.0f}x speedup")

if pipeline_enhanced.batch_processor:
    print(f"✅ Batch processing: {throughput:.1f} items/sec")
else:
    print(f"⚠️  Batch processing: Not available")

if stats.get('metrics', {}).get('total_evaluations', 0) > 0:
    print(f"✅ Statistics tracking: {stats['metrics']['total_evaluations']} evaluations recorded")
else:
    print(f"⚠️  Statistics tracking: Not available")

print("="*60)

# Health check
print("\nHealth Check:")
health = pipeline_enhanced.get_health_check()
print(f"  Status: {health.get('status', 'unknown')}")

if 'checks' in health:
    for check, status in health['checks'].items():
        print(f"  {check}: {status}")

print("\n✅ Benchmark complete!")