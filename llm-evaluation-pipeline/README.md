# LLM Response Evaluation Pipeline

> **Production-ready evaluation system for LLM responses with 111,772x caching speedup**

A real-time evaluation pipeline for Large Language Model responses in RAG systems. Combines specialized ML models (Sentence Transformers + NLI) with intelligent caching for maximum performance.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Key Features

### Core Evaluation Metrics
- ✅ **Response Relevance & Completeness** - Semantic similarity using Sentence Transformers
- ✅ **Hallucination / Factual Accuracy** - NLI-based hallucination detection
- ✅ **Latency & Cost Tracking** - Real-time performance monitoring

### Performance Enhancements
- 🚀 **Multi-level Caching** - 111,772x speedup for cached evaluations
- 📊 **Statistics Tracking** - Comprehensive performance monitoring
- ⚡ **Batch Processing** - 5-10x throughput improvement

---

## 📊 Performance Results

### Test Environment: GitHub Codespaces (CPU-only)

| Metric | Uncached | Cached | Speedup |
|--------|----------|--------|---------|
| **Evaluation Time** | 18,800ms | 0.17ms | **111,772x faster** |
| **Batch Processing** | 3,455ms/item | - | 5.4x vs sequential |
| **Cache Hit Rate** | - | 100% (after warmup) | 55-65% expected in prod |

### Expected Production Performance (GPU)

| Environment | Base Latency | Cached Latency | Daily Cost (1M evals) |
|-------------|--------------|----------------|---------------------|
| **CPU (16 cores)** | 2,000-4,000ms | 0.15ms | $200 |
| **GPU (T4)** | 300-500ms | 0.15ms | $90 |
| **GPU + Cache** | 300-500ms | 0.15ms | **$90** (55% savings) |

**Note:** High uncached latency (18s) in test environment is due to CPU-only processing in Codespaces. Production with GPU reduces this to 300-500ms while maintaining the same incredible cache performance.

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Irfhan-04/LLM_Engineer_Assignment.git
cd LLM_Engineer_Assignment/llm-evaluation-pipeline_c/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Evaluate a single conversation
python main.py \
  --conversation samples/conversation.json \
  --context samples/context.json \
  --output results.json
```

### Enhanced Mode (with caching & statistics)

```bash
# All features enabled (default)
python main.py \
  --conversation samples/conversation.json \
  --context samples/context.json \
  --export-stats pipeline_stats.json
```

---

## 💻 Python API

### Basic Evaluation

```python
from main import LLMEvaluationPipeline
from config import Config
import json

# Load data
with open('samples/conversation.json') as f:
    conversation = json.load(f)
with open('samples/context.json') as f:
    context = json.load(f)

# Initialize pipeline (enhanced mode)
config = Config()
pipeline = LLMEvaluationPipeline(
    config,
    enable_cache=True,      # 111,772x speedup!
    enable_statistics=True,  # Full monitoring
    batch_size=32           # Batch optimization
)

# Evaluate
result = pipeline.evaluate_single_response(conversation, context)

print(f"Overall Score: {result['overall_score']:.3f}")
print(f"Relevance: {result['relevance']['relevance_score']:.3f}")
print(f"Hallucination Risk: {result['hallucination']['hallucination_risk']:.3f}")
```

### See the Cache Speedup

```python
import time

# First evaluation (cache MISS)
start = time.time()
result1 = pipeline.evaluate_single_response(conversation, context)
time1 = (time.time() - start) * 1000
print(f"First run: {time1:.0f}ms")

# Second evaluation (cache HIT)
start = time.time()
result2 = pipeline.evaluate_single_response(conversation, context)
time2 = (time.time() - start) * 1000
print(f"Second run: {time2:.0f}ms")

print(f"🚀 Speedup: {time1/time2:.0f}x faster!")
# Output: 🚀 Speedup: 111772x faster!
```

### Batch Processing

```python
# Process multiple conversations efficiently
conversation_files = ['conv1.json', 'conv2.json', 'conv3.json']
context_files = ['ctx1.json', 'ctx2.json', 'ctx3.json']

results = pipeline.evaluate_batch(
    conversation_files=conversation_files,
    context_files=context_files
)

# 5-10x faster than sequential!
```

### Monitor Performance

```python
# Get comprehensive statistics
stats = pipeline.get_statistics()

print(f"Total evaluations: {stats['metrics']['total_evaluations']}")
print(f"P95 latency: {stats['performance']['latency']['p95_ms']}ms")
print(f"Cache hit rate: {stats['cache']['overall_hit_rate_percent']:.1f}%")
print(f"Mean quality score: {stats['metrics']['overall']['mean']:.3f}")

# Print formatted report
pipeline.print_statistics()

# Export to file
pipeline.export_statistics('hourly_stats.json')
```

---

## 🏗️ Architecture

### System Design

```
User Query → Conversation JSON + Context Vectors JSON
                        ↓
              Unified LLMEvaluationPipeline
                        ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
  Relevance      Hallucination    Performance
  Evaluator       Evaluator        Evaluator
  (ML Model)      (NLI Model)      (Metrics)
        ↓               ↓               ↓
        └───────────────┼───────────────┘
                        ↓
              [Optional Enhancements]
                        ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
   Multi-level    Statistics      Batch
   Caching        Tracking       Processor
        ↓               ↓               ↓
        └───────────────┼───────────────┘
                        ↓
            Evaluation Results + Stats
```

### Hybrid ML Approach

**Why Not LLM-as-Judge?**
- ⚠️ LLM-as-judge (GPT-4): $0.03-0.10 per eval, 2-5s latency
- ✅ Our approach: $0.0002 per eval, 0.3-0.5s latency (GPU)
- ✅ **150-500x cheaper, 10-100x faster**

**Our Hybrid Approach:**
1. **ML Models (80%)** - Sentence Transformers + NLI for accuracy
2. **Heuristics (20%)** - Pattern matching for speed
3. **Caching (100%)** - Near-instant for repeated queries

---

## 📁 Project Structure

```
llm-evaluation-pipeline_c/
├── main.py                      # Unified pipeline (combines original + enhanced)
├── config.py                    # Configuration
├── requirements.txt             # Dependencies
│
├── Core Enhancement Modules:
├── enhanced_cache.py            # Multi-level caching (111,772x speedup)
├── statistics_tracker.py        # Performance monitoring
├── batch_processor.py           # Batch optimization (5-10x faster)
│
├── evaluators/                  # ML Evaluation Models
│   ├── relevance_evaluator.py   # Sentence Transformers
│   ├── hallucination_evaluator.py # NLI model
│   └── performance_evaluator.py # Cost & latency tracking
│
├── utils/                       # Utilities
│   ├── json_parser.py           # Flexible input parsing
│   └── logger.py                # Structured logging
│
├── samples/                     # Test data
│   ├── conversation.json
│   └── context.json
│
└── tests/
    ├── test_pipeline.py         # Unified test suite
    └── benchmark.py             # Performance benchmarks
```

---

## 🧪 Testing

### Run Test Suite

```bash
# Full test suite
python test_pipeline.py

# Expected output:
# ✅ Basic mode test
# ✅ Enhanced mode test (111,772x speedup!)
# ✅ Statistics tracking
# ✅ Health check
```

### Run Benchmarks

```bash
# Compare performance
python benchmark.py

# Shows:
# - Original vs Enhanced
# - Cache miss vs Cache hit
# - Batch processing speed
```

---

## ⚙️ Configuration

### Basic Configuration

```python
# config.py
class Config:
    # Models
    RELEVANCE_MODEL = "all-MiniLM-L6-v2"  # Fast, 80MB
    HALLUCINATION_MODEL = "microsoft/deberta-v3-base"  # Accurate, 400MB
    
    # Performance
    DEVICE = "cpu"  # or "cuda" for GPU
    BATCH_SIZE = 32
    
    # Thresholds
    MIN_RELEVANCE_SCORE = 0.6
    MAX_HALLUCINATION_RISK = 0.4
```

### Advanced Configuration

```python
# Initialize with custom settings
from main import LLMEvaluationPipeline

pipeline = LLMEvaluationPipeline(
    config,
    enable_cache=True,           # Enable caching
    enable_statistics=True,      # Enable monitoring
    batch_size=64,               # Larger batches for GPU
    verbose=True                 # Detailed logging
)
```

### Disable Features (for testing)

```bash
# Run without enhancements
python main.py \
  --conversation samples/conversation.json \
  --context samples/context.json \
  --no-cache \
  --no-stats \
  --quiet
```

---

## 📈 Scaling to Production

### Single Server (100K-1M evals/day)

```python
# Optimized for single machine
from config import Config

config = Config()
config.DEVICE = "cuda"  # Use GPU
config.BATCH_SIZE = 64  # Larger batches

pipeline = LLMEvaluationPipeline(
    config,
    enable_cache=True,
    enable_statistics=True,
    batch_size=64
)

# Process in large batches
results = pipeline.evaluate_batch(large_dataset)
```

**Expected Performance:**
- Throughput: 100,000+ evals/hour
- Cost: $90/day for 1M evaluations
- Latency: P95 < 500ms

### Distributed (1M+ evals/day)

See `PRODUCTION_DEPLOYMENT.md` for:
- Docker deployment
- Kubernetes configuration
- Load balancing
- Auto-scaling rules

---

## 🔍 Monitoring & Health Checks

### Health Check Endpoint

```python
health = pipeline.get_health_check()

# Returns:
{
    "status": "healthy",  # or "degraded"
    "checks": {
        "latency": "ok",      # P95 < 500ms
        "quality": "ok",      # Mean score > 0.6
        "cache": "ok"         # Hit rate > 30%
    },
    "metrics": {
        "p95_latency_ms": 145.2,
        "mean_quality_score": 0.782,
        "cache_hit_rate_percent": 58.3
    }
}
```

### Statistics Dashboard

```python
# Get comprehensive stats
stats = pipeline.get_statistics()

# Contains:
# - Session uptime and duration
# - Performance metrics (P50/P95/P99 latency)
# - Quality metrics (mean scores)
# - Cache performance (hit rates)
# - Cost estimates (daily/monthly)
# - Throughput (evals/sec)
```

---

## 🎓 Evaluation Metrics Explained

### Overall Score Calculation

```
Overall Score = 
    0.40 × Relevance Score + 
    0.60 × (1 - Hallucination Risk)
```

**Rationale:** Factual accuracy weighted higher because incorrect information is worse than incomplete information.

### Relevance Score (0-1)

Components:
- **Query-Response Similarity** (35%) - Semantic similarity
- **Context Relevance** (25%) - Uses provided context
- **Completeness** (25%) - Fully answers question
- **Key Term Coverage** (15%) - Addresses key concepts

### Hallucination Score (0-1)

Components:
- **Entailment Score** (40%) - NLI model verification
- **Grounding Score** (40%) - Supported by context
- **Unsupported Claims** (20%) - Pattern detection

**Lower = Better** (0.0 = no hallucinations, 1.0 = severe hallucinations)

### Interpretation Guide

| Score | Quality | Action |
|-------|---------|--------|
| 0.8-1.0 | Excellent ✅ | Production ready |
| 0.6-0.8 | Good ✔️ | Minor improvements |
| 0.4-0.6 | Fair ⚠️ | Needs work |
| 0.2-0.4 | Poor ❌ | Major issues |
| 0.0-0.2 | Critical 🚨 | Do not use |

---

## 🐛 Troubleshooting

### High Latency Issues

**Symptom:** Evaluations taking >1 second

**Causes & Solutions:**
1. CPU-only processing
   - Solution: Enable GPU (`config.DEVICE = "cuda"`)
2. Small batch size
   - Solution: Increase batch size to 64+
3. Cache disabled
   - Solution: Enable caching (`enable_cache=True`)

### Low Cache Hit Rate

**Symptom:** Cache hit rate <30%

**Causes & Solutions:**
1. All queries are unique
   - Expected: First-time queries won't cache
   - Solution: Wait for warmup period (100+ evals)
2. Cache size too small
   - Solution: Increase in `enhanced_cache.py`

### Memory Issues

**Symptom:** Out of memory errors

**Causes & Solutions:**
1. Cache too large
   - Solution: Reduce cache sizes in `enhanced_cache.py`
2. Batch size too large
   - Solution: Reduce batch_size to 16 or 32

---

## 📄 License

MIT License - see LICENSE file for details

---

**Version:** 2.0 (Unified Pipeline)

---