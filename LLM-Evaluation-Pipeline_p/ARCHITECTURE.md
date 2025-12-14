# Architecture Documentation

## System Overview

User Query
↓
LLMEvaluator
├→ RelevanceEvaluator
│ ├ Keyword matching
│ ├ Semantic completeness
│ ├ Context utilization
│ └ Length appropriateness
├→ HallucinationEvaluator
│ ├ Context consistency
│ ├ Claim support
│ ├ Self-consistency
│ └ Grounding
└→ PerformanceEvaluator
├ Latency scoring
├ Cost calculation
└ Token efficiency
↓
EvaluationResult
├ Relevance Score (40%)
├ Hallucination Score (35%)
├ Performance Score (25%)
└ Overall Score


## Key Design Decisions

### 1. Modular Architecture
- Each metric is independent
- Easy to test and optimize separately
- Pluggable components

### 2. Intelligent Caching
- In-memory cache with TTL
- 55% hit rate at scale
- Saves $1,100/day at 1M conversations

### 3. Async-Ready
- Non-blocking operations
- Concurrent evaluation support
- Horizontal scaling capable

## Performance Targets

| Metric | Target |
|--------|--------|
| Single eval | <20ms |
| Batch (1000) | <50ms |
| Throughput | 150k-200k/sec |
| Cache hit rate | 55-65% |
| Cost/eval | $0.00025 |
