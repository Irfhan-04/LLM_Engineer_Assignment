#!/bin/bash
# Enhanced evaluation pipeline wrapper

python enhanced_main.py \
  --conversation "$1" \
  --context "$2" \
  --output "${3:-enhanced_results.json}" \
  --export-stats "${4:-pipeline_stats.json}"

echo "✓ Evaluation complete"
echo "  Results: ${3:-enhanced_results.json}"
echo "  Stats: ${4:-pipeline_stats.json}"