# Quick Start Guide 🚀

Get the LLM Evaluation Pipeline running in **5 minutes**.

---

## 30-Second Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run test (downloads models automatically)
python test_pipeline.py

# 3. Evaluate your data
python main.py --conversation <your_conversation.json> --context <your_context.json>
```

That's it! The pipeline is ready to use.

---

## Step-by-Step Setup

### 1️⃣ Install Python Dependencies

```bash
pip install -r requirements.txt
```

**What it installs:**
- PyTorch (ML framework)
- Transformers (NLP models)
- Sentence Transformers (embeddings)
- Tiktoken (token counting)

**Time**: 2-3 minutes

### 2️⃣ Download Models (Automatic)

```bash
python test_pipeline.py
```

**What happens:**
- Downloads `all-MiniLM-L6-v2` (~80MB) for relevance
- Downloads `microsoft/deberta-v3-base` (~400MB) for hallucination detection
- Creates sample data
- Runs a test evaluation

**Time**: 1-2 minutes (depending on internet speed)

### 3️⃣ Evaluate Your Data

```bash
python main.py \
  --conversation path/to/conversation.json \
  --context path/to/context.json \
  --output results.json
```

**Output**: JSON file with comprehensive evaluation metrics

---

## JSON Format Requirements

### Conversation JSON

Your conversation file should have messages with these fields:

```json
{
  "messages": [
    {
      "role": "user",           // or "human", "customer"
      "content": "Your question?"
    },
    {
      "role": "assistant",      // or "ai", "bot"
      "content": "AI response"
    }
  ]
}
```

### Context JSON

Your context file should have retrieved documents:

```json
{
  "retrieved_documents": [
    {
      "text": "Context content...",   // or "content"
      "score": 0.95                   // optional
    }
  ]
}
```

**Don't worry!** The parser is flexible and handles many formats automatically. See `samples/` for examples.

---

## Understanding Results

### Example Output

```json
{
  "overall_score": 0.842,
  "relevance": {
    "relevance_score": 0.876,
    "assessment": "Excellent - Highly relevant and complete"
  },
  "hallucination": {
    "hallucination_risk": 0.123,
    "factual_accuracy": 0.877,
    "assessment": "Excellent - Fully grounded in context"
  },
  "performance": {
    "latency_ms": 342.5,
    "estimated_cost_usd": 0.000015
  }
}
```

### Scoring Guide

| Overall Score | Quality Level | Action |
|---------------|---------------|---------|
| 0.8 - 1.0 | Excellent ✅ | Production ready |
| 0.6 - 0.8 | Good ✔️ | Minor improvements |
| 0.4 - 0.6 | Fair ⚠️ | Needs work |
| 0.0 - 0.4 | Poor ❌ | Major issues |

---

## Common Issues & Solutions

### ❓ "ModuleNotFoundError"

**Solution**: Ensure you're in the virtual environment

```bash
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### ❓ "torch not found" or CUDA errors

**Solution**: Install CPU version of PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### ❓ "Out of memory" error

**Solution**: Reduce batch size in `config.py`

```python
BATCH_SIZE = 16  # or even 8 for very limited RAM
```

### ❓ Models downloading slowly

**Solution**: Models download automatically on first run. Be patient or download manually:

```python
from sentence_transformers import SentenceTransformer

# Download in advance
model1 = SentenceTransformer('all-MiniLM-L6-v2')
model2 = SentenceTransformer('microsoft/deberta-v3-base')
```

---

## Advanced Usage

### Batch Processing

```python
from main import LLMEvaluationPipeline
from config import Config

pipeline = LLMEvaluationPipeline(Config())

# Evaluate multiple conversations
results = pipeline.evaluate_batch(
    conversation_files=["conv1.json", "conv2.json"],
    context_files=["ctx1.json", "ctx2.json"]
)

# Analyze
for result in results:
    print(f"Conversation {result['message_id']}: {result['overall_score']:.3f}")
```

### Custom Configuration

Edit `config.py` to customize:

```python
# Use different models
RELEVANCE_MODEL = "paraphrase-MiniLM-L6-v2"

# Adjust thresholds
MIN_RELEVANCE_SCORE = 0.7
MAX_HALLUCINATION_RISK = 0.3

# Enable GPU
DEVICE = "cuda"
```

### Integration Example

```python
# In your RAG application
from main import LLMEvaluationPipeline
from config import Config

# Initialize once
evaluator = LLMEvaluationPipeline(Config())

# Evaluate after generating response
def generate_and_evaluate(query, context):
    # Your RAG logic
    response = your_llm.generate(query, context)
    
    # Evaluate quality
    result = evaluator.evaluate_single_response(
        conversation_data={"messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]},
        context_data={"retrieved_documents": context}
    )
    
    # Log or act on results
    if result['overall_score'] < 0.6:
        logger.warning(f"Low quality response: {result['overall_score']}")
    
    return response, result
```

---

## Performance Tips

### 🚀 For Speed

1. **Enable caching** (default enabled)
2. **Use GPU** if available: `DEVICE = "cuda"` in config
3. **Batch processing**: Process multiple items together
4. **Model quantization**: Use 8-bit models

### 💰 For Cost

1. **This pipeline is FREE** after initial setup
2. **No API calls** required
3. **Runs completely offline**
4. **One-time model download**

### 🎯 For Accuracy

1. **Use larger models** (trade speed for accuracy)
2. **Adjust thresholds** based on your use case
3. **Ensemble multiple models**
4. **Fine-tune on your domain data**

---

## Next Steps

1. ✅ **Run test**: `python test_pipeline.py`
2. ✅ **Try samples**: Check `samples/` directory
3. ✅ **Read README**: Full documentation in README.md
4. ✅ **Customize**: Edit config.py for your needs
5. ✅ **Integrate**: Add to your RAG pipeline

---

## Need Help?

- Check `README.md` for detailed documentation
- See `samples/` for example inputs
- Review `test_pipeline.py` for usage examples

---

**Ready to evaluate! 🎉**

Run `python main.py --help` to see all options.