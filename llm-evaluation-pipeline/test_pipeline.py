"""
Quick test script to verify the evaluation pipeline works correctly.

Run this after installation to ensure everything is set up properly.
"""

import json
import sys
from pathlib import Path

def create_test_data():
    """Create sample test data if not exists."""
    
    # Create samples directory
    Path("samples").mkdir(exist_ok=True)
    
    # Sample conversation
    conversation = {
        "messages": [
            {
                "id": "msg_001",
                "role": "user",
                "content": "What is Python used for?",
                "timestamp": "2024-12-14T10:00:00Z"
            },
            {
                "id": "msg_002",
                "role": "assistant",
                "content": "Python is a versatile programming language used for web development, data science, machine learning, automation, and scientific computing. It's known for its simple syntax and extensive library ecosystem.",
                "timestamp": "2024-12-14T10:00:02Z"
            }
        ]
    }
    
    # Sample context
    context = {
        "retrieved_documents": [
            {
                "id": "doc_001",
                "text": "Python is a high-level, interpreted programming language. It is widely used in web development (Django, Flask), data analysis (pandas, numpy), machine learning (scikit-learn, TensorFlow), and automation. Python's simple syntax makes it popular for beginners and professionals alike.",
                "score": 0.95
            },
            {
                "id": "doc_002",
                "text": "Python has a rich ecosystem of libraries. The standard library is extensive, and PyPI hosts over 400,000 third-party packages. This makes Python suitable for almost any programming task, from simple scripts to complex applications.",
                "score": 0.82
            }
        ]
    }
    
    # Save files
    with open("samples/test_conversation.json", "w") as f:
        json.dump(conversation, f, indent=2)
    
    with open("samples/test_context.json", "w") as f:
        json.dump(context, f, indent=2)
    
    print("✓ Created test data files in samples/")


def test_imports():
    """Test that all required modules can be imported."""
    print("\n" + "="*60)
    print("Testing imports...")
    print("="*60)
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print("✓ Transformers imported successfully")
    except Exception as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ Sentence Transformers imported successfully")
    except Exception as e:
        print(f"✗ Sentence Transformers import failed: {e}")
        return False
    
    try:
        import tiktoken
        print("✓ Tiktoken imported successfully")
    except Exception as e:
        print(f"✗ Tiktoken import failed: {e}")
        return False
    
    return True


def test_pipeline():
    """Test the evaluation pipeline."""
    print("\n" + "="*60)
    print("Testing evaluation pipeline...")
    print("="*60)
    
    try:
        from main import LLMEvaluationPipeline
        from config import Config
        
        # Load test data
        with open("samples/test_conversation.json", "r") as f:
            conversation_data = json.load(f)
        
        with open("samples/test_context.json", "r") as f:
            context_data = json.load(f)
        
        # Initialize pipeline
        print("\n⏳ Initializing pipeline (this may take a minute on first run)...")
        config = Config()
        pipeline = LLMEvaluationPipeline(config)
        print("✓ Pipeline initialized")
        
        # Run evaluation
        print("\n⏳ Running evaluation...")
        result = pipeline.evaluate_single_response(
            conversation_data,
            context_data
        )
        print("✓ Evaluation completed")
        
        # Display results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Overall Score: {result['overall_score']:.3f}")
        print(f"\nRelevance:")
        print(f"  Score: {result['relevance']['relevance_score']:.3f}")
        print(f"  Assessment: {result['relevance']['assessment']}")
        print(f"\nHallucination:")
        print(f"  Risk: {result['hallucination']['hallucination_risk']:.3f}")
        print(f"  Assessment: {result['hallucination']['assessment']}")
        print(f"\nPerformance:")
        if result['performance'].get('generation_latency_ms'):
            print(f"  Generation Latency: {result['performance']['generation_latency_ms']:.2f}ms")
        else:
            print(f"  Generation Latency: Not available in metadata")
        print(f"  Estimated Cost: ${result['performance']['estimated_generation_cost_usd']:.6f}")
        print(f"  Evaluation Time: {result['performance']['evaluation_latency_ms']:.2f}ms")
        print("="*60)
                
        # Save results
        with open("test_results.json", "w") as f:
            json.dump(result, f, indent=2)
        print("\n✓ Results saved to test_results.json")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LLM Evaluation Pipeline - Test Suite")
    print("="*60)
    
    # Create test data
    create_test_data()
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your installation.")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Test pipeline
    if not test_pipeline():
        print("\n❌ Pipeline tests failed.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nThe evaluation pipeline is working correctly.")
    print("You can now run: python main.py --conversation <file> --context <file>")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()