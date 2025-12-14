"""
BeyondChats LLM Evaluation Pipeline
Main evaluation script for assessing LLM responses in real-time RAG systems.

Author: [Your Name]
Date: December 2024
"""

import json
import time
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime

from evaluators.relevance_evaluator import RelevanceEvaluator
from evaluators.hallucination_evaluator import HallucinationEvaluator
from evaluators.performance_evaluator import PerformanceEvaluator
from utils.json_parser import parse_conversation, parse_context
from utils.logger import setup_logger
from config import Config


class LLMEvaluationPipeline:
    """
    Main evaluation pipeline for LLM responses.
    
    Evaluates responses across three dimensions:
    1. Response Relevance & Completeness
    2. Hallucination / Factual Accuracy
    3. Latency & Costs
    """
    
    def __init__(self, config: Config):
        """Initialize the evaluation pipeline with configuration."""
        self.config = config
        self.logger = setup_logger("EvaluationPipeline")
        
        # Initialize evaluators
        self.relevance_evaluator = RelevanceEvaluator(config)
        self.hallucination_evaluator = HallucinationEvaluator(config)
        self.performance_evaluator = PerformanceEvaluator(config)
        
        self.logger.info("Evaluation pipeline initialized successfully")
    
    def evaluate_single_response(
        self, 
        conversation_data: Dict[str, Any], 
        context_data: Dict[str, Any],
        message_id: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single AI response.
        
        Args:
            conversation_data: JSON containing chat conversation
            context_data: JSON containing context vectors from vector DB
            message_id: Specific message ID to evaluate (optional)
            
        Returns:
            Dictionary containing evaluation results
        """
        start_time = time.time()
        
        try:
            # Parse inputs
            parsed_conv = parse_conversation(conversation_data)
            parsed_context = parse_context(context_data)
            
            # Extract the AI response to evaluate
            if message_id:
                ai_response = self._get_response_by_id(parsed_conv, message_id)
            else:
                ai_response = self._get_last_ai_response(parsed_conv)
            
            if not ai_response:
                raise ValueError("No AI response found to evaluate")
            
            # Get user query that prompted this response
            user_query = self._get_corresponding_query(parsed_conv, ai_response)
            
            # Perform evaluations
            relevance_score = self.relevance_evaluator.evaluate(
                query=user_query['content'],
                response=ai_response['content'],
                context=parsed_context
            )
            
            hallucination_score = self.hallucination_evaluator.evaluate(
                response=ai_response['content'],
                context=parsed_context
            )
            
            performance_metrics = self.performance_evaluator.evaluate(
                response=ai_response['content'],
                start_time=start_time,
                context_texts=[ctx.get('text', ctx.get('content', '')) 
                             for ctx in parsed_context],
                model_name=parsed_conv.get('metadata', {}).get('model'),
                metadata=ai_response  # Pass full message for latency extraction
            )
            
            # Compile results
            evaluation_result = {
                "message_id": ai_response.get('id', 'unknown'),
                "timestamp": datetime.now().isoformat(),
                "query": user_query['content'],
                "response": ai_response['content'],
                "relevance": relevance_score,
                "hallucination": hallucination_score,
                "performance": performance_metrics,
                "overall_score": self._calculate_overall_score(
                    relevance_score, 
                    hallucination_score
                ),
                "evaluation_time_ms": (time.time() - start_time) * 1000
            }
            
            self.logger.info(
                f"Evaluation completed: Overall Score = "
                f"{evaluation_result['overall_score']:.2f}"
            )
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def evaluate_batch(
        self,
        conversation_files: List[str],
        context_files: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple conversations in batch.
        
        Args:
            conversation_files: List of paths to conversation JSON files
            context_files: List of paths to context JSON files
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for conv_file, ctx_file in zip(conversation_files, context_files):
            try:
                with open(conv_file, 'r') as f:
                    conversation_data = json.load(f)
                with open(ctx_file, 'r') as f:
                    context_data = json.load(f)
                
                result = self.evaluate_single_response(
                    conversation_data, 
                    context_data
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(
                    f"Failed to evaluate {conv_file}: {str(e)}"
                )
                continue
        
        return results
    
    def _get_last_ai_response(
        self, 
        conversation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract the last AI response from conversation."""
        messages = conversation.get('messages', [])
        for message in reversed(messages):
            if message.get('role') in ['assistant', 'ai', 'bot']:
                return message
        return None
    
    def _get_response_by_id(
        self, 
        conversation: Dict[str, Any], 
        message_id: str
    ) -> Dict[str, Any]:
        """Get a specific response by message ID."""
        messages = conversation.get('messages', [])
        for message in messages:
            if message.get('id') == message_id:
                return message
        return None
    
    def _get_corresponding_query(
        self, 
        conversation: Dict[str, Any], 
        ai_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get the user query that prompted the AI response."""
        messages = conversation.get('messages', [])
        ai_index = messages.index(ai_response)
        
        # Look for the most recent user message before this AI response
        for i in range(ai_index - 1, -1, -1):
            if messages[i].get('role') in ['user', 'human']:
                return messages[i]
        
        return {'content': '', 'role': 'user'}
    
    def _calculate_overall_score(
        self, 
        relevance_score: Dict[str, Any], 
        hallucination_score: Dict[str, Any]
    ) -> float:
        """
        Calculate overall quality score.
        
        Weighted average:
        - Relevance: 40%
        - Hallucination (inverted): 60% (higher is better when no hallucination)
        """
        relevance = relevance_score.get('relevance_score', 0.0)
        # Invert hallucination score (lower hallucination = higher quality)
        factual_accuracy = 1.0 - hallucination_score.get('hallucination_risk', 0.0)
        
        overall = (0.4 * relevance) + (0.6 * factual_accuracy)
        return round(overall, 3)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate LLM responses in real-time'
    )
    parser.add_argument(
        '--conversation',
        type=str,
        required=True,
        help='Path to conversation JSON file'
    )
    parser.add_argument(
        '--context',
        type=str,
        required=True,
        help='Path to context vectors JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--message-id',
        type=str,
        help='Specific message ID to evaluate'
    )
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    
    # Create pipeline
    pipeline = LLMEvaluationPipeline(config)
    
    # Load input files
    with open(args.conversation, 'r') as f:
        conversation_data = json.load(f)
    
    with open(args.context, 'r') as f:
        context_data = json.load(f)
    
    # Run evaluation
    result = pipeline.evaluate_single_response(
        conversation_data,
        context_data,
        message_id=args.message_id
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Score: {result['overall_score']:.3f}")
    print(f"\nRelevance Score: {result['relevance']['relevance_score']:.3f}")
    print(f"Completeness: {result['relevance']['completeness']:.3f}")
    print(f"\nHallucination Risk: {result['hallucination']['hallucination_risk']:.3f}")
    print(f"Factual Accuracy: {result['hallucination']['factual_accuracy']:.3f}")
    print(f"\nPerformance:")
    if result['performance']['generation_latency_ms']:
        print(f"  Generation Latency: {result['performance']['generation_latency_ms']:.2f}ms")
    else:
        print(f"  Generation Latency: Not available in metadata")
    print(f"  Estimated Cost: ${result['performance']['estimated_generation_cost_usd']:.6f}")
    print(f"  Evaluation Time: {result['performance']['evaluation_latency_ms']:.2f}ms")
    print(f"\nResults saved to: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()