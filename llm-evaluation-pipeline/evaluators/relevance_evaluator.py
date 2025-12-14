"""
Relevance & Completeness Evaluator

Evaluates how well the AI response addresses the user query:
- Semantic relevance using sentence transformers
- Completeness analysis
- Context utilization
"""

import numpy as np
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer, util
import logging


class RelevanceEvaluator:
    """Evaluates response relevance and completeness."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.logger = logging.getLogger("RelevanceEvaluator")
        
        # Load semantic similarity model (fast and effective)
        model_name = config.RELEVANCE_MODEL
        self.logger.info(f"Loading relevance model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.logger.info("Relevance evaluator initialized")
    
    def evaluate(
        self, 
        query: str, 
        response: str, 
        context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate response relevance and completeness.
        
        Args:
            query: User query/question
            response: AI-generated response
            context: Retrieved context from vector DB
            
        Returns:
            Dictionary with relevance metrics
        """
        try:
            # 1. Query-Response Relevance
            query_response_similarity = self._compute_similarity(query, response)
            
            # 2. Context-Response Relevance
            context_texts = [ctx.get('text', ctx.get('content', '')) 
                           for ctx in context]
            context_relevance = self._compute_context_relevance(
                response, 
                context_texts
            )
            
            # 3. Completeness Check
            completeness = self._assess_completeness(query, response)
            
            # 4. Key Term Coverage
            key_term_coverage = self._check_key_term_coverage(query, response)
            
            # Overall relevance score (weighted combination)
            relevance_score = (
                0.35 * query_response_similarity +
                0.25 * context_relevance +
                0.25 * completeness +
                0.15 * key_term_coverage
            )
            
            return {
                "relevance_score": round(relevance_score, 3),
                "query_response_similarity": round(query_response_similarity, 3),
                "context_relevance": round(context_relevance, 3),
                "completeness": round(completeness, 3),
                "key_term_coverage": round(key_term_coverage, 3),
                "assessment": self._get_assessment(relevance_score)
            }
            
        except Exception as e:
            self.logger.error(f"Relevance evaluation failed: {str(e)}")
            return self._get_error_result()
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        
        embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return float(similarity.item())
    
    def _compute_context_relevance(
        self, 
        response: str, 
        context_texts: List[str]
    ) -> float:
        """
        Measure how well response utilizes provided context.
        """
        if not context_texts or not response:
            return 0.0
        
        # Encode response
        response_embedding = self.model.encode(response, convert_to_tensor=True)
        
        # Encode all context chunks
        context_embeddings = self.model.encode(
            context_texts, 
            convert_to_tensor=True
        )
        
        # Find maximum similarity with any context chunk
        similarities = util.pytorch_cos_sim(
            response_embedding, 
            context_embeddings
        )[0]
        
        # Return average of top-k similarities (use top 3)
        top_k = min(3, len(similarities))
        top_similarities = similarities.topk(top_k).values
        
        return float(top_similarities.mean().item())
    
    def _assess_completeness(self, query: str, response: str) -> float:
        """
        Assess if response completely addresses the query.
        
        Uses heuristics:
        - Response length appropriateness
        - Presence of conclusive statements
        - Answer structure
        """
        if not response:
            return 0.0
        
        completeness_score = 0.0
        
        # Check 1: Response is not too short
        response_words = response.split()
        query_words = query.split()
        
        if len(response_words) < 5:
            completeness_score += 0.2
        elif len(response_words) < len(query_words):
            completeness_score += 0.4
        else:
            completeness_score += 0.8
        
        # Check 2: Contains definitive language
        definitive_patterns = [
            'the answer is', 'based on', 'according to',
            'this means', 'therefore', 'in summary',
            'to conclude', 'the result is'
        ]
        response_lower = response.lower()
        if any(pattern in response_lower for pattern in definitive_patterns):
            completeness_score += 0.2
        
        return min(1.0, completeness_score)
    
    def _check_key_term_coverage(self, query: str, response: str) -> float:
        """
        Check if key terms from query appear in response.
        """
        if not query or not response:
            return 0.0
        
        # Extract potential key terms (simple approach)
        # In production, use NER or keyword extraction
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are',
            'was', 'were', 'what', 'where', 'when', 'why', 'how'
        }
        
        query_words = [
            w for w in query_lower.split() 
            if w not in stop_words and len(w) > 2
        ]
        
        if not query_words:
            return 1.0
        
        # Count how many key terms appear in response
        covered_terms = sum(
            1 for word in query_words if word in response_lower
        )
        
        coverage_ratio = covered_terms / len(query_words)
        return coverage_ratio
    
    def _get_assessment(self, score: float) -> str:
        """Convert score to qualitative assessment."""
        if score >= 0.8:
            return "Excellent - Highly relevant and complete"
        elif score >= 0.6:
            return "Good - Mostly relevant with minor gaps"
        elif score >= 0.4:
            return "Fair - Partially relevant, needs improvement"
        else:
            return "Poor - Low relevance or incomplete"
    
    def _get_error_result(self) -> Dict[str, Any]:
        """Return error result structure."""
        return {
            "relevance_score": 0.0,
            "query_response_similarity": 0.0,
            "context_relevance": 0.0,
            "completeness": 0.0,
            "key_term_coverage": 0.0,
            "assessment": "Error - Evaluation failed"
        }