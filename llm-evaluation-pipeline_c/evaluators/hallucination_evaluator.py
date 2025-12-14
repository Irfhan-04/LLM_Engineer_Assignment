"""
Hallucination & Factual Accuracy Evaluator

Detects when AI generates information not supported by context:
- Entailment checking using NLI models
- Fact extraction and verification
- Citation analysis
"""

import logging
from typing import Dict, List, Any
import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util


class HallucinationEvaluator:
    """Evaluates response for hallucinations and factual accuracy."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.logger = logging.getLogger("HallucinationEvaluator")
        
        # Load NLI model for entailment checking
        nli_model_name = config.HALLUCINATION_MODEL
        self.logger.info(f"Loading NLI model: {nli_model_name}")
        self.nli_pipeline = pipeline(
            "text-classification",
            model=nli_model_name,
            device=-1  # CPU (-1) or GPU (0)
        )
        
        # Load embedding model for semantic similarity
        self.embedding_model = SentenceTransformer(config.RELEVANCE_MODEL)
        
        self.logger.info("Hallucination evaluator initialized")
    
    def evaluate(
        self, 
        response: str, 
        context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate response for hallucinations.
        
        Args:
            response: AI-generated response
            context: Retrieved context from vector DB
            
        Returns:
            Dictionary with hallucination metrics
        """
        try:
            # Extract context texts
            context_texts = [
                ctx.get('text', ctx.get('content', '')) 
                for ctx in context
            ]
            combined_context = " ".join(context_texts)
            
            if not combined_context:
                return {
                    "hallucination_risk": 1.0,
                    "factual_accuracy": 0.0,
                    "entailment_score": 0.0,
                    "assessment": "No context provided - cannot verify"
                }
            
            # 1. Sentence-level entailment checking
            entailment_score = self._check_entailment(
                response, 
                combined_context
            )
            
            # 2. Fact extraction and grounding
            grounding_score = self._check_fact_grounding(
                response, 
                context_texts
            )
            
            # 3. Detect unsupported claims
            unsupported_ratio = self._detect_unsupported_claims(
                response, 
                combined_context
            )
            
            # Calculate overall hallucination risk
            # Lower is better (less hallucination)
            hallucination_risk = (
                0.4 * (1 - entailment_score) +
                0.4 * (1 - grounding_score) +
                0.2 * unsupported_ratio
            )
            
            factual_accuracy = 1.0 - hallucination_risk
            
            return {
                "hallucination_risk": round(hallucination_risk, 3),
                "factual_accuracy": round(factual_accuracy, 3),
                "entailment_score": round(entailment_score, 3),
                "grounding_score": round(grounding_score, 3),
                "unsupported_claims_ratio": round(unsupported_ratio, 3),
                "assessment": self._get_assessment(hallucination_risk)
            }
            
        except Exception as e:
            self.logger.error(f"Hallucination evaluation failed: {str(e)}")
            return self._get_error_result()
    
    def _check_entailment(self, response: str, context: str) -> float:
        """
        Check if response is entailed by context using NLI.
        
        Uses Natural Language Inference to determine if the
        context logically supports the response.
        """
        if not response or not context:
            return 0.0
        
        # Split response into sentences for granular checking
        sentences = self._split_sentences(response)
        
        if not sentences:
            return 0.0
        
        entailment_scores = []
        
        for sentence in sentences:
            # Truncate if too long
            sentence = sentence[:512]
            context_truncated = context[:512]
            
            try:
                # NLI expects premise (context) + hypothesis (claim)
                result = self.nli_pipeline(
                    f"{context_truncated} [SEP] {sentence}"
                )[0]
                
                # Extract entailment probability
                if result['label'] == 'ENTAILMENT':
                    entailment_scores.append(result['score'])
                elif result['label'] == 'NEUTRAL':
                    entailment_scores.append(0.5)  # Neutral cases
                else:  # CONTRADICTION
                    entailment_scores.append(0.0)
                    
            except Exception as e:
                self.logger.warning(f"NLI failed for sentence: {str(e)}")
                entailment_scores.append(0.5)  # Default to neutral
        
        # Return average entailment
        return sum(entailment_scores) / len(entailment_scores)
    
    def _check_fact_grounding(
        self, 
        response: str, 
        context_texts: List[str]
    ) -> float:
        """
        Check if facts in response are grounded in context.
        
        Uses semantic similarity to verify claims.
        """
        if not response or not context_texts:
            return 0.0
        
        # Extract potential factual claims (sentences with numbers, names, etc.)
        sentences = self._split_sentences(response)
        factual_sentences = [
            s for s in sentences 
            if self._is_factual_claim(s)
        ]
        
        if not factual_sentences:
            # No factual claims to verify
            return 1.0
        
        # Encode response sentences and context
        response_embeddings = self.embedding_model.encode(
            factual_sentences, 
            convert_to_tensor=True
        )
        context_embeddings = self.embedding_model.encode(
            context_texts, 
            convert_to_tensor=True
        )
        
        # For each factual claim, find best matching context
        grounding_scores = []
        for resp_emb in response_embeddings:
            similarities = util.pytorch_cos_sim(resp_emb, context_embeddings)[0]
            max_sim = similarities.max().item()
            grounding_scores.append(max_sim)
        
        # Average grounding score
        return sum(grounding_scores) / len(grounding_scores)
    
    def _detect_unsupported_claims(
        self, 
        response: str, 
        context: str
    ) -> float:
        """
        Detect ratio of unsupported claims.
        
        Uses pattern matching to find definitive statements
        and checks if they appear in context.
        """
        # Patterns indicating definitive claims
        claim_patterns = [
            r'\d+%',  # Percentages
            r'\$[\d,]+',  # Monetary values
            r'\d+ (years?|months?|days?)',  # Time periods
            r'(exactly|precisely|specifically) \w+',
            r'(always|never|must|cannot) \w+',
        ]
        
        response_lower = response.lower()
        context_lower = context.lower()
        
        unsupported_count = 0
        total_claims = 0
        
        for pattern in claim_patterns:
            matches = re.findall(pattern, response_lower)
            for match in matches:
                total_claims += 1
                if match not in context_lower:
                    unsupported_count += 1
        
        if total_claims == 0:
            return 0.0  # No specific claims made
        
        return unsupported_count / total_claims
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _is_factual_claim(self, sentence: str) -> bool:
        """
        Determine if sentence contains factual claims.
        
        Heuristic: contains numbers, proper nouns, or definitive language.
        """
        # Check for numbers
        if re.search(r'\d', sentence):
            return True
        
        # Check for capital letters (possible proper nouns)
        words = sentence.split()
        capitalized = sum(1 for w in words if w and w[0].isupper())
        if capitalized > 1:  # More than just first word
            return True
        
        # Check for definitive language
        definitive_words = [
            'is', 'are', 'was', 'were', 'has', 'have',
            'will', 'can', 'must', 'should'
        ]
        if any(word in sentence.lower().split() for word in definitive_words):
            return True
        
        return False
    
    def _get_assessment(self, risk: float) -> str:
        """Convert risk score to qualitative assessment."""
        if risk <= 0.2:
            return "Excellent - Fully grounded in context"
        elif risk <= 0.4:
            return "Good - Minor unsupported details"
        elif risk <= 0.6:
            return "Fair - Some hallucination detected"
        elif risk <= 0.8:
            return "Poor - Significant hallucination"
        else:
            return "Critical - Severe hallucination risk"
    
    def _get_error_result(self) -> Dict[str, Any]:
        """Return error result structure."""
        return {
            "hallucination_risk": 1.0,
            "factual_accuracy": 0.0,
            "entailment_score": 0.0,
            "grounding_score": 0.0,
            "unsupported_claims_ratio": 1.0,
            "assessment": "Error - Evaluation failed"
        }