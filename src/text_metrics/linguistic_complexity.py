"""
Linguistic complexity analyzer using UDPipe dependency parsing.
Based on LinguisticComplexityAnalyzer methodology.
"""

import logging
import numpy as np
from typing import Dict, Optional
from collections import Counter
from .base import BaseAnalyzer

logger = logging.getLogger(__name__)


class LinguisticComplexityAnalyzer(BaseAnalyzer):
    """Analyzes linguistic complexity using UDPipe-parsed dependency structures."""
    
    def analyze(self, sentences) -> Dict:
        """
        Extract linguistic complexity metrics from UDPipe parsed sentences.
        
        Args:
            sentences: List of CoNLL-U parsed sentences
            
        Returns:
            Dictionary with linguistic complexity metrics
        """
        if not sentences:
            return self._get_base_metrics()
        
        try:
            return self._analyze_sentences(sentences)
        except Exception as e:
            logger.error(f"Error analyzing linguistic complexity: {e}")
            return self._get_base_metrics()
    
    def _analyze_sentences(self, sentences) -> Dict:
        """Internal method to analyze sentences."""
        metrics = {}
        
        # Define helper functions
        def is_clause(token):
            for deprel in ["csubj", "ccomp", "xcomp", "advcl", "acl"]:
                if deprel in token["deprel"]:
                    return True
            return False
        
        def is_dependent_clause(token):
            for deprel in ["advcl", "acl"]:
                if deprel in token["deprel"]:
                    return True
            return False
        
        def is_coordination(token):
            for deprel in ["conj", "cc"]:
                if deprel in token["deprel"]:
                    return True
            return False
        
        def is_lexical(token):
            return token["upos"] in ["NOUN", "ADJ", "VERB"]
        
        def profundidade_maxima(no):
            if not no.children:
                return 0
            filho_mais_fundo = max([profundidade_maxima(child) for child in no.children])
            return 1 + filho_mais_fundo
        
        # Count clauses and coordinations
        total_clauses = sum(len([t for t in sent if is_clause(t)]) for sent in sentences)
        total_dependent_clauses = sum(len([t for t in sent if is_dependent_clause(t)]) for sent in sentences)
        total_coordinated = sum(len([t for t in sent if is_coordination(t)]) for sent in sentences)
        
        # Count tokens (exclude punctuation)
        total_tokens = sum(len([t for t in sent if t["deprel"] != "punct"]) for sent in sentences)
        total_sentences = len(sentences)
        
        # Tree depths
        depths = [profundidade_maxima(sent.to_tree()) for sent in sentences]
        
        # TTR (Type-Token Ratio)
        tokens = [t["form"] for sent in sentences for t in sent if t["deprel"] != "punct"]
        types = set(tokens)
        ttr = len(types) / len(tokens) if tokens else 0.0
        
        # Lexical density
        lexical_count = sum(len([t for t in sent if is_lexical(t)]) for sent in sentences)
        lexical_density = lexical_count / total_tokens if total_tokens > 0 else 0.0
        
        # Complexity metrics
        metrics["MLC"] = round(total_tokens / total_clauses if total_clauses > 0 else 0, 2)
        metrics["MLS"] = round(total_tokens / total_sentences if total_sentences > 0 else 0, 2)
        metrics["DCC"] = round(total_dependent_clauses / total_clauses if total_clauses > 0 else 0, 2)
        metrics["CPC"] = round(total_coordinated / total_clauses if total_clauses > 0 else 0, 2)
        metrics["profundidade_media"] = round(np.mean(depths), 2) if depths else 0.0
        metrics["profundidade_max"] = int(max(depths)) if depths else 0
        metrics["ttr"] = round(ttr, 4)
        metrics["lexical_density"] = round(lexical_density, 4)
        metrics["token_quantity"] = total_tokens
        
        return metrics
    
    def _get_base_metrics(self) -> Dict:
        """Return base metrics dictionary."""
        return {
            "MLC": 0.0,
            "MLS": 0.0,
            "DCC": 0.0,
            "CPC": 0.0,
            "profundidade_media": 0.0,
            "profundidade_max": 0,
            "ttr": 0.0,
            "lexical_density": 0.0,
            "token_quantity": 0,
        }
