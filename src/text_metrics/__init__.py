"""
Unified Text Metrics Analysis Module

This module provides a single entry point for comprehensive text analysis including:
- Lemmatization (via UDPipe)
- Linguistic complexity metrics (via UDPipe dependency parsing)
- Stylometric analysis (via spaCy)

Usage:
    analyzer = TextMetricsAnalyzer()
    metrics = analyzer.analyze(
        text="Your text here",
        udpipe_output=None,  # Optional pre-parsed UDPipe output
        include_ner=True
    )
"""

import logging
from typing import Dict, Optional

from .lemmatizer import LemmatizerAnalyzer
from .linguistic_complexity import LinguisticComplexityAnalyzer
from .stylometric import StylometricAnalyzer
from .udpipe_client import UDPipeClient
from .base import BaseAnalyzer

logger = logging.getLogger(__name__)


class TextMetricsAnalyzer(BaseAnalyzer):
    """
    Unified text metrics analyzer combining all analysis types.
    
    This class orchestrates lemmatization, linguistic complexity analysis,
    and stylometric analysis to provide comprehensive text metrics.
    """
    
    def __init__(self, udpipe_enabled: bool = True, spacy_model: str = "pt_core_news_sm"):
        """
        Initialize the unified text metrics analyzer.
        
        Args:
            udpipe_enabled: Whether to use UDPipe (requires external API)
            spacy_model: spaCy model to use for stylometric analysis
        """
        self.udpipe_enabled = udpipe_enabled
        
        if udpipe_enabled:
            self.udpipe_client = UDPipeClient()
            self.linguistic_complexity = LinguisticComplexityAnalyzer()
            self.lemmatizer = LemmatizerAnalyzer()
        
        self.stylometric = StylometricAnalyzer(model_name=spacy_model)
        
        logger.info(f"TextMetricsAnalyzer initialized (UDPipe: {udpipe_enabled})")
    
    def analyze(
        self,
        text: str,
        udpipe_output: Optional[str] = None,
        include_ner: bool = True,
        include_lemmatization: bool = True
    ) -> Dict:
        """
        Comprehensive text analysis in a single call.
        
        Args:
            text: Raw text to analyze
            udpipe_output: Optional pre-parsed UDPipe CoNLL-U output
            include_ner: Whether to extract named entities
            include_lemmatization: Whether to extract lemmas
            
        Returns:
            Dictionary with all computed metrics
        """
        if not text or not text.strip():
            return self._get_empty_metrics()
        
        metrics = {}
        
        try:
            # UDPipe-based analysis (if enabled)
            if self.udpipe_enabled:
                # Get UDPipe output if not provided
                if udpipe_output is None:
                    udpipe_output = self.udpipe_client.generate_one_response(text)
                
                # Parse UDPipe output
                sentences = self.udpipe_client.parse_response(udpipe_output)
                
                # Linguistic complexity metrics from UDPipe
                complexity_metrics = self.linguistic_complexity.analyze(sentences)
                metrics.update(complexity_metrics)
                
                # Lemmatization from UDPipe
                if include_lemmatization:
                    lemma_metrics = self.lemmatizer.analyze(text, udpipe_output)
                    metrics.update(lemma_metrics)
                else:
                    metrics["lemmas"] = ""
            else:
                # Fallback when UDPipe not available
                metrics["lemmas"] = ""
                # Set default values for UDPipe metrics
                metrics.update({
                    "MLC": 0.0,
                    "MLS": 0.0,
                    "DCC": 0.0,
                    "CPC": 0.0,
                    "profundidade_media": 0.0,
                    "profundidade_max": 0,
                    "ttr": 0.0,
                    "lexical_density": 0.0,
                    "token_quantity": 0,
                })
            
            # Stylometric analysis (always available)
            stylometric_metrics = self.stylometric.analyze(text)
            
            # Only include NER if requested
            if not include_ner:
                stylometric_metrics["per_count"] = 0
                stylometric_metrics["org_count"] = 0
                stylometric_metrics["loc_count"] = 0
            
            metrics.update(stylometric_metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during comprehensive text analysis: {e}")
            return self._get_empty_metrics()


__all__ = [
    'TextMetricsAnalyzer',
    'LemmatizerAnalyzer',
    'LinguisticComplexityAnalyzer',
    'StylometricAnalyzer',
    'UDPipeClient',
]
