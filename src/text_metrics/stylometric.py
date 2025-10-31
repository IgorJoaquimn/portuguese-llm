"""
Stylometric analyzer using spaCy for Portuguese text analysis.
Provides Flesch Reading Ease, POS frequencies, NER, and lexical metrics.
"""

import logging
import spacy
import numpy as np
from typing import Dict, Optional
from collections import Counter
from .base import BaseAnalyzer

logger = logging.getLogger(__name__)


class StylometricAnalyzer(BaseAnalyzer):
    """
    Comprehensive stylometric analyzer for Portuguese text.
    Uses spaCy for POS tagging, NER, and lexical analysis.
    """
    
    def __init__(self, model_name: str = "pt_core_news_sm"):
        """
        Initialize the stylometric analyzer.
        
        Args:
            model_name: spaCy Portuguese model name
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except IOError:
            logger.warning(f"Model {model_name} not found. Trying pt_core_news_sm...")
            try:
                self.nlp = spacy.load("pt_core_news_sm")
                logger.info("Loaded spaCy model: pt_core_news_sm")
            except IOError:
                logger.error("No Portuguese spaCy model found. Install with: python -m spacy download pt_core_news_sm")
                raise
    
    def analyze(self, text: str) -> Dict:
        """
        Perform comprehensive stylometric analysis on text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with all stylometric metrics
        """
        if not text or not text.strip():
            return self._get_stylometric_metrics()
        
        try:
            doc = self.nlp(text)
            metrics = {}
            
            # POS frequencies
            metrics.update(self._calculate_pos_frequencies(doc))
            
            # Named entities
            metrics.update(self._extract_named_entities(doc))
            
            # Lexical metrics
            metrics.update(self._calculate_lexical_metrics(doc))
            
            # Flesch Reading Ease
            metrics["flesch_reading_ease"] = self._calculate_flesch_reading_ease(text, doc)
            
            # Syntactic metrics
            metrics.update(self._calculate_syntactic_metrics(doc))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in comprehensive stylometric analysis: {e}")
            return self._get_stylometric_metrics()
    
    def _calculate_pos_frequencies(self, doc) -> Dict:
        """Calculate POS tag frequencies."""
        metrics = {}
        
        tokens = [token for token in doc if not token.is_punct and not token.is_space]
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return {
                "noun_freq": 0.0,
                "verb_freq": 0.0,
                "adj_freq": 0.0,
                "adv_freq": 0.0
            }
        
        pos_counts = Counter(token.pos_ for token in tokens)
        
        metrics["noun_freq"] = round((pos_counts.get('NOUN', 0) / total_tokens) * 100, 2)
        metrics["verb_freq"] = round((pos_counts.get('VERB', 0) / total_tokens) * 100, 2)
        metrics["adj_freq"] = round((pos_counts.get('ADJ', 0) / total_tokens) * 100, 2)
        metrics["adv_freq"] = round((pos_counts.get('ADV', 0) / total_tokens) * 100, 2)
        
        return metrics
    
    def _extract_named_entities(self, doc) -> Dict:
        """Extract named entities using spaCy NER."""
        entity_counts = {"per_count": 0, "org_count": 0, "loc_count": 0}
        
        for ent in doc.ents:
            if ent.label_ in ["PER", "PERSON"]:
                entity_counts["per_count"] += 1
            elif ent.label_ == "ORG":
                entity_counts["org_count"] += 1
            elif ent.label_ in ["LOC", "GPE", "PLACE"]:
                entity_counts["loc_count"] += 1
        
        return entity_counts
    
    def _calculate_lexical_metrics(self, doc) -> Dict:
        """Calculate lexical metrics."""
        metrics = {}
        
        tokens = [token for token in doc if not token.is_punct and not token.is_space]
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return {
                "avg_word_length": 0.0,
                "long_words_ratio": 0.0
            }
        
        word_lengths = [len(token.text) for token in tokens]
        metrics["avg_word_length"] = round(np.mean(word_lengths), 2)
        
        long_words = [t for t in tokens if len(t.text) > 6]
        metrics["long_words_ratio"] = round((len(long_words) / total_tokens) * 100, 2)
        
        return metrics
    
    def _calculate_flesch_reading_ease(self, text: str, doc) -> float:
        """
        Calculate Flesch Reading Ease score for Portuguese.
        
        Formula: FRE = 248.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        
        Args:
            text: Original text
            doc: Parsed spaCy Doc object
            
        Returns:
            Flesch Reading Ease score
        """
        try:
            sentences = list(doc.sents)
            total_sentences = len(sentences)
            
            if total_sentences == 0:
                return 0.0
            
            words = [token for token in doc if not token.is_punct and not token.is_space]
            total_words = len(words)
            
            if total_words == 0:
                return 0.0
            
            # Estimate syllables for Portuguese
            total_syllables = sum(self._count_syllables_pt(token.text) for token in words)
            
            avg_sentence_length = total_words / total_sentences
            avg_syllables_per_word = total_syllables / total_words
            
            # Portuguese Flesch formula (adapted)
            fre_score = 248.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            return round(fre_score, 2)
            
        except Exception as e:
            logger.warning(f"Error calculating Flesch score: {e}")
            return 0.0
    
    def _calculate_syntactic_metrics(self, doc) -> Dict:
        """Calculate syntactic metrics."""
        metrics = {}
        
        sentences = list(doc.sents)
        
        # Sentence length variance
        sentence_lengths = []
        for sent in sentences:
            sent_words = [token for token in sent if not token.is_punct and not token.is_space]
            sentence_lengths.append(len(sent_words))
        
        metrics["sentence_length_variance"] = round(np.var(sentence_lengths), 2) if len(sentence_lengths) > 1 else 0.0
        
        # Punctuation ratio
        total_tokens = len([t for t in doc if not t.is_space])
        punct_count = len([t for t in doc if t.is_punct])
        metrics["punctuation_ratio"] = round((punct_count / total_tokens) * 100, 2) if total_tokens > 0 else 0.0
        
        return metrics
    
    def _get_stylometric_metrics(self) -> Dict:
        """Return empty stylometric metrics."""
        return {
            "noun_freq": 0.0,
            "verb_freq": 0.0,
            "adj_freq": 0.0,
            "adv_freq": 0.0,
            "per_count": 0,
            "org_count": 0,
            "loc_count": 0,
            "avg_word_length": 0.0,
            "long_words_ratio": 0.0,
            "flesch_reading_ease": 0.0,
            "sentence_length_variance": 0.0,
            "punctuation_ratio": 0.0,
        }
