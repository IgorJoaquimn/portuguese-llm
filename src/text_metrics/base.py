"""
Base analyzer class for text metrics.
"""

import logging
from typing import Dict, Optional
import numpy as np
import re

logger = logging.getLogger(__name__)


class BaseAnalyzer:
    """Base class for all text analyzers."""
    
    @staticmethod
    def _count_syllables_pt(word: str) -> int:
        """
        Count syllables in a Portuguese word using basic rules.
        
        Args:
            word: Portuguese word
            
        Returns:
            Number of syllables
        """
        word = word.lower().strip()
        if not word:
            return 0
        
        vowels = "aeiouáéíóúàèìòùâêîôûãõ"
        word = re.sub(r'[^a-záéíóúàèìòùâêîôûãõ]', '', word)
        
        if not word:
            return 1
        
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Handle common diphthongs
        diphthongs = ['ai', 'au', 'ei', 'eu', 'oi', 'ou', 'ui']
        for diphthong in diphthongs:
            syllable_count -= word.count(diphthong)
        
        return max(1, syllable_count)
    
    @staticmethod
    def _get_empty_metrics() -> Dict:
        """Return empty metrics dictionary with all expected keys."""
        return {
            # UDPipe-based metrics
            "MLC": 0.0,
            "MLS": 0.0,
            "DCC": 0.0,
            "CPC": 0.0,
            "profundidade_media": 0.0,
            "profundidade_max": 0,
            "ttr": 0.0,
            "lexical_density": 0.0,
            "token_quantity": 0,
            # POS frequencies
            "noun_freq": 0.0,
            "verb_freq": 0.0,
            "adj_freq": 0.0,
            "adv_freq": 0.0,
            # Lexical metrics
            "avg_word_length": 0.0,
            "long_words_ratio": 0.0,
            # Syntactic metrics
            "sentence_length_variance": 0.0,
            "punctuation_ratio": 0.0,
            "flesch_reading_ease": 0.0,
            # NER metrics
            "per_count": 0,
            "org_count": 0,
            "loc_count": 0,
            # Lemmatization
            "lemmas": "",
        }
