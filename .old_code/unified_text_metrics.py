"""
Unified Text Metrics Analysis

Combines UDPipe linguistic analysis with stylometric metrics into a single class.
Computes all text analysis metrics in one pass:
  - POS frequencies (from UDPipe)
  - Flesch Reading Ease (Portuguese adapted)
  - Type-Token Ratio (TTR)
  - Syntactic complexity (tree depth, clauses, etc.)
  - Named entities (from spaCy NER)
  - Lexical metrics (word length, long word ratio, etc.)
"""

import numpy as np
import spacy
import re
import logging
from typing import Dict, Optional, List
from collections import Counter
from conllu import parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedTextMetrics:
    """
    Single unified class for comprehensive text analysis.
    
    Accepts either:
    1. Pre-parsed UDPipe output (CoNLL-U format) + raw text
    2. Raw text only (parses with spaCy)
    
    Returns all metrics in a single dictionary.
    """
    
    def __init__(self, udpipe_available: bool = True, spacy_model: str = "pt_core_news_sm"):
        """
        Initialize metrics analyzer.
        
        Args:
            udpipe_available: Whether UDPipe data will be provided
            spacy_model: spaCy model name for NER (lightweight model recommended)
        """
        self.udpipe_available = udpipe_available
        
        # Load spaCy for NER only (using lightweight model)
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except IOError:
            logger.warning(f"Model {spacy_model} not found. Trying pt_core_news_sm...")
            try:
                self.nlp = spacy.load("pt_core_news_sm")
                logger.info("Loaded spaCy model: pt_core_news_sm")
            except IOError:
                logger.error("No Portuguese spaCy model found. Install with: python -m spacy download pt_core_news_sm")
                raise
    
    def analyze(self, text: str, udpipe_output: Optional[str] = None, include_ner: bool = True) -> Dict:
        """
        Comprehensive text analysis in a single call.
        
        Args:
            text: Raw text to analyze
            udpipe_output: Optional pre-parsed UDPipe CoNLL-U output
            include_ner: Whether to extract named entities (slower, can disable if not needed)
            
        Returns:
            Dictionary with all computed metrics
        """
        if not text or not text.strip():
            return self._get_empty_metrics()
        
        metrics = {}
        
        try:
            # Parse with UDPipe if available
            if udpipe_output:
                sentences = parse(udpipe_output)
                metrics.update(self._analyze_with_udpipe(text, sentences))
                
                # Add NER metrics (from spaCy) only if requested
                if include_ner:
                    doc = self.nlp(text)
                    metrics.update(self._extract_named_entities(doc))
                else:
                    # Add empty NER metrics
                    metrics.update({"per_count": 0, "org_count": 0, "loc_count": 0})
            else:
                # Fallback to spaCy only (if UDPipe not available)
                doc = self.nlp(text)
                metrics.update(self._analyze_with_spacy(text, doc))
                
                # Add NER metrics from same doc (already parsed)
                if include_ner:
                    metrics.update(self._extract_named_entities(doc))
                else:
                    metrics.update({"per_count": 0, "org_count": 0, "loc_count": 0})
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return self._get_empty_metrics()
    
    def _analyze_with_udpipe(self, text: str, sentences) -> Dict:
        """
        Extract metrics from pre-parsed UDPipe sentences (CoNLL-U format).
        
        Args:
            text: Original text (for spaCy sentence segmentation fallback)
            sentences: Parsed CoNLL-U sentences
            
        Returns:
            Dictionary with UDPipe-based metrics
        """
        metrics = {}
        
        # Basic linguistic metrics from LinguisticComplexityAnalyzer
        complexity_metrics = self._analyze_linguistic_complexity(sentences)
        metrics.update(complexity_metrics)
        
        # POS frequencies
        pos_metrics = self._calculate_pos_frequencies_udpipe(sentences)
        metrics.update(pos_metrics)
        
        # Lexical metrics
        lexical_metrics = self._calculate_lexical_metrics_udpipe(sentences)
        metrics.update(lexical_metrics)
        
        # Flesch Reading Ease
        flesch_score = self._calculate_flesch_udpipe(text, sentences)
        metrics["flesch_reading_ease"] = flesch_score
        
        # Syntactic structure metrics
        syntax_metrics = self._calculate_syntax_metrics_udpipe(sentences)
        metrics.update(syntax_metrics)
        
        return metrics
    
    def _analyze_with_spacy(self, text: str, doc) -> Dict:
        """
        Fallback analysis using only spaCy (when UDPipe not available).
        
        Args:
            text: Original text
            doc: Parsed spaCy Doc object
            
        Returns:
            Dictionary with spaCy-based metrics
        """
        metrics = {}
        
        # POS frequencies
        tokens = [token for token in doc if not token.is_punct and not token.is_space]
        total_tokens = len(tokens)
        
        if total_tokens > 0:
            pos_counts = Counter(token.pos_ for token in tokens)
            metrics["noun_freq"] = round((pos_counts.get('NOUN', 0) / total_tokens) * 100, 2)
            metrics["verb_freq"] = round((pos_counts.get('VERB', 0) / total_tokens) * 100, 2)
            metrics["adj_freq"] = round((pos_counts.get('ADJ', 0) / total_tokens) * 100, 2)
            metrics["adv_freq"] = round((pos_counts.get('ADV', 0) / total_tokens) * 100, 2)
        
        # Lexical metrics
        metrics["avg_word_length"] = round(np.mean([len(token.text) for token in tokens]), 2) if tokens else 0.0
        long_words = [t for t in tokens if len(t.text) > 6]
        metrics["long_words_ratio"] = round((len(long_words) / total_tokens) * 100, 2) if tokens else 0.0
        metrics["ttr"] = self._calculate_ttr_spacy(tokens)
        
        # Flesch Reading Ease (spaCy version)
        sentences = list(doc.sents)
        flesch = self._calculate_flesch_spacy(tokens, sentences)
        metrics["flesch_reading_ease"] = flesch
        
        # Sentence variance
        sentence_lengths = [len([t for t in sent if not t.is_punct]) for sent in sentences]
        metrics["sentence_length_variance"] = round(np.var(sentence_lengths), 2) if len(sentence_lengths) > 1 else 0.0
        
        # Punctuation ratio
        total_tokens_with_punct = len([t for t in doc if not t.is_space])
        punct_count = len([t for t in doc if t.is_punct])
        metrics["punctuation_ratio"] = round((punct_count / total_tokens_with_punct) * 100, 2) if total_tokens_with_punct > 0 else 0.0
        
        return metrics
    
    def _analyze_linguistic_complexity(self, sentences) -> Dict:
        """
        Extract linguistic complexity metrics from UDPipe (dependency parsing).
        Based on LinguisticComplexityAnalyzer methodology.
        """
        metrics = {}
        
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
        
        def profundidade_maxima(no):
            if not no.children:
                return 0
            filho_mais_fundo = max([profundidade_maxima(child) for child in no.children])
            return 1 + filho_mais_fundo
        
        # Count clauses and coordinations
        total_clauses = sum(len([t for t in sent if is_clause(t)]) for sent in sentences)
        total_dependent_clauses = sum(len([t for t in sent if is_dependent_clause(t)]) for sent in sentences)
        total_coordinated = sum(len([t for t in sent if is_coordination(t)]) for sent in sentences)
        
        # Count tokens
        total_tokens = sum(len([t for t in sent if t["deprel"] != "punct"]) for sent in sentences)
        total_sentences = len(sentences)
        
        # Tree depths
        depths = [profundidade_maxima(sent.to_tree()) for sent in sentences]
        
        # TTR (Type-Token Ratio)
        tokens = [t["form"] for sent in sentences for t in sent if t["deprel"] != "punct"]
        types = set(tokens)
        ttr = len(types) / len(tokens) if tokens else 0.0
        
        # Lexical density
        def is_lexical(token):
            return token["upos"] in ["NOUN", "ADJ", "VERB"]
        
        lexical_count = sum(len([t for t in sent if is_lexical(t)]) for sent in sentences)
        lexical_density = lexical_count / total_tokens if total_tokens > 0 else 0.0
        
        # Complexity metrics
        metrics["MLC"] = round(total_tokens / total_clauses if total_clauses > 0 else 0, 2)  # Mean length of clauses
        metrics["MLS"] = round(total_tokens / total_sentences if total_sentences > 0 else 0, 2)  # Mean length of sentences
        metrics["DCC"] = round(total_dependent_clauses / total_clauses if total_clauses > 0 else 0, 2)  # Dependent clauses per clause
        metrics["CPC"] = round(total_coordinated / total_clauses if total_clauses > 0 else 0, 2)  # Coordinated phrases per clause
        metrics["profundidade_media"] = round(np.mean(depths), 2) if depths else 0.0
        metrics["profundidade_max"] = int(max(depths)) if depths else 0
        metrics["ttr"] = round(ttr, 4)
        metrics["lexical_density"] = round(lexical_density, 4)
        metrics["token_quantity"] = total_tokens
        
        return metrics
    
    def _calculate_pos_frequencies_udpipe(self, sentences) -> Dict:
        """Calculate POS tag frequencies from UDPipe."""
        metrics = {}
        
        tokens = [t for sent in sentences for t in sent if t["deprel"] != "punct"]
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            metrics["noun_freq"] = 0.0
            metrics["verb_freq"] = 0.0
            metrics["adj_freq"] = 0.0
            metrics["adv_freq"] = 0.0
            return metrics
        
        pos_counts = Counter(t["upos"] for t in tokens)
        
        metrics["noun_freq"] = round((pos_counts.get("NOUN", 0) / total_tokens) * 100, 2)
        metrics["verb_freq"] = round((pos_counts.get("VERB", 0) / total_tokens) * 100, 2)
        metrics["adj_freq"] = round((pos_counts.get("ADJ", 0) / total_tokens) * 100, 2)
        metrics["adv_freq"] = round((pos_counts.get("ADV", 0) / total_tokens) * 100, 2)
        
        return metrics
    
    def _calculate_lexical_metrics_udpipe(self, sentences) -> Dict:
        """Calculate lexical metrics from UDPipe tokens."""
        metrics = {}
        
        tokens = [t for sent in sentences for t in sent if t["deprel"] != "punct"]
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return {"avg_word_length": 0.0, "long_words_ratio": 0.0}
        
        word_lengths = [len(t["form"]) for t in tokens]
        metrics["avg_word_length"] = round(np.mean(word_lengths), 2)
        
        long_words = [t for t in tokens if len(t["form"]) > 6]
        metrics["long_words_ratio"] = round((len(long_words) / total_tokens) * 100, 2)
        
        return metrics
    
    def _calculate_syntax_metrics_udpipe(self, sentences) -> Dict:
        """Calculate syntactic metrics from UDPipe."""
        metrics = {}
        
        # Sentence length variance
        sentence_lengths = [len([t for t in sent if t["deprel"] != "punct"]) for sent in sentences]
        metrics["sentence_length_variance"] = round(np.var(sentence_lengths), 2) if len(sentence_lengths) > 1 else 0.0
        
        # Punctuation ratio
        total_tokens = sum(len(sent) for sent in sentences)
        punct_count = sum(len([t for t in sent if t["deprel"] == "punct"]) for sent in sentences)
        metrics["punctuation_ratio"] = round((punct_count / total_tokens) * 100, 2) if total_tokens > 0 else 0.0
        
        return metrics
    
    def _calculate_flesch_udpipe(self, text: str, sentences) -> float:
        """
        Calculate Flesch Reading Ease for Portuguese using UDPipe data.
        
        Formula: FRE = 248.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        """
        try:
            total_sentences = len(sentences)
            if total_sentences == 0:
                return 0.0
            
            # Count words (exclude punctuation)
            words = [t for sent in sentences for t in sent if t["deprel"] != "punct"]
            total_words = len(words)
            
            if total_words == 0:
                return 0.0
            
            # Estimate syllables from word forms
            total_syllables = sum(self._count_syllables_pt(t["form"]) for t in words)
            
            avg_sentence_length = total_words / total_sentences
            avg_syllables_per_word = total_syllables / total_words
            
            fre_score = 248.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            return round(fre_score, 2)
            
        except Exception as e:
            logger.warning(f"Error calculating Flesch score: {e}")
            return 0.0
    
    def _calculate_flesch_spacy(self, tokens, sentences) -> float:
        """Calculate Flesch Reading Ease using spaCy (fallback)."""
        try:
            total_sentences = len(sentences)
            if total_sentences == 0:
                return 0.0
            
            total_words = len(tokens)
            if total_words == 0:
                return 0.0
            
            total_syllables = sum(self._count_syllables_pt(t.text) for t in tokens)
            
            avg_sentence_length = total_words / total_sentences
            avg_syllables_per_word = total_syllables / total_words
            
            fre_score = 248.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            return round(fre_score, 2)
            
        except Exception as e:
            logger.warning(f"Error calculating Flesch score: {e}")
            return 0.0
    
    def _count_syllables_pt(self, word: str) -> int:
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
    
    def _calculate_ttr_spacy(self, tokens) -> float:
        """Calculate Type-Token Ratio from spaCy tokens."""
        if not tokens:
            return 0.0
        
        types = set(t.text.lower() for t in tokens)
        ttr = len(types) / len(tokens)
        return round(ttr, 4)
    
    def _extract_named_entities(self, doc) -> Dict:
        """Extract named entities using spaCy NER."""
        metrics = {}
        
        entity_counts = {"per_count": 0, "org_count": 0, "loc_count": 0}
        
        for ent in doc.ents:
            if ent.label_ in ["PER", "PERSON"]:
                entity_counts["per_count"] += 1
            elif ent.label_ == "ORG":
                entity_counts["org_count"] += 1
            elif ent.label_ in ["LOC", "GPE", "PLACE"]:
                entity_counts["loc_count"] += 1
        
        metrics.update(entity_counts)
        return metrics
    
    def _get_empty_metrics(self) -> Dict:
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
        }
