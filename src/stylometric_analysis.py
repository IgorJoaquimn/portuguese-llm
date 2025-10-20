"""
Stylometric and Linguistic Analysis Module

This module implements comprehensive stylometric and linguistic analysis features
for parliamentary discourse analysis, including:

1. Readability metrics (Flesch Reading Ease)
2. Lexical diversity (TTR - already implemented)
3. Syntactic structure analysis (POS frequencies)
4. Named Entity Recognition (NER)

Based on the methodology described in parliamentary discourse analysis research.
"""

import pandas as pd
import numpy as np
import spacy
import textstat
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StylometricAnalyzer:
    """
    A comprehensive stylometric and linguistic analyzer for Portuguese text.
    """
    
    def __init__(self, model_name: str = "pt_core_news_lg"):
        """
        Initialize the analyzer with spaCy Portuguese model.
        
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
                logger.error("No Portuguese spaCy model found. Please install with: python -m spacy download pt_core_news_lg")
                raise
    
    def calculate_flesch_reading_ease_pt(self, text: str) -> float:
        """
        Calculate Flesch Reading Ease score adapted for Portuguese.
        
        The original Flesch formula adapted for Portuguese:
        FRE = 248.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
        
        Args:
            text: Input text
            
        Returns:
            Flesch Reading Ease score (can be negative for very complex texts)
        """
        if not text or not text.strip():
            return 0.0
        
        try:
            # Use textstat library which has Portuguese support
            # Note: textstat might not have perfect Portuguese syllable counting,
            # so we'll implement a basic version for Portuguese
            doc = self.nlp(text)
            
            # Count sentences
            sentences = list(doc.sents)
            total_sentences = len(sentences)
            
            if total_sentences == 0:
                return 0.0
            
            # Count words (exclude punctuation)
            words = [token for token in doc if not token.is_punct and not token.is_space]
            total_words = len(words)
            
            if total_words == 0:
                return 0.0
            
            # Estimate syllables for Portuguese
            total_syllables = sum(self._count_syllables_pt(token.text) for token in words)
            
            # Calculate Flesch Reading Ease for Portuguese
            # Adapted formula based on Portuguese language characteristics
            avg_sentence_length = total_words / total_sentences
            avg_syllables_per_word = total_syllables / total_words
            
            # Portuguese Flesch formula (adapted)
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
        
        # Portuguese vowels (including nasal vowels)
        vowels = "aeiouáéíóúàèìòùâêîôûãõç"
        
        # Remove punctuation
        word = re.sub(r'[^a-záéíóúàèìòùâêîôûãõç]', '', word)
        
        if not word:
            return 1
        
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Portuguese specific adjustments
        # Handle common diphthongs that should count as one syllable
        diphthongs = ['ai', 'au', 'ei', 'eu', 'oi', 'ou', 'ui']
        for diphthong in diphthongs:
            syllable_count -= word.count(diphthong)
        
        # Ensure at least one syllable
        return max(1, syllable_count)
    
    def calculate_pos_frequencies(self, text: str) -> Dict[str, float]:
        """
        Calculate relative frequencies of main grammatical categories.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with POS frequencies as percentages
        """
        if not text or not text.strip():
            return {"noun_freq": 0.0, "verb_freq": 0.0, "adj_freq": 0.0, "adv_freq": 0.0}
        
        try:
            doc = self.nlp(text)
            
            # Filter out punctuation and spaces
            tokens = [token for token in doc if not token.is_punct and not token.is_space]
            total_tokens = len(tokens)
            
            if total_tokens == 0:
                return {"noun_freq": 0.0, "verb_freq": 0.0, "adj_freq": 0.0, "adv_freq": 0.0}
            
            # Count POS tags
            pos_counts = Counter(token.pos_ for token in tokens)
            
            # Calculate frequencies as percentages
            noun_freq = (pos_counts.get('NOUN', 0) / total_tokens) * 100
            verb_freq = (pos_counts.get('VERB', 0) / total_tokens) * 100
            adj_freq = (pos_counts.get('ADJ', 0) / total_tokens) * 100
            adv_freq = (pos_counts.get('ADV', 0) / total_tokens) * 100
            
            return {
                "noun_freq": round(noun_freq, 2),
                "verb_freq": round(verb_freq, 2),
                "adj_freq": round(adj_freq, 2),
                "adv_freq": round(adv_freq, 2)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating POS frequencies: {e}")
            return {"noun_freq": 0.0, "verb_freq": 0.0, "adj_freq": 0.0, "adv_freq": 0.0}
    
    def extract_named_entities(self, text: str) -> Dict[str, int]:
        """
        Extract and count named entities (PER, ORG, LOC/GPE).
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with entity counts
        """
        if not text or not text.strip():
            return {"per_count": 0, "org_count": 0, "loc_count": 0}
        
        try:
            doc = self.nlp(text)
            
            # Count entities by type
            entity_counts = {"per_count": 0, "org_count": 0, "loc_count": 0}
            
            for ent in doc.ents:
                if ent.label_ == "PER" or ent.label_ == "PERSON":
                    entity_counts["per_count"] += 1
                elif ent.label_ == "ORG":
                    entity_counts["org_count"] += 1
                elif ent.label_ in ["LOC", "GPE", "PLACE"]:
                    entity_counts["loc_count"] += 1
            
            return entity_counts
            
        except Exception as e:
            logger.warning(f"Error extracting named entities: {e}")
            return {"per_count": 0, "org_count": 0, "loc_count": 0}
    
    def calculate_additional_metrics(self, text: str) -> Dict[str, float]:
        """
        Calculate additional stylometric metrics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with additional metrics
        """
        if not text or not text.strip():
            return {
                "avg_word_length": 0.0,
                "long_words_ratio": 0.0,
                "sentence_length_variance": 0.0,
                "punctuation_ratio": 0.0
            }
        
        try:
            doc = self.nlp(text)
            
            # Calculate average word length
            words = [token.text for token in doc if not token.is_punct and not token.is_space]
            avg_word_length = np.mean([len(word) for word in words]) if words else 0.0
            
            # Calculate ratio of long words (>6 characters)
            long_words = [word for word in words if len(word) > 6]
            long_words_ratio = (len(long_words) / len(words)) * 100 if words else 0.0
            
            # Calculate sentence length variance
            sentences = list(doc.sents)
            sentence_lengths = []
            for sent in sentences:
                sent_words = [token for token in sent if not token.is_punct and not token.is_space]
                sentence_lengths.append(len(sent_words))
            
            sentence_length_variance = np.var(sentence_lengths) if sentence_lengths else 0.0
            
            # Calculate punctuation ratio
            total_tokens = len(list(doc))
            punct_tokens = len([token for token in doc if token.is_punct])
            punctuation_ratio = (punct_tokens / total_tokens) * 100 if total_tokens > 0 else 0.0
            
            return {
                "avg_word_length": round(avg_word_length, 2),
                "long_words_ratio": round(long_words_ratio, 2),
                "sentence_length_variance": round(sentence_length_variance, 2),
                "punctuation_ratio": round(punctuation_ratio, 2)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating additional metrics: {e}")
            return {
                "avg_word_length": 0.0,
                "long_words_ratio": 0.0,
                "sentence_length_variance": 0.0,
                "punctuation_ratio": 0.0
            }
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Perform comprehensive stylometric analysis on text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with all stylometric metrics
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for analysis")
            return self._get_empty_metrics()
        
        try:
            # Calculate all metrics
            flesch_score = self.calculate_flesch_reading_ease_pt(text)
            pos_freqs = self.calculate_pos_frequencies(text)
            entities = self.extract_named_entities(text)
            additional = self.calculate_additional_metrics(text)
            
            # Combine all metrics
            results = {
                "flesch_reading_ease": flesch_score,
                **pos_freqs,
                **entities,
                **additional
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive text analysis: {e}")
            return self._get_empty_metrics()
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary with all expected keys."""
        return {
            "flesch_reading_ease": 0.0,
            "noun_freq": 0.0,
            "verb_freq": 0.0,
            "adj_freq": 0.0,
            "adv_freq": 0.0,
            "per_count": 0,
            "org_count": 0,
            "loc_count": 0,
            "avg_word_length": 0.0,
            "long_words_ratio": 0.0,
            "sentence_length_variance": 0.0,
            "punctuation_ratio": 0.0
        }


def process_dataframe(df: pd.DataFrame, 
                     text_column: str = 'response',
                     batch_size: int = 100) -> pd.DataFrame:
    """
    Process a DataFrame and add stylometric analysis columns.
    
    Args:
        df: Input DataFrame
        text_column: Name of the column containing text to analyze
        batch_size: Number of rows to process at once (for memory management)
        
    Returns:
        DataFrame with added stylometric columns
    """
    logger.info(f"Starting stylometric analysis on {len(df)} rows...")
    
    # Initialize analyzer
    analyzer = StylometricAnalyzer()
    
    # Initialize result lists
    results = []
    
    # Process in batches
    for i in range(0, len(df), batch_size):
        batch_end = min(i + batch_size, len(df))
        batch_df = df.iloc[i:batch_end].copy()
        
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1} (rows {i+1}-{batch_end})")
        
        batch_results = []
        for idx, row in batch_df.iterrows():
            text = str(row[text_column]) if pd.notna(row[text_column]) else ""
            analysis = analyzer.analyze_text(text)
            batch_results.append(analysis)
        
        results.extend(batch_results)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Add results to original DataFrame
    for col in results_df.columns:
        df[col] = results_df[col].values
    
    logger.info("Stylometric analysis completed!")
    logger.info(f"Added columns: {list(results_df.columns)}")
    
    return df


def add_stylometric_features(input_file: str, 
                           output_file: str,
                           text_column: str = 'response',
                           batch_size: int = 100) -> None:
    """
    Add stylometric features to a parquet file.
    
    Args:
        input_file: Path to input parquet file
        output_file: Path to output parquet file
        text_column: Name of the column containing text to analyze
        batch_size: Number of rows to process at once
    """
    try:
        # Load data
        logger.info(f"Loading data from {input_file}...")
        df = pd.read_parquet(input_file)
        logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        
        # Check if text column exists
        if text_column not in df.columns:
            available_text_cols = [col for col in df.columns if 'response' in col.lower() or 'text' in col.lower()]
            raise ValueError(f"Column '{text_column}' not found. Available text columns: {available_text_cols}")
        
        # Process DataFrame
        df_processed = process_dataframe(df, text_column, batch_size)
        
        # Save results
        logger.info(f"Saving results to {output_file}...")
        df_processed.to_parquet(output_file, index=False)
        logger.info(f"Successfully saved {len(df_processed)} rows to {output_file}")
        
        # Print summary
        new_columns = [col for col in df_processed.columns if col not in df.columns]
        logger.info(f"Added {len(new_columns)} new columns: {new_columns}")
        
        # Print some statistics
        if new_columns:
            print("\n📊 Summary Statistics for New Columns:")
            print("=" * 50)
            for col in new_columns:
                if df_processed[col].dtype in ['float64', 'int64']:
                    stats = df_processed[col].describe()
                    print(f"\n{col}:")
                    print(f"  Mean: {stats['mean']:.2f}")
                    print(f"  Std:  {stats['std']:.2f}")
                    print(f"  Min:  {stats['min']:.2f}")
                    print(f"  Max:  {stats['max']:.2f}")
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    input_file = "/home/igo/faculdade/poc/data/sentiment_results.parquet"
    output_file = "/home/igo/faculdade/poc/data/sentiment_results_with_stylometric.parquet"
    
    # Add stylometric features
    add_stylometric_features(
        input_file=input_file,
        output_file=output_file,
        text_column='response',
        batch_size=50  # Smaller batch size for memory efficiency
    )