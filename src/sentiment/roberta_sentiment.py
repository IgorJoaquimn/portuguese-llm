#!/usr/bin/env python3
"""
Sentiment analysis using RoBERTa model for Portuguese text data.
"""

import argparse
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_roberta_sentiment():
    """Setup RoBERTa sentiment analysis pipeline for Portuguese."""
    try:
        # Using a Portuguese RoBERTa model for sentiment analysis
        model_name = "cardiffnlp/xlm-roberta-base-tweet-sentiment-pt"
        
        # Alternative Portuguese models:
        # "neuralmind/bert-base-portuguese-cased" 
        # "pierreguillou/bert-base-cased-pt-lenerbr"
        
        logger.info(f"Loading RoBERTa model: {model_name}")
        
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512
        )
        
        return sentiment_pipeline
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Fallback to a simpler model
        logger.info("Falling back to default sentiment model")
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512
        )
        return sentiment_pipeline


def analyze_sentiment_batch(texts, sentiment_pipeline, batch_size=32):
    """
    Analyze sentiment for a batch of texts.
    
    Args:
        texts (list): List of text strings to analyze
        sentiment_pipeline: Hugging Face sentiment pipeline
        batch_size (int): Batch size for processing
    
    Returns:
        list: List of sentiment results with labels and scores
    """
    results = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing sentiment"):
        batch = texts[i:i + batch_size]
        
        try:
            batch_results = sentiment_pipeline(batch)
            results.extend(batch_results)
        except Exception as e:
            logger.warning(f"Error processing batch {i//batch_size}: {e}")
            # Add empty results for failed batch
            for _ in batch:
                results.append({"label": "UNKNOWN", "score": 0.0})
    
    return results


def perform_sentiment_analysis(text_column="response", data_file="../../data/merged_data.parquet"):
    """
    Perform sentiment analysis on text data using RoBERTa.
    
    Args:
        text_column (str): Column name containing text to analyze
        data_file (str): Path to the parquet data file
    
    Returns:
        pandas.DataFrame: DataFrame with original data plus sentiment columns
    """
    # Load data
    logger.info(f"Loading data from {data_file}...")
    df = pd.read_parquet(data_file)
    
    model_data = df.copy()
    logger.info(f"Processing all data: {len(model_data)} records")
    
    # Check if text column exists
    if text_column not in model_data.columns:
        raise ValueError(f"Text column '{text_column}' not found in data")
    
    # Clean text data
    logger.info("Cleaning text data...")
    model_data[text_column] = model_data[text_column].fillna("")
    model_data = model_data[model_data[text_column].str.len() > 0]
    
    # Setup sentiment pipeline
    sentiment_pipeline = setup_roberta_sentiment()
    
    # Analyze sentiment
    logger.info("Starting sentiment analysis...")
    texts = model_data[text_column].tolist()
    
    sentiment_results = analyze_sentiment_batch(texts, sentiment_pipeline)
    
    # Add results to dataframe
    logger.info("Adding sentiment results to dataframe...")
    model_data["sentiment_label"] = [result["label"] for result in sentiment_results]
    model_data["sentiment_score"] = [result["score"] for result in sentiment_results]
    
    # Convert sentiment labels to standardized format
    label_mapping = {
        "POSITIVE": "positive",
        "NEGATIVE": "negative", 
        "NEUTRAL": "neutral",
        "LABEL_0": "negative",  # Some models use these labels
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
        "UNKNOWN": "unknown"
    }
    
    model_data["sentiment_normalized"] = model_data["sentiment_label"].map(
        lambda x: label_mapping.get(x, x.lower())
    )
    
    return model_data


def generate_sentiment_summary(df):
    """
    Generate summary statistics for sentiment analysis.
    
    Args:
        df (pandas.DataFrame): DataFrame with sentiment analysis results
    
    Returns:
        pandas.DataFrame: Summary statistics
    """
    # Overall sentiment distribution
    sentiment_summary = df["sentiment_normalized"].value_counts().to_frame("count")
    sentiment_summary["percentage"] = (sentiment_summary["count"] / len(df) * 100).round(2)
    
    # Basic sentiment scores statistics
    score_summary = df["sentiment_score"].describe().to_frame("sentiment_score")
    
    return sentiment_summary, score_summary


def main():
    parser = argparse.ArgumentParser(description="Perform sentiment analysis using RoBERTa on Portuguese text data")
    
    parser.add_argument("--input_file", type=str, default="data/merged_data.parquet",
                        help="Path to input parquet file")
    parser.add_argument("--text_column", type=str, default="response",
                        help="Column name containing text to analyze")
    parser.add_argument("--output_dir", type=str, default="sentiment_results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Perform sentiment analysis
        results_df = perform_sentiment_analysis(
            text_column=args.text_column,
            data_file=args.input_file
        )
        
        # Save detailed results
        output_file = os.path.join(args.output_dir, f"sentiment_analysis.parquet")
        results_df.to_parquet(output_file, index=False)
        logger.info(f"Detailed results saved to: {output_file}")
        
        # Generate and save summary
        sentiment_summary, score_summary = generate_sentiment_summary(results_df)
        
        summary_file = os.path.join(args.output_dir, f"sentiment_summary.csv")
        sentiment_summary.to_csv(summary_file)
        logger.info(f"Sentiment summary saved to: {summary_file}")
        
        score_file = os.path.join(args.output_dir, f"sentiment_scores.csv") 
        score_summary.to_csv(score_file)
        logger.info(f"Score summary saved to: {score_file}")
        
        # Print basic statistics
        logger.info("\n=== Sentiment Analysis Results ===")
        logger.info(f"Total texts analyzed: {len(results_df)}")
        logger.info(f"Sentiment distribution:")
        sentiment_counts = results_df["sentiment_normalized"].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(results_df) * 100)
            logger.info(f"  {sentiment}: {count} ({percentage:.1f}%)")
        
        logger.info(f"\nAverage sentiment score: {results_df['sentiment_score'].mean():.3f}")
        logger.info(f"Sentiment score std: {results_df['sentiment_score'].std():.3f}")
        
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}")
        raise


if __name__ == "__main__":
    main()
