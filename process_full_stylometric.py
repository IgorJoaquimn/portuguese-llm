#!/usr/bin/env python3
"""
Full Dataset Stylometric Processing Script

This script processes the complete sentiment results dataset and adds 
comprehensive stylometric and linguistic analysis metrics.

Usage:
    python process_full_stylometric.py

The script will:
1. Load the sentiment_results.parquet file
2. Add stylometric metrics (Flesch Reading Ease, POS frequencies, NER, etc.)
3. Save the enhanced dataset with all new columns
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Add src directory to Python path
sys.path.append('/home/igo/faculdade/poc/src')

from stylometric_analysis import StylometricAnalyzer, process_dataframe

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/igo/faculdade/poc/data/stylometric_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main processing function."""
    
    # File paths
    input_file = "/home/igo/faculdade/poc/data/sentiment_results.parquet"
    output_file = "/home/igo/faculdade/poc/data/sentiment_results_with_stylometric.parquet"
    backup_file = "/home/igo/faculdade/poc/data/sentiment_results_backup.parquet"
    
    logger.info("=" * 60)
    logger.info("STYLOMETRIC ANALYSIS PROCESSING STARTED")
    logger.info("=" * 60)
    
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Load data
        logger.info(f"Loading data from: {input_file}")
        df = pd.read_parquet(input_file)
        logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
        
        # Create backup
        logger.info(f"Creating backup: {backup_file}")
        df.to_parquet(backup_file, index=False)
        
        # Check available text columns
        text_columns = [col for col in df.columns if 'response' in col.lower()]
        logger.info(f"Available text columns: {text_columns}")
        
        # Choose the best text column
        if 'response' in df.columns:
            text_column = 'response'
        elif 'response_lemm' in df.columns:
            text_column = 'response_lemm'
        else:
            text_column = text_columns[0] if text_columns else None
            
        if not text_column:
            raise ValueError("No suitable text column found for analysis")
            
        logger.info(f"Using text column: {text_column}")
        
        # Process dataset in batches
        logger.info("Starting stylometric analysis...")
        logger.info("This process may take 30-60 minutes depending on dataset size")
        
        df_processed = process_dataframe(
            df, 
            text_column=text_column, 
            batch_size=25  # Small batch size for memory efficiency
        )
        
        # Check new columns
        original_columns = set(df.columns)
        new_columns = [col for col in df_processed.columns if col not in original_columns]
        logger.info(f"Added {len(new_columns)} new columns: {new_columns}")
        
        # Save processed data
        logger.info(f"Saving processed data to: {output_file}")
        df_processed.to_parquet(output_file, index=False)
        
        # Generate summary statistics
        logger.info("Generating summary statistics...")
        numeric_new_cols = [col for col in new_columns if df_processed[col].dtype in ['float64', 'int64']]
        
        if numeric_new_cols:
            summary_stats = df_processed[numeric_new_cols].describe()
            
            # Save summary to file
            summary_file = "/home/igo/faculdade/poc/data/stylometric_summary.csv"
            summary_stats.to_csv(summary_file)
            logger.info(f"Summary statistics saved to: {summary_file}")
            
            # Print key statistics
            logger.info("\nKEY STATISTICS:")
            logger.info("-" * 40)
            for col in numeric_new_cols:
                mean_val = summary_stats.loc['mean', col]
                std_val = summary_stats.loc['std', col]
                logger.info(f"{col}: Mean={mean_val:.2f}, Std={std_val:.2f}")
        
        # Success message
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETED SUCCESSFULLY!")
        logger.info(f"Input: {len(df):,} rows")
        logger.info(f"Output: {len(df_processed):,} rows with {len(new_columns)} new metrics")
        logger.info(f"Enhanced dataset saved to: {output_file}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        logger.error("Check the log file for detailed error information")
        return False


def check_requirements():
    """Check if required packages are installed."""
    
    logger.info("Checking requirements...")
    
    required_packages = ['pandas', 'numpy', 'spacy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package}")
    
    # Check spaCy Portuguese model
    try:
        import spacy
        nlp = spacy.load("pt_core_news_lg")
        logger.info("✅ spaCy Portuguese model (pt_core_news_lg)")
    except IOError:
        try:
            nlp = spacy.load("pt_core_news_sm")
            logger.warning("⚠️  Using pt_core_news_sm (smaller model)")
            logger.warning("   For better results, install: python -m spacy download pt_core_news_lg")
        except IOError:
            logger.error("❌ No Portuguese spaCy model found")
            logger.error("   Install with: python -m spacy download pt_core_news_lg")
            missing_packages.append("pt_core_news_lg")
    
    if missing_packages:
        logger.error(f"Missing requirements: {missing_packages}")
        return False
    
    logger.info("All requirements satisfied!")
    return True


if __name__ == "__main__":
    print("🚀 Stylometric Analysis Processing Script")
    print("📊 This will enhance your dataset with linguistic metrics")
    print()
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements not met. Please install missing packages.")
        sys.exit(1)
    
    # Ask for confirmation
    response = input("🤔 Process full dataset? This may take 30-60 minutes (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("👋 Processing cancelled")
        sys.exit(0)
    
    # Run processing
    success = main()
    
    if success:
        print("\n🎉 Processing completed successfully!")
        print("📁 Check the output file for your enhanced dataset")
    else:
        print("\n💥 Processing failed. Check the log file for details.")
        sys.exit(1)