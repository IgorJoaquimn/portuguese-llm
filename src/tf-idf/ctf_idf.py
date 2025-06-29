#!/usr/bin/env python3
"""
Class-based TF-IDF analysis for text data grouped by specified column.
"""

import argparse
import os
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer


def setup_nltk_stopwords():
    """Download and setup Portuguese stopwords from NLTK."""
    try:
        from nltk.corpus import stopwords
        portuguese_stopwords = set(stopwords.words('portuguese'))
    except:
        # Download stopwords if not available
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        portuguese_stopwords = set(stopwords.words('portuguese'))
    return portuguese_stopwords


def perform_ctfidf_analysis(model_name, agg_column, data_file="../../data/merged_data.parquet"):
    """
    Perform Class-based TF-IDF analysis on text data.
    
    Args:
        model_name (str): Name of the model to filter data by
        agg_column (str): Column name to group data by
        data_file (str): Path to the parquet data file
    
    Returns:
        pandas.DataFrame: DataFrame with top words and TF-IDF values per class
    """
    # Load data
    print(f"Loading data from {data_file}...")
    df = pd.read_parquet(data_file)
    
    # Filter by model
    print(f"Filtering data for model: {model_name}")
    model_data = df[df["model"] == model_name]
    
    if model_data.empty:
        raise ValueError(f"No data found for model '{model_name}'")
    
    print(f"Found {len(model_data)} records for model '{model_name}'")
    
    # Group by specified column and aggregate responses
    print(f"Grouping data by '{agg_column}' column...")
    docs_per_class = model_data.groupby([agg_column], as_index=False).agg({'response': ' '.join})
    
    print(f"Created {len(docs_per_class)} groups")
    
    # Setup Portuguese stopwords
    portuguese_stopwords = setup_nltk_stopwords()
    
    # Vectorize text
    print("Vectorizing text...")
    count_vectorizer = CountVectorizer(stop_words=list(portuguese_stopwords)).fit(docs_per_class.response)
    count = count_vectorizer.transform(docs_per_class.response)
    words = count_vectorizer.get_feature_names_out()
    
    # Apply Class-based TF-IDF
    print("Applying Class-based TF-IDF transformation...")
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True).fit_transform(count).toarray()
    
    # Extract top words per class
    print("Extracting top words per class...")
    words_per_class = {}
    for label in range(len(docs_per_class[agg_column])):
        # Get word indices in decreasing order of TF-IDF score
        top_indices = ctfidf[label].argsort()[::-1]
        # Create list of tuples (word, tfidf_value)
        words_per_class[docs_per_class[agg_column].iloc[label]] = [
            (words[index], ctfidf[label][index])
            for index in top_indices
        ]
    
    # Convert to DataFrame
    print("Creating combined DataFrame...")
    combined_data = {}
    
    for class_name, word_tfidf_list in words_per_class.items():
        # Format as "word (tfidf_value)" for each entry
        combined_data[class_name] = [f"{item[0]} ({item[1]:.4f})" for item in word_tfidf_list]
    
    # Create DataFrame
    combined_df = pd.DataFrame.from_dict(combined_data, orient='index').T
    combined_df.index.name = 'rank'
    combined_df.columns.name = agg_column
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(description='Perform Class-based TF-IDF analysis on text data')
    parser.add_argument('model_name', help='Name of the model to filter data by')
    parser.add_argument('agg_column', help='Column name to group data by (e.g., genero, regiao)')
    parser.add_argument('--data-file', default='data/merged_data.parquet', 
                       help='Path to the parquet data file (default: data/merged_data.parquet)')
    parser.add_argument('--output-dir', default='.', 
                       help='Output directory for CSV file (default: current directory)')
    
    args = parser.parse_args()
    
    try:
        # Perform analysis
        result_df = perform_ctfidf_analysis(args.model_name, args.agg_column, args.data_file)
        
        # Generate output filename
        output_filename = f"ctfidf_{args.model_name.replace('.', '_')}_{args.agg_column}.csv"
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Save to CSV
        print(f"Saving results to {output_path}...")
        result_df.to_csv(output_path, index=True)
        
        print(f"Analysis complete! Results saved to: {output_path}")
        print(f"\nTop 15 words preview:")
        print(result_df.head(15))
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())



