#!/bin/bash
# This script runsecho "All sentiment analysis jobs started."
echo "Monitor progress with: tail -f $OUTPUT_DIR/sentiment_analysis.log"
echo "Results will be saved in: $OUTPUT_DIR/"

# Optional: Run analysis options
echo ""
echo "Additional analysis options:"
echo "1. Use lemmatized text column:"
echo "   python3 src/sentiment/roberta_sentiment.py --input_file=$INPUT_FILE --text_column=response_lemm"timent analysis on Portuguese text data

# Common parameters
INPUT_FILE="data/merged_data.parquet"
TEXT_COL="response"
OUTPUT_DIR="sentiment_results"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run sentiment analysis
run_sentiment_analysis() {
    local output_file=$1
    
    echo "Starting sentiment analysis"
    
    nohup python3 -u src/sentiment/roberta_sentiment.py \
        --input_file="$INPUT_FILE" \
        --text_column="$TEXT_COL" \
        --output_dir="$OUTPUT_DIR" > "$output_file" 2>&1 &
    
    echo "Sentiment analysis started in background (PID: $!)"
    echo "Check output: $output_file"
}

# Run the analysis
output_file="$OUTPUT_DIR/sentiment_analysis.log"
run_sentiment_analysis "$output_file"

echo "All sentiment analysis jobs started."
echo "Monitor progress with: tail -f $OUTPUT_DIR/sentiment_*.log"
echo "Results will be saved in: $OUTPUT_DIR/"

# Optional: Run analysis grouped by different columns
echo ""
echo "Additional analysis options:"
echo "1. Analyze specific model:"
echo "   python3 src/sentiment/roberta_sentiment.py --input_file=$INPUT_FILE --model_name=gemini-1.5-flash"
echo ""
echo "2. Use lemmatized text column:"
echo "   python3 src/sentiment/roberta_sentiment.py --input_file=$INPUT_FILE --text_column=response_lemm --model_name=all"
