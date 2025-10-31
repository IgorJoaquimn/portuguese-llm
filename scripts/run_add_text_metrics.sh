#!/bin/bash
# Script to compute and add text metrics to parquet file
# Uses UnifiedTextMetrics to add 23 linguistic/stylometric metrics as columns

# Check if spaCy Portuguese model is installed
echo "Checking for spaCy Portuguese model..."
python3 -c "import spacy; spacy.load('pt_core_news_sm')" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "📥 Installing Portuguese spaCy model..."
    python3 -m spacy download pt_core_news_sm
fi

echo ""
echo "Starting metrics enrichment..."
echo ""

python3 -u -m src.add_text_metrics \
    --input_file=data/gemini_data.parquet \
    --output_file=data/gemini_data_with_metrics.parquet \
    --text_column=response \
    --udpipe_column=udpipe_result \
    --batch_size=100
