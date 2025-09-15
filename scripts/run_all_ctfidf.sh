#!/bin/bash

# Script to run Class-based TF-IDF analysis for all combinations of models and traits
# Models: gemini-1.5-flash, gemini-2.0-flash
# Traits: raca, genero, regiao, localidade

set -e  # Exit on any error

# Define models and traits
models=("gpt-4o" "gpt-4o-mini")
traits=("original_prompt" "raca" "genero" "regiao" "localidade")

# Create output directory
output_dir="ctfidf_results"
mkdir -p "$output_dir"

echo "Starting Class-based TF-IDF analysis for all combinations..."
echo "Output directory: $output_dir"
echo "Models: ${models[*]}"
echo "Traits: ${traits[*]}"
echo "Total combinations: $((${#models[@]} * ${#traits[@]}))"
echo ""

# Counter for progress tracking
counter=0
total=$((${#models[@]} * ${#traits[@]}))

# Loop through all combinations
for model in "${models[@]}"; do
    for trait in "${traits[@]}"; do
        counter=$((counter + 1))
        echo "[$counter/$total] Processing: Model=$model, Trait=$trait"
        
        # Run the analysis
        python3 -u -m src.tf-idf.ctf_idf "$model" "$trait" --data-file "data/merged_data_sentiment.parquet" --output-dir "$output_dir"
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed: $model - $trait"
        else
            echo "✗ Failed: $model - $trait"
        fi
        echo ""
    done
done

echo "All analyses completed!"
echo "Results saved in: $output_dir"
echo ""
echo "Generated files:"
ls -la "$output_dir"/ctfidf_*.csv
