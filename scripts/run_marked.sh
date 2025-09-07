#!/bin/bash
# This script runs the masked personas generation process

# Common parameters
INPUT_FILE="data/merged_data.parquet"
TEXT_COL="response_lemm"
MODEL_NAMES=("gemini-1.5-flash" "gemini-2.0-flash")

# Define arrays for each category
declare -A RACA_TARGETS=(
    ["preta"]="branca"
    ["parda"]="branca"
    ["branca"]="branca"
    ["amarela"]="branca"
    ["indígena"]="branca"
)

declare -A GENERO_TARGETS=(
    ["homem"]="homem"
    ["mulher"]="homem"
    ["não-binária"]="homem"
)

declare -A REGIAO_TARGETS=(
    ["nortista"]="sudestina"
    ["nordestina"]="sudestina"
    ["sulista"]="sudestina"
    ["sudestina"]="sudestina"
    ["centro-oestina"]="sudestina"
)

# Function to run masked personas
run_masked_persona() {
    local target_col=$1
    local target_name=$2
    local unmarked_name=$3
    local model_name=$4
    local output_file=$5
    
    nohup python3 -u src/masked/masked_personas.py \
        --input_file="$INPUT_FILE" \
        --target_col="$target_col" \
        --target_name="$target_name" \
        --unmarked_name="$unmarked_name" \
        --text_col="$TEXT_COL" \
        --model_name="$model_name" > "$output_file" 2>&1 &
}

# Process each model
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    OUTPUT_DIR="personas/${MODEL_NAME}/response"
    
    # Process raca category
    for target in "${!RACA_TARGETS[@]}"; do
        unmarked="${RACA_TARGETS[$target]}"
        output_file="$OUTPUT_DIR/${target}_${unmarked}_response.txt"
        run_masked_persona "raca" "$target" "$unmarked" "$MODEL_NAME" "$output_file"
    done

    # Process genero category
    for target in "${!GENERO_TARGETS[@]}"; do
        unmarked="${GENERO_TARGETS[$target]}"
        output_file="$OUTPUT_DIR/${target}_${unmarked}_response.txt"
        run_masked_persona "genero" "$target" "$unmarked" "$MODEL_NAME" "$output_file"
    done

    # Process regiao category
    for target in "${!REGIAO_TARGETS[@]}"; do
        unmarked="${REGIAO_TARGETS[$target]}"
        output_file="$OUTPUT_DIR/${target}_${unmarked}_response.txt"
        run_masked_persona "regiao" "$target" "$unmarked" "$MODEL_NAME" "$output_file"
    done
done