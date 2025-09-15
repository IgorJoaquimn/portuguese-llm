#!/bin/bash

# Directory containing prompt files
PROMPT_DIR="prompts/rendered"

# Check if there are any .pickle files
shopt -s nullglob  # Prevents glob from returning a literal string if no matches are found
for file in "$PROMPT_DIR"/*.pickle; do
    # Extract the filename
    filename=$(basename "$file")

    # Construct the command
    CMD="python3 -u -m src.prompting.call_llm --record_path=$file --ntimes=10"

    # Echo the command
    echo "$CMD"

    # Run the Python script sequentially
    eval "$CMD"
done

# Restore default behavior
shopt -u nullglob
