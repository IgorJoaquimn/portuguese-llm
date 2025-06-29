#!/bin/bash

# Directory containing prompt files
PROMPT_DIR="prompts/rendered"

# Create logs directory if it doesn't exist
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# Check if there are any .pickle files
shopt -s nullglob  # Prevents glob from returning a literal string if no matches are found
for file in "$PROMPT_DIR"/*.pickle; do
    # Extract the base filename (without extension)
    base_filename=$(basename "$file" .pickle)

    # Construct the command
    CMD="nohup python3 -u -m src.udpipe.call_udpipe --record_path=\"$file\" > \"$LOG_DIR/${base_filename}.log\" 2>&1 &"

    # Echo the command
    echo "$CMD"

    # Run the command
    eval "$CMD"
done

# Wait for all background jobs to finish
wait

shopt -u nullglob  # Restore default behavior

