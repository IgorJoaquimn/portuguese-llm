#!/bin/bash

# Directory containing prompt files
PROMPT_DIR="prompts"

# Loop over each file in the directory
for file in "$PROMPT_DIR"/*.tmplt; do
    # Extract the filename
    filename=$(basename "$file")
    
    # Construct the command
    CMD="python3 src/promptl.py --prompt_path=$PROMPT_DIR/$filename --trait_list_path=$PROMPT_DIR/traits.json"
    
    # Echo the command
    echo "$CMD"
    
    # Run the Python script
    eval $CMD
done

