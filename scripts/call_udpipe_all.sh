
#!/bin/bash

# Directory containing prompt files
PROMPT_DIR="prompts/rendered"

# Check if there are any .pickle files
shopt -s nullglob  # Prevents glob from returning a literal string if no matches are found
for file in "$PROMPT_DIR"/*.pickle; do
    # Extract the filename
    filename=$(basename "$file")

    # Construct the command
    CMD="python3 -u -m src.udpipe.call_udpipe --record_path=$file"

    # Echo the command
    echo "$CMD"

    # Run the Python script
    eval "$CMD"
done
shopt -u nullglob  # Restore default behavior

