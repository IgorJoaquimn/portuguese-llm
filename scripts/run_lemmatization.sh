#!/bin/bash
# Script to run lemmatization on data using UDPipe

# Default parameters
INPUT_FILE="data/merged_data.parquet"
OUTPUT_FILE="data/merged_data_lemm.parquet"
TEXT_COL="response"
OUTPUT_COL="response_lemm"
SAVE_INTERVAL=100
WORKERS=""
PARALLEL="process"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input|-i)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --text-col|-t)
            TEXT_COL="$2"
            shift 2
            ;;
        --output-col|-c)
            OUTPUT_COL="$2"
            shift 2
            ;;
        --save-interval|-s)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --workers|-w)
            WORKERS="$2"
            shift 2
            ;;
        --parallel|-p)
            PARALLEL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -i, --input FILE           Input parquet file (default: data/merged_data.parquet)"
            echo "  -o, --output FILE          Output parquet file (default: data/merged_data_lemm.parquet)"
            echo "  -t, --text-col COLUMN      Text column to lemmatize (default: response)"
            echo "  -c, --output-col COLUMN    Output column name (default: response_lemm)"
            echo "  -s, --save-interval N      Save progress every N items (default: 100)"
            echo "  -w, --workers N            Number of parallel workers (default: auto)"
            echo "  -p, --parallel METHOD      Parallel method: thread|process|sequential (default: thread)"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                          # Use defaults with threading"
            echo "  $0 --workers 4 --parallel thread           # 4 threads"
            echo "  $0 --parallel process --workers 2          # 2 processes"
            echo "  $0 --parallel sequential                   # No parallelization"
            echo "  $0 --input data/custom.parquet --workers 8 # Custom input with 8 workers"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Running lemmatization with:"
echo "  Input file: $INPUT_FILE"
echo "  Output file: $OUTPUT_FILE"
echo "  Text column: $TEXT_COL"
echo "  Output column: $OUTPUT_COL"
echo "  Save interval: $SAVE_INTERVAL"
echo "  Parallel method: $PARALLEL"
echo "  Workers: ${WORKERS:-auto}"
echo ""

# Check if input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

# Build command with optional workers parameter
CMD="python3 src/udpipe/lemmatize_data.py \
    --input_file=\"$INPUT_FILE\" \
    --output_file=\"$OUTPUT_FILE\" \
    --text_column=\"$TEXT_COL\" \
    --output_column=\"$OUTPUT_COL\" \
    --save_interval=\"$SAVE_INTERVAL\" \
    --parallel=\"$PARALLEL\""

if [[ -n "$WORKERS" ]]; then
    CMD="$CMD --workers=\"$WORKERS\""
fi

# Run the lemmatization
eval $CMD

echo ""
echo "Lemmatization completed! Results saved to: $OUTPUT_FILE"
