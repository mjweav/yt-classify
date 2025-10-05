#!/bin/bash

# YT-Classify Runner Script
# Usage: ./run.sh [input_file] [output_dir]

set -e  # Exit on any error

# Default values
INPUT_FILE="${1:-data/channels.json}"
OUTPUT_DIR="${2:-out}"

echo "YT-Classify Runner"
echo "=================="
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if Python script exists
if [ ! -f "classify.py" ]; then
    echo "Error: classify.py not found!"
    exit 1
fi

# Run the classifier
echo "Running classification..."
python3 classify.py "$INPUT_FILE" "$OUTPUT_DIR"

# Check if outputs were generated
echo ""
echo "Checking outputs..."
if [ -f "$OUTPUT_DIR/clusters.json" ]; then
    echo "✓ clusters.json generated"
    CHANNEL_COUNT=$(python3 -c "
import json
with open('$OUTPUT_DIR/clusters.json', 'r') as f:
    data = json.load(f)
    total = sum(sum(len(cluster['items']) for cluster in umbrella['clusters']) for umbrella in data['umbrellas'])
    print(total)
")
    echo "  - Classified: $CHANNEL_COUNT channels"
else
    echo "✗ clusters.json missing"
fi

if [ -f "$OUTPUT_DIR/clusters.csv" ]; then
    echo "✓ clusters.csv generated"
    CSV_LINES=$(wc -l < "$OUTPUT_DIR/clusters.csv")
    echo "  - CSV rows: $CSV_LINES"
else
    echo "✗ clusters.csv missing"
fi

if [ -f "$OUTPUT_DIR/report.md" ]; then
    echo "✓ report.md generated"
else
    echo "✗ report.md missing"
fi

if [ -f "$OUTPUT_DIR/report.html" ]; then
    echo "✓ report.html generated"
else
    echo "✗ report.html missing"
fi

echo ""
echo "Classification complete!"
echo "Open $OUTPUT_DIR/report.html to review results"
