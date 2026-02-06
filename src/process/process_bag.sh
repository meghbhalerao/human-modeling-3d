#!/bin/bash

##############################################################################
# Simple Batch Processing Script for RealSense Bag Files
#
# Usage: ./simple_batch_process.sh
#
# Edit the variables below to customize:
##############################################################################

# Configuration - EDIT THESE
INPUT_FOLDER="./data/recordings"      # Folder containing .bag files
OUTPUT_FOLDER="./data/processed"       # Where to save JSON outputs
PRESET="batch"                         # Hydra preset: fast, quality, or batch

##############################################################################
# Script Start - No need to edit below this line
##############################################################################

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "Batch Processing RealSense Recordings"
echo "========================================="
echo ""
echo "Input folder:  $INPUT_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
echo "Preset:        $PRESET"
echo ""

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Check if input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo -e "${RED}Error: Input folder does not exist: $INPUT_FOLDER${NC}"
    exit 1
fi

# Count total bag files
total_bags=$(find "$INPUT_FOLDER" -type f -name "*.bag" | wc -l)
echo "Found $total_bags .bag file(s)"
echo ""

# Initialize counters
processed=0
failed=0

# Process each .bag file
find "$INPUT_FOLDER" -type f -name "*.bag" | while read -r bag_file; do
    # Get filename without path and extension
    basename=$(basename "$bag_file" .bag)
    output_file="$OUTPUT_FOLDER/${basename}_smpl.json"
    
    echo "---"
    echo "Processing: $basename"
    
    # Check if it's actually a bag file (basic check)
    if ! head -c 13 "$bag_file" 2>/dev/null | grep -q "ROSBAG"; then
        echo -e "${YELLOW}Warning: $basename may not be a valid bag file, skipping${NC}"
        continue
    fi
    
    # Run the extraction
    if python extract_smpl_poses_hydra.py \
        bag_file="$bag_file" \
        output_path="$output_file" \
        preset="$PRESET" \
        processing.visualize=false \
        logging.level=WARNING 2>&1 | grep -E "(ERROR|Successfully)" || true; then
        echo -e "${GREEN}✓ Done: $basename${NC}"
        ((processed++))
    else
        echo -e "${RED}✗ Failed: $basename${NC}"
        ((failed++))
    fi
done

# Print summary
echo ""
echo "========================================="
echo "Summary"
echo "========================================="
echo -e "Processed: ${GREEN}$processed${NC}"
if [ $failed -gt 0 ]; then
    echo -e "Failed:    ${RED}$failed${NC}"
fi
echo ""
echo "Output folder: $OUTPUT_FOLDER"
