#!/bin/bash

##############################################################################
# Simple Batch Processing Script for RealSense Bag Files
#
# Usage: ./simple_batch_process.sh
#
# Edit the variables below to customize:
##############################################################################

# Configuration - EDIT THESE
INPUT_FOLDER="../../data/TOY/recordings"      # Folder containing .bag files
OUTPUT_FOLDER="../../data/TOY/processed"       # Where to save JSON outputs


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
echo ""

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Check if input folder exists
if [ ! -d "$INPUT_FOLDER" ]; then
    echo -e "${RED}Error: Input folder does not exist: $INPUT_FOLDER${NC}"
    exit 1
fi

# Find all .bag files into an array
bag_files=()
while IFS= read -r -d '' file; do
    bag_files+=("$file")
done < <(find "$INPUT_FOLDER" -type f -name "*.bag" -print0)

total_bags=${#bag_files[@]}

echo "Found $total_bags .bag file(s)"
echo ""

if [ $total_bags -eq 0 ]; then
    echo "No .bag files found in $INPUT_FOLDER"
    exit 0
fi

# Initialize counters
processed=0
failed=0
skipped=0

# Process each .bag file
for bag_file in "${bag_files[@]}"; do
    # Get filename without path and extension
    basename=$(basename "$bag_file" .bag)
    output_file="$OUTPUT_FOLDER/${basename}_smpl.json"
    
    echo "---"
    echo "Processing: $basename"
    echo "  Input:  $bag_file"
    echo "  Output: $output_file"
    
    # Check if it's actually a bag file (basic check)
    if ! head -c 13 "$bag_file" 2>/dev/null | grep -q "ROSBAG"; then
        echo -e "${YELLOW}Warning: $basename may not be a valid bag file, skipping${NC}"
        ((skipped++))
        continue
    fi
    
    # Run the extraction
    if python extract_smpl_mano.py \
        bag_file="$bag_file" \
        output_path="$output_file" \
        processing.visualize=false \
        logging.level=WARNING; then
        
        # Verify output file was created
        if [ -f "$output_file" ]; then
            echo -e "${GREEN}✓ Success: $basename${NC}"
            ((processed++))
        else
            echo -e "${RED}✗ Failed: Output file not created for $basename${NC}"
            ((failed++))
        fi
    else
        echo -e "${RED}✗ Failed: Python script error for $basename${NC}"
        ((failed++))
    fi
    echo ""
done

# Print summary
echo "========================================="
echo "Summary"
echo "========================================="
echo "Total files found:     $total_bags"
echo -e "Successfully processed: ${GREEN}$processed${NC}"

if [ $skipped -gt 0 ]; then
    echo -e "Skipped (invalid):      ${YELLOW}$skipped${NC}"
fi

if [ $failed -gt 0 ]; then
    echo -e "Failed:                 ${RED}$failed${NC}"
fi

echo ""
echo "Output folder: $OUTPUT_FOLDER"

# List generated files
if [ $processed -gt 0 ]; then
    echo ""
    echo "Generated files:"
    ls -lh "$OUTPUT_FOLDER"/*.json 2>/dev/null || echo "  (none found)"
fi

# Exit with error if any failed
if [ $failed -gt 0 ]; then
    exit 1
fi