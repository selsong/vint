#!/bin/bash

# Base directory containing the progression images
BASE_DIR="/p0/data/selina/vint_release/train/logs/vint_test/log_run_1_2025_04_01_22_26_42"

echo "Looking for progression directories in: $BASE_DIR"

# Check if directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist!"
    exit 1
fi

# List all contents of the directory
echo "Contents of directory:"
ls -la "$BASE_DIR"

# Find all progression directories
for prog_dir in "$BASE_DIR"/progression_*; do
    if [ -d "$prog_dir" ]; then
        echo "Processing progression directory: $prog_dir"
        
        # Get the example number from the directory name
        example_num=$(basename "$prog_dir" | sed 's/progression_//')
        output_video="$prog_dir/progression_video_${example_num}.mp4"
        
        echo "Looking for progression PNG files in: $prog_dir"
        
        # Check if progression PNG files exist
        if ls "$prog_dir"/progression_*.png 1> /dev/null 2>&1; then
            echo "Found progression PNG files, creating video..."
            # Use ffmpeg to create the video
            # -framerate 10 frames per second
            # -i progression_%d.png matches the progression_0.png, progression_1.png, etc. pattern
            # -c:v libx264 uses H.264 codec
            # -pix_fmt yuv420p ensures compatibility with most video players
            ffmpeg -y -framerate 10 -i "$prog_dir/progression_%d.png" \
                -c:v libx264 -pix_fmt yuv420p \
                "$output_video"
            
            # Check if video was created
            if [ -f "$output_video" ]; then
                echo "Successfully created video: $output_video"
            else
                echo "Failed to create video: $output_video"
            fi
        else
            echo "No progression PNG files found in $prog_dir"
        fi
    fi
done 