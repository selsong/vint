#!/bin/bash

# Base directory containing the progression images
BASE_DIR="/bigdata/selina/vint_release/train/logs/vint_test/log_run_1_2025_03_19_16_26_27/visualize/progression"

echo "Looking for progression images in: $BASE_DIR"

# Check if directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist!"
    exit 1
fi

# List all contents of the directory
echo "Contents of directory:"
ls -la "$BASE_DIR"

# Find all epoch directories
for epoch_dir in "$BASE_DIR"/epoch*; do
    if [ -d "$epoch_dir" ]; then
        echo "Found epoch directory: $epoch_dir"
        # Find all trajectory directories within this epoch
        for traj_dir in "$epoch_dir"/*; do
            if [ -d "$traj_dir" ]; then
                # Get the trajectory name from the directory path
                traj_name=$(basename "$traj_dir")
                # Create output video path
                output_video="$traj_dir/${traj_name}_progression.mp4"
                
                echo "Processing trajectory: $traj_name"
                echo "Looking for PNG files in: $traj_dir"
                
                # Check if PNG files exist
                if ls "$traj_dir"/frame_*.png 1> /dev/null 2>&1; then
                    echo "Found PNG files, creating video..."
                    # Use ffmpeg to create the video
                    # -framerate 2 means 2 frames per second
                    # -i frame_%03d.png matches the frame_000.png, frame_001.png, etc. pattern
                    # -c:v libx264 uses H.264 codec
                    # -pix_fmt yuv420p ensures compatibility with most video players
                    ffmpeg -y -framerate 2 -i "$traj_dir/frame_%03d.png" \
                        -c:v libx264 -pix_fmt yuv420p \
                        "$output_video"
                    
                    # Check if video was created
                    if [ -f "$output_video" ]; then
                        echo "Successfully created video: $output_video"
                    else
                        echo "Failed to create video: $output_video"
                    fi
                else
                    echo "No PNG files found in $traj_dir"
                fi
            fi
        done
    else
        echo "No epoch directories found in $BASE_DIR"
    fi
done 