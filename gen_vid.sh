#!/bin/bash

# Path to the processed recon folder
recon_dir="./processed_recon"
videos_dir="$(dirname "$recon_dir")/recon_videos"

# Create the video directory if it doesn't exist
mkdir -p "$videos_dir"
counter=0
# Loop over each trajectory folder
for traj_folder in "$recon_dir"/*/; do
    # Ensure it's a directory
    if [ -d "$traj_folder" ]; then
	((counter++))
	if [ "$counter" -lt 6158 ]; then
		continue
	fi
	traj_name=$(basename "$traj_folder")
        traj_video_dir="$videos_dir/$traj_name"
        output_video="$traj_video_dir/video.mp4"

        # Create subdirectory for the trajectory
        mkdir -p "$traj_video_dir"

        # Check if there are image files
        if ls "$traj_folder"/*.jpg 1> /dev/null 2>&1; then
            echo "Creating video for $traj_name..."

            # Sort images numerically and pass them to ffmpeg
            ffmpeg -framerate 10 -pattern_type glob -i "$traj_folder/*.jpg" \
                   -c:v libx264 -pix_fmt yuv420p "$output_video"

            if [ $? -eq 0 ]; then
                echo "Video created: $output_video"
            else
                echo "FFmpeg failed for $traj_name"
            fi
        else
            echo "No images found in $traj_folder. Skipping..."
        fi

        # Copy traj_data.pkl if it exists
        if [ -f "$traj_folder/traj_data.pkl" ]; then
            cp "$traj_folder/traj_data.pkl" "$traj_video_dir/"
            echo "Copied traj_data.pkl to $traj_video_dir/"
        else
            echo "traj_data.pkl not found in $traj_folder. Skipping copy."
        fi
    fi
done

