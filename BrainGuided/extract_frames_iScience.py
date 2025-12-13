import os
import cv2
from tqdm import tqdm

# Set the input and output directories
input_dir = "iScience--raw" # VideoEmotion8--raw, EK6--raw
output_dir = "iScience--imgs" # VideoEmotion8--imgs, EK6--imgs

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get the total number of videos to process
total_videos = 0
for root, dirs, files in os.walk(input_dir):
    for file_name in files:
        if file_name.endswith(".mp4") or file_name.endswith(".avi"):
            total_videos += 1

# Create a progress bar for the total processs
total_frames = 0
pbar = tqdm(total=total_videos * 10 * 16, desc="Processing videos", unit="frame")

# Loop through the input directory and its subdirectories
for root, dirs, files in os.walk(input_dir):
    for file_name in files:
        if file_name.endswith(".mp4") or file_name.endswith(".avi"):
            # Get the video name without the extension
            video_name = os.path.splitext(file_name)[0]
            
            # Create the output directory for the video
            video_output_dir = os.path.join("iScience--imgs/iScience", video_name)
            if not os.path.exists(video_output_dir):
                os.makedirs(video_output_dir)
            
            # Load the video file
            video_path = os.path.join(root, file_name)
            cap = cv2.VideoCapture(video_path)
            
            # Get the total number of frames in the video
            total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Divide the video into 10 segments and sample 16 frames from each segment
            for segment in range(10):
                start_frame = int(segment * (total_frames_in_video / 10))
                end_frame = int((segment + 1) * (total_frames_in_video / 10))
                
                # Sample 16 frames from the segment
                for i in range(16):
                    frame_index = int(start_frame + i * (end_frame - start_frame) / 15)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = cap.read()
                    if ret:
                        # Save the frame to the output directory
                        output_file_name = '{:06d}.jpg'.format(total_frames+1)
                        output_file_path = os.path.join(video_output_dir, output_file_name)
                        cv2.imwrite(output_file_path, frame)
                        
                        # Update the progress bar
                        pbar.update(1)
                        total_frames += 1
            
            # Release the video capture object
            cap.release()

            # Save the number of frames to a text file
            with open(os.path.join(video_output_dir, "n_frames"), "w") as f:
                f.write(f"{total_frames}")

            total_frames = 0

# Close the progress bar
pbar.close()