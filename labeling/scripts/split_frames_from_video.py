import cv2
from pathlib import Path
from PIL import Image
from io import BytesIO
import numpy as np
import shutil
import time

def image_to_bytes(image):
    """
    Converts an image from OpenCV format (a multi-dimension array) to bytes.
    Resizes to 1280x720, converts to PNG format, and returns the bytes.
    """
    # Converting the image to RGB colorspace
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    # Resize the image to 1280x720
    pil_image = pil_image.resize((1280, 720))

    # Convert to PNG bytes
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()

def add_legend_bar(frame, frame_rate=None, current_frame=None, total_frames=None):
    """
    Adds a legend bar to the bottom of the frame.
    Optionally adds framerate and frame position info on the right.
    Returns a new frame with the legend.
    """
    legend_text = "Legend: n=Next, f=Next 5, h=Skip 1/2s, j=Skip 1s, k=Skip 5s, s=Save, ESC=Exit"
    frame_with_legend = frame.copy()
    bar_height = 40
    # Draw a filled rectangle at the bottom
    cv2.rectangle(frame_with_legend, (0, 720-bar_height), (1280, 720), (50, 50, 50), -1)
    # Put the legend text with smaller font size
    cv2.putText(frame_with_legend, legend_text, (10, 715),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    # Add framerate and frame info on the right
    if frame_rate is not None and current_frame is not None and total_frames is not None:
        info_text = f"FPS: {frame_rate} | Frame: {current_frame}/{total_frames}"
        text_size, _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        text_x = 1280 - text_size[0] - 10
        cv2.putText(frame_with_legend, info_text, (text_x, 715),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)
    return frame_with_legend

# Making all of our paths to the various directories, starting with our labeling directory
base_path = Path(__file__).parent.parent
unlabeled_frames = base_path / "unlabeled_frames"
unprocessed_videos = base_path / "videos" / "unprocessed"
processed_videos = base_path / "videos" / "processed"
processed_videos.mkdir(parents=True, exist_ok=True)

# Getting a list of unprocessed video files. Exclude hidden files.
video_files = [
    f.name for f in unprocessed_videos.iterdir()
    if f.is_file() and not f.name.startswith('.')
]

# Checking if nothing is found in the unprocessed videos directory
if not video_files:
    print("Warning: No videos found in the directory:", unprocessed_videos)
    exit(1)

# Show video filenames with number labels
print("Select a video to process: \n_________________________________________")
for idx, name in enumerate(video_files):
    print(f"{idx}: {name}")

# User selects an index
selected_index = int(input("Enter the number of the video to process: "))
selected_video = video_files[selected_index]
selected_video_path = unprocessed_videos / selected_video

# Set window title to the selected video filename and create the window
window_title = selected_video  # e.g. "my_video.mp4"
cv2.namedWindow(window_title)

# Load the video using OpenCV
cap = cv2.VideoCapture(str(selected_video_path))
if not cap.isOpened():
    print(f"Error: Could not open video file {selected_video_path}")
    exit(1)
print(f"Video info successfully loaded for '{selected_video}'")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
 
print(f"  Length (frames): {frame_count}")
print(f"  Frame rate (FPS): {frame_rate}")

# Opening the cv2 frame and getting the keybinds ready
ret, frame = cap.read()
while True:
    # Checking if we have reached the end of the video
    current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if not ret or current_frame_num >= frame_count:
        # Building out end of video text
        print(f"End of video reached. Moving video. Press any key to exit.")
        end_frame = 255 * np.ones((720, 1280, 3), dtype=np.uint8)
        cv2.putText(end_frame, "End of video. Press any key to exit and move the video file.", (100, 360),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        end_frame = add_legend_bar(end_frame, frame_rate=frame_rate, current_frame=frame_count, total_frames=frame_count)

        # Displaying the end of frame text.
        cv2.imshow(window_title, end_frame)
        cv2.waitKey(0)

        # Move the processed video to the processed directory (avoid name collision)
        try:
            dest_path = processed_videos / selected_video_path.name
            if dest_path.exists():
                dest_path = processed_videos / f"{selected_video_path.stem}_{int(time.time())}{selected_video_path.suffix}"
            shutil.move(str(selected_video_path), str(dest_path))
            print(f"Moved video to processed: {dest_path}")
        except Exception as e:
            print(f"Warning: Failed to move video to processed directory: {e}")

        print("Exiting frame viewer.")
        break

    # Ensure display frame is 1280x720 for legend bar
    display_frame = cv2.resize(frame, (1280, 720))
    current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    display_frame = add_legend_bar(display_frame, frame_rate=frame_rate, current_frame=current_frame_num, total_frames=frame_count)
    cv2.imshow(window_title, display_frame)

    # Keeping track of key presses
    key = cv2.waitKey(0) & 0xFF

    if key == 27:  # ESC key to exit
        print("Exiting frame viewer.")
        break
    elif key == ord('s'):
        # Save the current frame as an image (original frame, not display_frame)
        frame_bytes = image_to_bytes(frame)
        frame_filepath = unlabeled_frames / "yolo" / f"{Path(selected_video).stem}_frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.png"
        with open(frame_filepath, 'wb') as img_file:
            img_file.write(frame_bytes)
        print(f"Saved frame to {frame_filepath}")

        # going to the next frame after saving
        ret, frame = cap.read()
        continue
    elif key == ord('n'):
        # Next frame
        ret, frame = cap.read()
        continue
    elif key in (ord('f'), ord('h'), ord('j'), ord('k')):
        # Skip frames based on key
        skip_map = {
            ord('f'): 5,
            ord('h'): int(frame_rate * 0.5),
            ord('j'): int(frame_rate * 1),
            ord('k'): int(frame_rate * 5),
        }
        target_frame = current_frame_num + skip_map[key]
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        continue
