import cv2
import os

# Input video file
video_path = 'mycarplate.mp4'  # Change to your video path

# Output directory for extracted frames
output_dir = 'extracted_frames'
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps

print(f"Video Info: {width}x{height} at {fps:.2f} FPS, {total_frames} frames total, {duration_sec:.2f} seconds duration")

# Calculate frame interval for 1 second captures
frame_interval = int(fps)  # Capture 1 frame per second
if frame_interval < 1:
    frame_interval = 1  # Ensure at least 1 frame if FPS < 1

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Capture one frame per second
    if frame_count % frame_interval == 0:
        # Get current timestamp in seconds
        current_time = frame_count / fps
        
        # Save frame with original dimensions
        frame_filename = os.path.join(output_dir, f"frame_{current_time:.1f}s.jpg")
        cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])  # 95% quality
        print(f"Saved {frame_filename} ({width}x{height}) at {current_time:.1f} seconds")
        saved_count += 1
    
    frame_count += 1

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Finished! Extracted {saved_count} frames (1 per second) to '{output_dir}'")