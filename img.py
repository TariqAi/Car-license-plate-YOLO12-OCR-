import cv2
import time

# Initialize
cpt = 0
maxFrames = 85  # Max frames to capture
cap = cv2.VideoCapture('video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)  # Get video FPS
skip_frames = int(fps * 0.2)  # Skip frames to get 0.5s intervals
count = 0

while cpt < maxFrames:
    ret, frame = cap.read()
    if not ret:
        break
    
    count += 1
    if count % skip_frames != 0:  # Skip frames to get 0.5s intervals
        continue
    
    frame = cv2.resize(frame, (2160, 1462))
    cv2.imshow("Frame", frame)
    cv2.imwrite(r"C:\Users\HP\Desktop\Car_plate\images\numberplate_%d.jpg" % cpt, frame)
    cpt += 1
    
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'Esc'
        break

cap.release()
cv2.destroyAllWindows()