import cv2
from ultralytics import YOLO
import numpy as np
import easyocr 
from datetime import datetime
import os

# ===== Configuration =====
MODEL_PATH = 'anpr-demo-model.pt'
CLASSES_FILE = "images/classes.txt"
VIDEO_PATH = 'video.mp4'
OUTPUT_FILE = "car_plate_data.txt"
RESIZE_DIM = (900, 500)  # Width, Height
DETECTION_AREA = [(305, 1025), (1748, 1031), (1800, 1328), (276, 1296)]

# ===== Initialization =====
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=True)

# Load class names
with open(CLASSES_FILE, "r") as f:
    class_list = [c.strip() for c in f.readlines() if c.strip()]

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {VIDEO_PATH}")

# Create output file with header if not exists
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "w") as f:
        f.write("NumberPlate")

# Load previously saved plates
processed_plates = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        for line in f.readlines()[1:]:  # Skip header
            plate = line.strip()
            if plate:
                processed_plates.add(plate)

# ===== Functions =====
def preprocess_for_easyocr(img):
    """Preprocessing optimized for EasyOCR"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
######## Focus on choose the best filter and the best OCR ####

    # gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return gray

def clean_ocr_text(text):
    """Clean detected plate text for EasyOCR results"""
    # Remove all non-alphanumeric characters
    return ''.join(c for c in text if c.isalnum()).upper()

# ===== Main Processing =====
cv2.namedWindow('RGB')

frame_count = 0
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Scale detection area
scale_x = RESIZE_DIM[0] / original_width
scale_y = RESIZE_DIM[1] / original_height
scaled_detection_area = [(int(x * scale_x), int(y * scale_y)) for (x, y) in DETECTION_AREA]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.resize(frame, RESIZE_DIM)
    
    # Skip processing every 3 frames for performance
    if frame_count % 3 != 0:
        continue

    # YOLO Detection
    results = model.predict(frame, verbose=False)
    boxes = results[0].boxes.cpu().numpy()
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls_id = int(box.cls[0])
        
        # Check if detection is in our area of interest
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        if cv2.pointPolygonTest(np.array(scaled_detection_area, np.int32), center, False) >= 0:
            plate_img = frame[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue

            # Preprocess and save plate image
            processed = preprocess_for_easyocr(plate_img)
            PLATE_SAVE_DIR = "detected_plates"
            os.makedirs(PLATE_SAVE_DIR, exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(os.path.join(PLATE_SAVE_DIR, f"plate_{timestamp_str}.jpg"), plate_img)
            
            # EasyOCR processing
            try:
                ocr_results = reader.readtext(processed)
                if ocr_results:
                    # Get the result with highest confidence
                    best_result = max(ocr_results, key=lambda x: x[2])
                    text = clean_ocr_text(best_result[1])
                    
                    # Only process new plates
                    if text and text not in processed_plates:
                        processed_plates.add(text)
                        with open(OUTPUT_FILE, "a") as f:
                            f.write(f"{text}\n")
                        
                        # Visual feedback
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, text, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        cv2.imshow('Plate', plate_img)
            except Exception as e:
                print(f"OCR Error: {e}")

    # Draw detection area
    cv2.polylines(frame, [np.array(scaled_detection_area, np.int32)], True, (0,0,255), 2)
    
    # Display
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

# ===== Cleanup =====
cap.release()
cv2.destroyAllWindows()
print(f"Processing complete. Detected {len(processed_plates)} unique plates.")