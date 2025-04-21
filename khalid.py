import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re

# ---- Config ----
video_path = 'video.mp4'
output_txt_path = 'license_plates_processed.txt'
roi_points = np.array([[1823, 1042], [300, 800], [269, 1226], [1864, 1212]])

# ---- Load YOLOv8 Model ----
model = YOLO("Yolo_v8.pt")
model.conf = 0.5

# ---- Initialize PaddleOCR ----
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

# ---- Video Input Setup ----
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(3)), int(cap.get(4)))

# ---- Store Detected Plates for Post-Processing ----
detected_plates = []  # Store raw plates before processing

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    # Get the shape of the frame
    height, width = frame.shape[:2]

    # Create a blank single-channel mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill the ROI in the mask with white (255)
    cv2.fillPoly(mask, [roi_points], 255)

    frame_num += 1
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    results = model(masked_frame)

    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy().astype(int)

        for box in boxes:
            x1, y1, x2, y2 = box
            cropped = frame[y1:y2, x1:x2]

            # ---- Preprocess for OCR ----
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            contrast = clahe.apply(gray)
            clean = cv2.bilateralFilter(contrast, 9, 75, 75)
            cropped_resized = cv2.resize(clean, (300, 100), interpolation=cv2.INTER_CUBIC)
            cropped_resized = cv2.cvtColor(cropped_resized, cv2.COLOR_GRAY2BGR)

            # ---- OCR ----
            ocr_results = ocr.ocr(cropped_resized, cls=True)
            best_text = ""
            best_conf = 0

            if ocr_results and isinstance(ocr_results, list):
                for result_line in ocr_results:
                    if result_line:
                        for res in result_line:
                            if res and len(res) >= 2:
                                text, confidence = res[1][0], res[1][1]
                                if confidence > best_conf:
                                    best_text = text
                                    best_conf = confidence

            # ---- Store Plate if Confidence ≥ 0.95 ----
            if best_conf >= 0.95:
                detected_plates.append(best_text)
                print(f"Detected on Frame {frame_num}: {best_text} (conf: {best_conf:.2f})")

# ---- Post-Processing Steps ----
def process_plates(plates):
    # Step 1: Remove symbols (keep only alphanumeric)
    cleaned_plates = [re.sub(r'[^a-zA-Z0-9]', '', plate) for plate in plates]

    # Step 2: Remove duplicates (keep first occurrence)
    unique_plates = []
    seen = set()
    for plate in cleaned_plates:
        if plate not in seen:
            seen.add(plate)
            unique_plates.append(plate)

    # Step 3: Merge two-digit plates with the next plate
    merged_plates = []
    i = 0
    while i < len(unique_plates):
        current_plate = unique_plates[i]
        if len(current_plate) == 2 and i + 1 < len(unique_plates):
            next_plate = unique_plates[i + 1]
            merged_plate = current_plate + next_plate
            merged_plates.append(merged_plate)
            i += 2  # Skip next plate since it's merged
        else:
            merged_plates.append(current_plate)
            i += 1

    # return merged_plates

    hyphenated_plates = []
    for plate in merged_plates:
        if len(plate) > 2:
            plate = f"{plate[:2]}-{plate[2:]}"
            hyphenated_plates.append(plate)

    return hyphenated_plates

# ---- Apply Post-Processing ----
final_plates = process_plates(detected_plates)

# ---- Save Processed Results ----
with open(output_txt_path, 'w') as f:
    for plate in final_plates:
        f.write(f"{plate}\n")

# ---- Clean up ----
cap.release()
print("✅ Processing complete. Output saved to 'license_plates_processed.txt'.")