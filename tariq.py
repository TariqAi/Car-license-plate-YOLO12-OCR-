import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import re

# ---- Config ----
video_path = 'video.mp4'
output_video_path = 'output_video.mp4'
output_txt_path = 'license_plates.txt'
roi_points = np.array([[436, 234], [1653, 274], [1824, 1071], [293, 1020]])
# roi_points = np.array([[351, 412], [1679, 461], [1737, 755], [368, 700]])


# ---- Load YOLOv8 Model ----
model = YOLO("Yolo_v8.pt")  # yolov8 format
model.conf = 0.5  # Confidence threshold

# ---- Initialize PaddleOCR ----
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

# ---- Video Input/Output Setup ----
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(3)), int(cap.get(4)))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

# ---- Create ROI Mask ----
roi_mask = np.zeros((frame_size[1], frame_size[0]), dtype=np.uint8)
cv2.fillPoly(roi_mask, [roi_points], 255)

# ---- Output Text File Setup ----
text_output = open(output_txt_path, 'w')

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    # masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
    # results = model(masked_frame)
    results = model(frame)


    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy().astype(int)

        for box in boxes:
            x1, y1, x2, y2 = box
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            if roi_mask[center_y, center_x] != 255:
                continue


            # ---- Crop Detected License Plate ----
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

            # ---- Show Detection & Save to File (Only if Confidence ≥ 0.95) ----
            if best_conf >= 0.95:  # Changed from 0.3 to 0.95
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, best_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # text_output.write(f"Frame {frame_num}: {best_text} (conf: {best_conf:.2f})\n")
                text_output.write(f"{best_text}\n")

                print(f"Detected on Frame {frame_num}: {best_text} (conf: {best_conf:.2f})")

    # Draw ROI
    cv2.polylines(frame, [roi_points], isClosed=True, color=(255, 0, 0), thickness=2)

    # Save processed frame
    out.write(frame)

# ---- Clean up ----
cap.release()
out.release()
text_output.close()

print("✅ Processing complete. Outputs saved.")
