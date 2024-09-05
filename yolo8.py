from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Define video path and output directory
video_path = "vid3.mp4"
output_dir = "frames"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video file
video = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Count and label detected people
    people_count = 0
    for result in results:
        # Access detection boxes
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0].item())
            if class_id == 0:  # Class ID 0 is 'person'
                people_count += 1
                xyxy = box.xyxy[0].cpu().numpy()  # Convert tensor to numpy array
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"Frame {frame_count:04d}: People count = {people_count}")

    # Save frame with annotations
    frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_path, frame)

    frame_count += 1

video.release()
cv2.destroyAllWindows()
