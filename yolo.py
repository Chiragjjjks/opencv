from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Define video paths
video_path = "vid2.mp4"
output_path = "output.mp4"

# Open the video file
video = cv2.VideoCapture(video_path)

# Get video properties
fps = video.get(cv2.CAP_PROP_FPS)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize VideoWriter to save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Count detected people
    people_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0].item())
            if class_id == 0:  # Class ID 0 is 'person'
                people_count += 1
                xyxy = box.xyxy[0].cpu().numpy()  # Convert tensor to numpy array
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display count on the frame with better visibility
    count_text = f"People Count: {people_count}"
    cv2.putText(frame, count_text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Write the frame to the output video file
    out.write(frame)

    # Display the frame
    cv2.imshow("Detected Video", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video objects
video.release()
out.release()
cv2.destroyAllWindows()
