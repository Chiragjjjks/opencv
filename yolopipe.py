from ultralytics import YOLO
import cv2
import os
import numpy as np
from IPython.display import display, clear_output
from PIL import Image

# Step 1: Setup and Initialization
def initialize_pipeline(model_path, video_path, output_path):
    model = YOLO(model_path)
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    return model, video, out, frame_width, frame_height

# Step 2: Process Frames
def process_frames(model, video, out, frame_width, frame_height):
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

        # Convert frame to RGB and then to PIL image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Display the frame using IPython display
        clear_output(wait=True)
        display(pil_image)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Step 3: Finalization
def finalize_pipeline(video, out):
    video.release()
    out.release()
    cv2.destroyAllWindows()

# Main function to run the pipeline
def main():
    model_path = "yolov8n.pt"
    video_path = "vid3.mp4"
    output_path = "output2.mp4"

    model, video, out, frame_width, frame_height = initialize_pipeline(model_path, video_path, output_path)
    process_frames(model, video, out, frame_width, frame_height)
    finalize_pipeline(video, out)

if __name__ == "__main__":
    main()
