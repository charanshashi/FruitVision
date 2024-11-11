import cv2
from ultralytics import YOLO
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Load YOLO model (replace with your model path if it's different)
    model = YOLO("yolov8n.pt")  # or your custom model like "yolov8-fruits.pt"
    
    # Keep a record of unique IDs
    unique_fruit_ids = set()
    total_fruit_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO inference
        results = model(frame)

        # Extract detections
        for result in results:
            boxes = result.boxes  # Get bounding boxes

            for box in boxes:
                # Extract class_id, confidence, and bounding box
                class_id = int(box.cls[0])  # Get class ID
                confidence = float(box.conf[0])  # Get confidence score
                bbox = box.xyxy[0].tolist()  # Bounding box in (x1, y1, x2, y2) format

                # Check if detection is a fruit (assuming class_id for fruit is known, e.g., `FRUIT_CLASS_ID`)
                if class_id == 0:  
                    # Create a unique identifier for each detection
                    detection_id = hash((class_id, bbox[0], bbox[1], bbox[2], bbox[3]))
                    if detection_id not in unique_fruit_ids:
                        unique_fruit_ids.add(detection_id)
                        total_fruit_count += 1

                    # Draw bounding box and label
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f"Fruit {total_fruit_count}", (int(bbox[0]), int(bbox[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display total count
        cv2.putText(frame, f"Total Fruits Counted: {total_fruit_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Show frame
        cv2.imshow("Fruit Counting", frame)

        # Exit on Escape key
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
