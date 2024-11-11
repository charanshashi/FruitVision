import cv2
from ultralytics import YOLO
import numpy as np

# Define the fruit class ID (adjust as per your model's output)
FRUIT_CLASS_ID = 47

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#47 apple 48 orange 46 banana
    model = YOLO("yolov8n.pt")  # Use your custom YOLO model for fruit detection
    print(model.model.names)
    total_fruit_count = 0
    unique_fruit_ids = set()  # To keep track of unique fruits
    trackers = []  # List to store individual trackers
    tracker_objects = []  # List to store tracked bounding boxes
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # YOLO model inference

        detections = []  # To store bounding boxes for tracking

        for result in results:
            boxes = result.boxes  # Get bounding boxes

            for box in boxes:
                class_id = int(box.cls[0])  # Get class ID
                confidence = float(box.conf[0])  # Get confidence score
                bbox = box.xyxy[0].tolist()  # Bounding box in (x1, y1, x2, y2) format

                if class_id == FRUIT_CLASS_ID:
                    # Convert bounding box to (x, y, width, height) format
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    bbox_converted = (int(x1), int(y1), int(width), int(height))

                    # Generate a unique detection ID based on the bounding box (or use another method)
                    detection_id = hash(tuple(bbox_converted))
                    
                    if detection_id not in unique_fruit_ids:
                        # Add this detection to the set and start tracking the object
                        unique_fruit_ids.add(detection_id)
                        tracker = cv2.TrackerCSRT_create()  # Create a new tracker
                        trackers.append(tracker)  # Append the tracker to the list
                        tracker.init(frame, tuple(bbox_converted))  # Initialize the tracker with the bounding box
                        total_fruit_count += 1

        # Update all trackers
        for i, tracker in enumerate(trackers):
            success, tracked_bbox = tracker.update(frame)

            if success:
                x, y, w, h = [int(v) for v in tracked_bbox]
                tracker_objects.append((x, y, w, h))  # Store tracked bounding boxes
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Fruit {total_fruit_count}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display total count
        cv2.putText(frame, f"Total Fruits Counted: {total_fruit_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show frame
        cv2.imshow("Fruit Tracking", frame)

        # Exit on Escape key
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
