# Human Counting Project

This project uses **YOLOv8** and **OpenCV** to detect and count the number of people in real-time using a webcam or a video file.

## Features
- Real-time human detection and counting.
- Uses **YOLOv8** (You Only Look Once) for accurate object detection.
- Works with webcam or pre-recorded videos.
- Displays bounding boxes and count on the screen.

## Requirements

Ensure you have Python installed, then install dependencies:
```bash
pip install opencv-python numpy ultralytics torch
```

## Download YOLOv8 Model Weights

The script uses `yolov8n.pt` (small model). You can download a more accurate model like `yolov8s.pt` if needed.

## Usage

Run the script with the following command:
```bash
python human_counting.py
```

### Keyboard Controls
- **Press 'q'** to exit the program.

## Code Overview

```python
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open video capture (0 for webcam, or provide a video file path)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)
    
    # Count number of people detected
    count = sum(1 for obj in results[0].boxes.data if int(obj[5]) == 0)
    
    # Draw bounding boxes and labels
    for box in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = map(int, box)
        if cls == 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display count on screen
    cv2.putText(frame, f'Count: {count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Human Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## License

This project is open-source and can be used for educational and research purposes.

## Author
Ganesh Datta

