import cv2
import numpy as np
import pyttsx3
import time
import threading
from ultralytics import YOLO

# Load YOLOv8-Nano model
model = YOLO("yolov8n.pt")


# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# Start webcam
cap = cv2.VideoCapture(1)

# Global variables
last_speak_time = time.time()
obstacle_memory = []
lock = threading.Lock()

# Display settings
frame_width, frame_height = 320, 240
grid_size = (24, 32)  # More detailed grid
grid_counts = np.zeros(grid_size, dtype=np.int32)  # Track object density


def speak(text):
    """Speak detected objects & navigation commands safely."""
    global last_speak_time
    if time.time() - last_speak_time >= 3:
        try:
            print(f"Saying: {text}")
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Speech Error: {e}")  # Debugging error message
        last_speak_time = time.time()


def find_best_path(detected_objects, frame_width):
    left_blocked = False
    right_blocked = False
    center_blocked = False
    closest_object = None
    min_distance = float('inf')

    for obj in detected_objects:
        if len(obj) == 5:  # Ensure label is included
            x1, y1, x2, y2, label = obj
        else:
            continue  # Skip if the object data is incomplete

        obj_center = (x1 + x2) // 2  # Middle point of object
        distance = y2  # The lower the y2, the closer the object

        # Track the closest object
        if distance < min_distance:
            min_distance = distance
            closest_object = label

        # Determine where the object is
        if obj_center < frame_width // 3:
            left_blocked = True
        elif obj_center > 2 * frame_width // 3:
            right_blocked = True
        else:
            center_blocked = True

    # Decide movement based on the blockages
    if center_blocked:
        if left_blocked and right_blocked:
            return f"Stop, {closest_object} ahead!"
        elif left_blocked:
            return "Move Right carefully"
        elif right_blocked:
            return "Move Left carefully"
        else:
            return f"Stop, {closest_object} ahead!"
    return "Clear path, move forward"


def run_detection():
    """Runs YOLO detection in a separate thread."""
    global grid_counts
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (frame_width, frame_height))
        results = model(frame_resized, conf=0.4, iou=0.45)

        detected_objects = set()
        new_obstacles = []
        grid_counts.fill(0)  # Reset grid

        # Create a blank grid display
        grid_display = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])  # Confidence score
                cls = int(box.cls[0])  # Class index
                label = model.names[cls]  # Get object label

                detected_objects.add(label)
                new_obstacles.append((x1, y1, x2, y2, label))

                # Draw bounding boxes
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_resized, f"{label} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Update grid display for visualization
                gx1, gy1 = int(x1 / frame_width * grid_size[1]), int(y1 / frame_height * grid_size[0])
                gx2, gy2 = int(x2 / frame_width * grid_size[1]), int(y2 / frame_height * grid_size[0])
                grid_counts[gy1:gy2 + 1, gx1:gx2 + 1] += 1

                # Draw grid locations (Debugging visualization)
                cv2.rectangle(grid_display, (gx1 * 10, gy1 * 10), (gx2 * 10, gy2 * 10), (0, 0, 255), 1)

        # Determine best movement based on detected objects
        best_movement = find_best_path(new_obstacles, frame_resized.shape[1])
        print(f"Best movement: {best_movement}")

        # Store obstacles in memory
        obstacle_memory.append(new_obstacles)
        if len(obstacle_memory) > 10:
            obstacle_memory.pop(0)

        # Show the detection window
        cv2.imshow("Object Detection", frame_resized)
        cv2.imshow("Grid Visualization", grid_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Start detection in a background thread
threading.Thread(target=run_detection, daemon=True).start()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (frame_width, frame_height))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)

    # Fix: Pass detected obstacles instead of edges
    path_direction = find_best_path(obstacle_memory[-1] if obstacle_memory else [], frame_resized.shape[1])

    # Display navigation info
    cv2.putText(frame_resized, path_direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Speak detected objects & movement
    if obstacle_memory:
        speak(f"Detected: {', '.join(set([obj[4] for obj in obstacle_memory[-1]]))}. {path_direction}")
    else:
        speak(path_direction)

    # Show the edge detection window
    cv2.imshow("Edge Detection (Obstacle Mapping)", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
