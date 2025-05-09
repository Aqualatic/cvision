import cv2
import time
from ultralytics import YOLO
import serial

# Load pretrained YOLOv8n detection model (trained on COCO dataset)
model = YOLO("yolov8n.pt") # Change path to the model
print("Model loaded with classes:", model.names)

# Map detected COCO classes to general material groups
# NOTE: COCO does not have "metal" or "paper" directly, so we infer from object type
coco_to_group = {
    # PAPER-like objects
    "book": "paper",
    "newspaper": "paper",  # not in COCO, but leaving for custom additions
    "toilet": "paper",  # often toilet paper (if applicable)

    # METAL-like objects
    "scissors": "metal",
    "fork": "metal",
    "knife": "metal",
    "spoon": "metal",
    "remote": "metal",
    "cell phone": "metal",  # likely includes aluminum/copper
    "laptop": "metal",
    "keyboard": "metal",
    "microwave": "metal",
    "refrigerator": "metal",
    "oven": "metal",
    "toaster": "metal",

    # TRASH-like objects (plastics, food waste, ambiguous)
    "bottle": "trash",  # assumed plastic
    "cup": "trash",
    "banana": "trash",
    "apple": "trash",
    "orange": "trash",
    "broccoli": "trash",
    "carrot": "trash",
    "sandwich": "trash",
    "hot dog": "trash",
    "pizza": "trash",
    "donut": "trash",
    "cake": "trash",
    "toothbrush": "trash",
    "mouse": "trash",
    "tv": "trash",  # mixed waste
    "handbag": "trash",
    "backpack": "trash",
    "suitcase": "trash",
    "tie": "trash",
    "umbrella": "trash"
}

# Characters to send over serial
class_to_char = {
    "trash": b'T',
    "metal": b'M',
    "paper": b'P'
}

# ----- SERIAL SETUP -----
# ser = serial.Serial('COM5', 9600)  # Uncomment and set correct COM port
# time.sleep(2)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model(frame)
        detections = results[0].boxes
        label = "trash"  # default to trash
        selected_label = None

        # Go through detections and find first mappable class
        for box in detections:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            mapped_group = coco_to_group.get(class_name, None)
            if mapped_group:
                selected_label = mapped_group
                print(f"Detected: {class_name} → Grouped as: {mapped_group}")
                break  # just use first relevant detection

        if selected_label is None:
            print("No relevant object found → Defaulting to: trash")
            selected_label = "trash"

        # Show result
        display_text = f"{selected_label}"
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Live Detection", frame)

        # Send to Arduino
        if selected_label in class_to_char:
            # ser.write(class_to_char[selected_label])  # Uncomment to send
            print(f"Sent to Arduino: {class_to_char[selected_label].decode()}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(3)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    cap.release()
    # ser.close()  # Uncomment if serial used
    cv2.destroyAllWindows()
