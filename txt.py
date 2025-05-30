# Old version
import cv2
import time
from ultralytics import YOLO
import serial

# Load YOLO classification model
model = YOLO("bestv2.pt")  # Replace with your model path
print("Model classes:", model.names)

# Map specific raw class labels to grouped categories
label_mapping = {
    "glass": "trash",
    "plastic": "trash",
    "biological": "trash",
    "trash": "trash",
    "paper": "paper",
    "cardboard": "paper",
    "metal": "metal",
    "battery": "metal"
}

# Define characters to send over serial for each group
class_to_char = {
    "trash": b'T',
    "metal": b'M',
    "paper": b'P'
}
""" # COMMENT / UNCOMMENT WHEN NEEDED
# Setup serial communication with Arduino
ser = serial.Serial('COM5', 9600)  # Replace 'COM5' with the correct port if needed
time.sleep(2)  # Allow time for Arduino to initialize
"""

# Initialize webcam (0 = default camera)
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

        # Run classification on the frame
        results = model(frame)
        top_idx = int(results[0].probs.top1)
        confidences = results[0].probs.data.tolist()
        confidence = confidences[top_idx]
        raw_label = model.names[top_idx]

        # Use mapped label if confidence is high enough, else default to "trash"
        if confidence < 0.25:
            label = "trash"
            print(f"Low confidence ({confidence:.2f}) → Defaulting to: trash")
        else:
            label = label_mapping.get(raw_label, "unknown")
            print(f"Predicted: {raw_label} ({confidence:.2f}) → Grouped as: {label}")

        # Show label and confidence on frame
        display_text = f"{label} ({confidence:.2f})"
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Live Classification", frame)

        # Send character to Arduino if valid label
        if label in class_to_char:
            #ser.write(class_to_char[label]) # COMMENT / UNCOMMENT WHEN NEEDED
            print(f"Sent to Arduino: {class_to_char[label].decode()}")

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Throttle inferences (one every ~3 seconds)
        time.sleep(3)

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    cap.release()
    #ser.close()
    cv2.destroyAllWindows()
