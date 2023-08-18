from ultralytics import YOLO
import cv2
import pyttsx3

# Load the YOLOv5 model
model = YOLO("yolov8s.pt")

# Load class labels
class_labels = model.names

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize the camera
cap = cv2.VideoCapture(1)# 0 indicates the default camera

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Perform object detection
    results = model.predict(frame)

    # Get detected labels from the first result
    first_result = results[0]  # Get the first result
    detected_labels = first_result.boxes.cls.cpu().numpy().astype(int)

    if len(detected_labels) > 0:
        # Get class names for detected labels
        detected_classes = [class_labels[label] for label in detected_labels]
        speech_text = ", ".join(detected_classes) + " detected"

        # Generate speech output
        engine.say(speech_text)
        engine.runAndWait()

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()