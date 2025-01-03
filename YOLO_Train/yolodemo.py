import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("best.pt")  # Replace with the correct path to your weights

# Open the webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Resize the frame to match YOLO's expected input size
    resized_frame = cv2.resize(frame, (416, 416))

    # Perform inference
    results = model.predict(resized_frame, conf=0.5)  # Confidence threshold of 0.5

    # Annotate frame with detection results
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLO Real-Time Mask Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
