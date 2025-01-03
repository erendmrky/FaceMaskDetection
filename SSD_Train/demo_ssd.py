import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained SSD model
model = load_model('ssd_model.h5', compile=False)

# Preprocess input image for SSD model
def preprocess_image(image, input_shape=(300, 300)):
    """
    Resize and normalize an image for SSD model input.

    Args:
        image (np.array): Input image.
        input_shape (tuple): Target input size for the SSD model.

    Returns:
        np.array: Preprocessed image ready for inference.
    """
    image_resized = cv2.resize(image, input_shape)
    image_normalized = image_resized / 255.0  # Normalize pixel values
    return np.expand_dims(image_normalized, axis=0)  # Add batch dimension

# Draw predictions on the image
def draw_predictions(image, predictions, threshold=0.5, input_shape=(300, 300)):
    """
    Draw bounding boxes and labels on the image.

    Args:
        image (np.array): Original image.
        predictions (np.array): SSD model predictions.
        threshold (float): Confidence threshold for predictions.
        input_shape (tuple): Target input shape used during training.

    Returns:
        np.array: Annotated image.
    """
    height, width, _ = image.shape
    scale_x, scale_y = width / input_shape[0], height / input_shape[1]

    for pred in predictions:
        # Extract class probabilities and bounding box
        class_probs = pred[:3]  # Assuming 3 classes
        bbox = pred[3:]  # [xmin, ymin, xmax, ymax]

        # Get the predicted class and confidence
        class_id = np.argmax(class_probs)
        confidence = class_probs[class_id]

        if confidence > threshold:
            # Scale bounding box to original image size
            xmin, ymin, xmax, ymax = bbox
            xmin, xmax = int(xmin * scale_x), int(xmax * scale_x)
            ymin, ymax = int(ymin * scale_y), int(ymax * scale_y)

            # Draw bounding box and label
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"Class {class_id} ({confidence:.2f})"
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Run webcam demo
def run_demo():
    """
    Open webcam and display real-time SSD detections.
    """
    cap = cv2.VideoCapture(0)  # Open webcam (index 0)
    input_shape = (300, 300)  # Input size for the SSD model

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for the SSD model
        input_image = preprocess_image(frame, input_shape)

        # Perform inference
        predictions = model.predict(input_image)[0]  # Remove batch dimension

        print("Predict finished")

        # Draw predictions on the frame
        annotated_frame = draw_predictions(frame, predictions, threshold=0.5, input_shape=input_shape)


        # Display the frame
        cv2.imshow('SSD Detection Demo', annotated_frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the demo
if __name__ == "__main__":
    run_demo()
