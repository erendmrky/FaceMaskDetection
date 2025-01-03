import tensorflow as tf
import cv2
import numpy as np
import os

def data_generator(images_dir, annotations_dir, batch_size=16, input_shape=(300, 300), num_anchors=360000, num_classes=3):
    import os
    import numpy as np
    import cv2

    files = os.listdir(images_dir)
    num_samples = len(files)

    while True:
        for i in range(0, num_samples, batch_size):
            batch_images = []
            batch_targets = []

            for j in range(i, min(i + batch_size, num_samples)):
                # Load and preprocess the image
                image_path = os.path.join(images_dir, files[j])
                image = cv2.imread(image_path)
                image = cv2.resize(image, input_shape[:2])
                image = image / 255.0  # Normalize to [0, 1]
                batch_images.append(image)

                # Generate a target tensor for 360,000 anchors
                targets = np.zeros((num_anchors, num_classes + 4))  # [num_anchors, num_classes + 4]

                # Example: Assign class "1" and a bounding box to anchor 0
                class_one_hot = [0] * num_classes
                class_one_hot[1] = 1  # Example: Class "1"
                targets[0, :num_classes] = class_one_hot
                targets[0, num_classes:] = [50, 50, 150, 150]  # Example bbox (xmin, ymin, xmax, ymax)

                batch_targets.append(targets)

            yield np.array(batch_images), np.array(batch_targets)

if __name__ == "__main__":
    gen = data_generator('ssd_dataset/train/images', 'ssd_dataset/train/annotations', batch_size=2)
    for batch_images, (batch_boxes, batch_labels) in gen:
        print(f"Batch Images: {batch_images.shape}, Boxes: {batch_boxes.shape}, Labels: {batch_labels.shape}")
        break
