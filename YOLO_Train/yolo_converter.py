import os
import xml.etree.ElementTree as ET

def convert_to_yolo(xml_file, img_width, img_height, class_dict):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    yolo_annotations = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in class_dict:
            continue
        class_id = class_dict[class_name]

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Normalize bounding box coordinates
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

    return yolo_annotations

def process_dataset(xml_dir, output_dir, class_dict):
    os.makedirs(output_dir, exist_ok=True)

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_width = int(root.find("size/width").text)
        img_height = int(root.find("size/height").text)

        yolo_annotations = convert_to_yolo(xml_path, img_width, img_height, class_dict)
        
        # Save YOLO annotations
        output_path = os.path.join(output_dir, xml_file.replace(".xml", ".txt"))
        with open(output_path, "w") as f:
            f.writelines(yolo_annotations)

# Example usage
class_dict = {"without_mask": 0, "with_mask": 1, "mask_weared_incorrect": 2}
xml_dir = "dataset/annotations"
output_dir = "yolo/annotations"

# process_dataset(xml_dir, output_dir, class_dict)

import os
import shutil
import random

# Paths
image_folder = "dataset/images"  # Path to the images folder
annotation_folder = "yolo/annotations"  # Path to the YOLO annotation files
output_folder = "organized_dataset"  # Path to the new dataset folder

# Ratios for splitting
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create output folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_folder, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, split, "labels"), exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))]

# Shuffle images for random splits
random.shuffle(image_files)

# Calculate split sizes
total_images = len(image_files)
train_count = int(total_images * train_ratio)
val_count = int(total_images * val_ratio)

# Split the dataset
train_files = image_files[:train_count]
val_files = image_files[train_count:train_count + val_count]
test_files = image_files[train_count + val_count:]

# Function to move files
def move_files(file_list, split):
    for image_file in file_list:
        # Image paths
        src_image = os.path.join(image_folder, image_file)
        dest_image = os.path.join(output_folder, split, "images", image_file)
        
        # Annotation paths
        annotation_file = os.path.splitext(image_file)[0] + ".txt"
        src_annotation = os.path.join(annotation_folder, annotation_file)
        dest_annotation = os.path.join(output_folder, split, "labels", annotation_file)

        # Move image
        if os.path.exists(src_image):
            shutil.copy(src_image, dest_image)
        
        # Move annotation
        if os.path.exists(src_annotation):
            shutil.copy(src_annotation, dest_annotation)

# Move files to train, val, and test folders
move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print(f"Dataset organized successfully into {output_folder}.")
print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")


