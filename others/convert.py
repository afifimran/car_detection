import os
import pandas as pd
from PIL import Image

# === Set your paths ===
csv_path = "anno_train.csv"
images_dir = "datasets/stanford_car/images/train"      # Change to your actual image path
labels_dir = "datasets/stanford_car/labels/train"      # Output directory for YOLO labels
os.makedirs(labels_dir, exist_ok=True)

# === Load annotations ===
df = pd.read_csv(csv_path)

# === Convert to YOLO format ===
for i, row in df.iterrows():
    img_path = os.path.join(images_dir, row["filename"])
    label_path = os.path.join(labels_dir, row["filename"].replace(".jpg", ".txt"))

    if not os.path.exists(img_path):
        print(f"Warning: Image not found: {img_path}")
        continue

    img = Image.open(img_path)
    width, height = img.size

    # Convert bbox to YOLO format
    x_center = ((row["xmin"] + row["xmax"]) / 2) / width
    y_center = ((row["ymin"] + row["ymax"]) / 2) / height
    bbox_width = (row["xmax"] - row["xmin"]) / width
    bbox_height = (row["ymax"] - row["ymin"]) / height

    class_id = row["class"]  # Optionally remap if needed

    # Save to label file
    with open(label_path, "a") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
