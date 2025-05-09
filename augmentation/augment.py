import os
import random
from PIL import Image
from torchvision import transforms

# Input paths
image_dir = r"C:\Users\afif\Desktop\car_detection\yolov5\datasets\stanford_cars\images\train"
label_dir = r"C:\Users\afif\Desktop\car_detection\yolov5\datasets\stanford_cars\labels\train"

# Output paths
output_image_dir = r"C:\Users\afif\Desktop\car_detection\augmentation\images\augmented"
output_label_dir = r"C:\Users\afif\Desktop\car_detection\augmentation\labels\augmented"

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# Transformations
color_transform = transforms.ColorJitter(hue=0.2)

# Start saving from image name 08145.jpg
start_index = 8145

for i in range(1, 151):
    original_name = f"{i:05}"  # e.g., "00001"
    new_name = f"{start_index + i - 1:05}"  # e.g., "08145"

    image_path = os.path.join(image_dir, f"{original_name}.jpg")
    label_path = os.path.join(label_dir, f"{original_name}.txt")

    if not os.path.exists(image_path) or not os.path.exists(label_path):
        print(f"Skipping {original_name}: missing file.")
        continue

    # Load and augment image
    img = Image.open(image_path).convert("RGB")
    img = color_transform(img)

    do_flip = random.choice([True, False])
    if do_flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Save augmented image
    new_img_path = os.path.join(output_image_dir, f"{new_name}.jpg")
    img.save(new_img_path)

    # Adjust and save label
    new_label_lines = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls, x_center, y_center, box_w, box_h = parts
            x_center = float(x_center)

            if do_flip:
                x_center = 1.0 - x_center

            new_line = f"{cls} {x_center:.6f} {y_center} {box_w} {box_h}"
            new_label_lines.append(new_line)

    new_label_path = os.path.join(output_label_dir, f"{new_name}.txt")
    with open(new_label_path, "w") as f:
        f.write("\n".join(new_label_lines))

    print(f"Saved {new_name}.jpg and {new_name}.txt")
