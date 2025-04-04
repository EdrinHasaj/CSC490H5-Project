import os

# Correct dataset path
dataset_path = "/Users/edrinhasaj/Desktop/CSC490H5/chestxray"

# Find all image directories
image_dirs = [os.path.join(dataset_path, d, "images") for d in os.listdir(dataset_path) if d.startswith("images_")]

print("Image directories found:", image_dirs)

# Collect image file paths
all_images = []
for img_dir in image_dirs:
    if os.path.isdir(img_dir):  # Check if it's a valid directory
        img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if img_files:  # Only add if images exist
            all_images.extend(img_files)

print(f"Total images found: {len(all_images)}")
print("Sample images:", all_images[:5])  # Show a few sample images

import cv2
from tqdm import tqdm

# Define output folder for preprocessed images
output_folder = os.path.join(dataset_path, "preprocessed_images")
os.makedirs(output_folder, exist_ok=True)

def preprocess_images(image_paths, output_dir, img_size=(224, 224)):
    for img_path in tqdm(image_paths):
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, img_name)

        # Load and preprocess image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size)

        # Save preprocessed image
        cv2.imwrite(output_path, img)

# Apply preprocessing
preprocess_images(all_images, output_folder)

print(f"Preprocessed images saved to: {output_folder}")


