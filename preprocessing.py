import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def normalize_and_save_image(image_path, output_path):
    # Load image and convert to RGB
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to HSV and isolate green areas (leaf)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, (25, 40, 40), (85, 255, 255))
    leaf_only = cv2.bitwise_and(img_rgb, img_rgb, mask=green_mask)

    # Normalize to [0, 1], then scale to [0, 255] and convert to uint8
    normalized = (leaf_only / 255.0) * 255
    normalized_uint8 = normalized.astype(np.uint8)

    # Save the normalized image
    cv2.imwrite(output_path, cv2.cvtColor(normalized_uint8, cv2.COLOR_RGB2BGR))

    # Show the image
    plt.imshow(normalized_uint8)
    plt.title("Normalized & Saved Image")
    plt.axis("off")
    plt.show()

# === Usage ===
input_image = "/Users/jameskierdoliguez/Downloads/Rice Leaf Disease Images/Tungro/TUNGRO1_010.jpg"
output_image = "/Users/jameskierdoliguez/Downloads/Figure_1.png"
normalize_and_save_image(input_image, output_image)

