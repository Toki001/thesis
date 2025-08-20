import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img_path = "/Users/jameskierdoliguez/Downloads/Rice Leaf Disease Images/Tungro/TUNGRO1_001.jpg"
img = cv2.imread(img_path)

# Convert to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Check shape and data type
print("Original dtype:", img_rgb.dtype)
print("Original min/max:", img_rgb.min(), "/", img_rgb.max())

# Normalize
img_normalized = img_rgb.astype(np.float32) / 255.0

# Confirm it worked
print("Normalized dtype:", img_normalized.dtype)
print("Normalized min/max:", img_normalized.min(), "/", img_normalized.max())

# Plot result (side by side)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original (0-255)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_normalized)
plt.title("Normalized (0.0-1.0)")
plt.axis("off")
plt.tight_layout()
plt.show()

