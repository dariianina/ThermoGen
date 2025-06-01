import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

def create_advanced_thermal_mask_drone(white_mask, thermal_params=None):
    if thermal_params is None:
        thermal_params = {
            'center_intensity': 240,
            'edge_intensity': 160,
            'gradient_power': 1.5,
            'noise_strength': 5,
            'blur_sigma': 2.0
        }
    _, binary_mask = cv2.threshold(white_mask, 127, 255, cv2.THRESH_BINARY)
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    if dist_transform.max() > 0:
        normalized_dist = (dist_transform / dist_transform.max()) ** thermal_params['gradient_power']
    else:
        normalized_dist = dist_transform
    thermal_mask = np.zeros_like(binary_mask, dtype=np.float32)
    mask_indices = binary_mask > 0
    dilated_mask = cv2.dilate(binary_mask, np.ones((5, 5), np.uint8), iterations=1)
    mask_indices = dilated_mask > 0
    intensity_range = thermal_params['center_intensity'] - thermal_params['edge_intensity']
    thermal_mask[mask_indices] = (normalized_dist[mask_indices] * intensity_range +
                                  thermal_params['edge_intensity'])
    noise = np.random.normal(0, thermal_params['noise_strength'], thermal_mask.shape)
    thermal_mask[mask_indices] += noise[mask_indices]
    thermal_mask = cv2.GaussianBlur(thermal_mask, (0, 0), thermal_params['blur_sigma'])
    final_mask = np.clip(thermal_mask, 0, 255).astype(np.uint8)
    return final_mask

def blend_thermal_mask_on_image(image, white_mask, show=False, thermal_params=None):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 2:
        image = np.stack([image]*3, axis=-1)
    final_mask = create_advanced_thermal_mask_drone(white_mask, thermal_params)
    if len(final_mask.shape) == 3:
        mask_gray = cv2.cvtColor(final_mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = final_mask
    mask_normalized = mask_gray.astype(np.float32) / 255.0
    mask_3c = np.repeat(mask_normalized[:, :, np.newaxis], 3, axis=2)
    image = image.astype(np.float32)
    thermal_white = np.ones_like(image) * 255
    blended = image * (1 - mask_3c) + thermal_white * mask_3c
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    if show:
        plt.imshow(blended)
        plt.axis('off')
        plt.show()
    return blended

def process_and_save_blended_image(thermal_image_path, mask_image_path, output_path, show=False):
    """
    Loads images from disk, blends the mask, and saves the result.
    """
    # Load images
    thermal_image = Image.open(thermal_image_path).convert("RGB")
    white_mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
    # Blend
    result = blend_thermal_mask_on_image(thermal_image, white_mask, show=show)
    # Save
    Image.fromarray(result).save(output_path)
    print(f"Saved blended image to {output_path}")


# --- Batch processing loop ---
inference_dir = "./video_masks/v1_130_inference"
mask_dir = "./video_masks/v1_130_s"
output_dir = "./video_masks/blended_imgs"
os.makedirs(output_dir, exist_ok=True)

# List all image files in inference_dir (assuming .jpg, .png, .jpeg)
image_files = sorted([
    f for f in os.listdir(inference_dir)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

for fname in image_files:
    thermal_image_path = os.path.join(inference_dir, fname)
    # Assume mask has the same name but .png extension
    mask_name = os.path.splitext(fname)[0] + ".png"
    mask_image_path = os.path.join(mask_dir, mask_name)
    output_path = os.path.join(output_dir, fname)
    try:
        process_and_save_blended_image(
            thermal_image_path,
            mask_image_path,
            output_path,
            show=False  # Set to True if you want to visualize each
        )
    except Exception as e:
        print(f"Failed for {fname}: {e}")

print("Batch blending complete!")