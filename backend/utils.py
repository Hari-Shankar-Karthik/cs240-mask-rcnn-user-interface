import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Optional


def save_image(file, path: str) -> None:
    """Save an uploaded file to the specified path."""
    file.save(path)


def load_image(path: str) -> np.ndarray:
    """Load an image from the specified path in BGR format."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not load image at {path}")
    return image


def image_to_base64(image: np.ndarray) -> str:
    """Convert a NumPy array (image or mask) to a base64-encoded PNG string."""
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")


def compute_metrics(original_mask: np.ndarray, custom_mask: np.ndarray) -> dict:
    """
    Compute metrics comparing original and custom masks.

    Args:
        original_mask (np.ndarray): Original binary mask (uint8, 0 or 255).
        custom_mask (np.ndarray): Refined binary mask (uint8, 0 or 255).

    Returns:
        dict: Metrics including iou_improvement, dice_coefficient, and processing_time.
    """
    # Placeholder metrics (replace with actual computations if needed)
    return {
        "iou_improvement": 0.0,
        "dice_coefficient": 0.0,
        # processing_time is added in app.py
    }
