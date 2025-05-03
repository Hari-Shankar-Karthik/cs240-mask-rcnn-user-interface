import numpy as np
import cv2
from typing import Optional


def compute_metrics(mask: Optional[np.ndarray], image: np.ndarray) -> dict:
    """
    Compute performance metrics for a mask given the corresponding image.

    Args:
        mask (Optional[np.ndarray]): Binary mask (np.uint8 array of shape (height, width)
                                    with values 0 or 255), or None if invalid.
        image (np.ndarray): Input image (BGR format, np.uint8 array of shape (height, width, 3)).

    Returns:
        dict: Metrics including edge_alignment_score and region_homogeneity_score.
              Returns {"edge_alignment_score": 0.0, "region_homogeneity_score": 0.0} if mask is None.
    """
    if mask is None:
        return {"edge_alignment_score": 0.0, "region_homogeneity_score": 0.0}

    # Ensure mask is binary (0 or 255)
    mask = (mask > 0).astype(np.uint8) * 255

    # 1. Edge Alignment Score
    # Compute edge map of the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    edge_map = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, strong_edges = cv2.threshold(edge_map, 50, 255, cv2.THRESH_BINARY)

    # Extract mask contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        edge_alignment_score = 0.0
    else:
        contour = max(contours, key=cv2.contourArea)  # Largest contour
        contour_points = contour.reshape(-1, 2)  # Shape: (N, 2)
        total_points = len(contour_points)
        if total_points == 0:
            edge_alignment_score = 0.0
        else:
            # Dilate strong edges to allow small misalignment (2-pixel radius)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated_edges = cv2.dilate(strong_edges, kernel)
            # Check if contour points lie on dilated edges
            aligned_points = sum(
                1 for x, y in contour_points if dilated_edges[y, x] > 0
            )
            edge_alignment_score = (
                aligned_points / total_points if total_points > 0 else 0.0
            )

    # 2. Region Homogeneity Score
    # Compute variance of grayscale intensities in the masked region
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    masked_pixels = gray[mask > 0]
    if len(masked_pixels) == 0:
        region_homogeneity_score = 0.0
    else:
        mask_variance = np.var(masked_pixels) if len(masked_pixels) > 1 else 0.0
        image_variance = (
            np.var(gray) if np.var(gray) > 0 else 1.0
        )  # Avoid division by zero
        # Normalize variance: lower variance -> higher score
        normalized_variance = mask_variance / image_variance
        # Convert to score: exp(-variance) for 0 to 1 range, lower variance -> higher score
        region_homogeneity_score = np.exp(
            -min(normalized_variance, 10.0)
        )  # Cap to avoid overflow

    return {
        "edge_alignment_score": float(edge_alignment_score),
        "region_homogeneity_score": float(region_homogeneity_score),
    }
