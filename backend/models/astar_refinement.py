import cv2
import numpy as np
from heapq import heappush, heappop


def load_image(image_path):
    """Load an image from the given path in BGR format."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    return image


def extract_mask_contour(mask):
    """Extract the largest contour from a binary mask."""
    mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Return the largest contour by area
    contour = max(contours, key=cv2.contourArea)
    return contour


def compute_edge_map(image):
    """Compute a gradient-based edge map using Sobel filters."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return grad_mag


def a_star_snap_point(
    image,
    edge_map,
    start_point,
    prev_point=None,
    search_radius=10,
    lambda_smooth=0.5,
    lambda_prox=0.2,
):
    """Use A* search to snap a contour point to the strongest edge in a search region."""
    h, w = edge_map.shape
    max_edge_value = np.max(edge_map)

    # Define search region
    x, y = start_point
    x_min, x_max = max(0, x - search_radius), min(w, x + search_radius)
    y_min, y_max = max(0, y - search_radius), min(h, y + search_radius)

    # Priority queue for A* (min-heap)
    open_set = []
    heappush(open_set, (0, x, y, 0))  # (f_score, x, y, g_score)

    # Track visited nodes and costs
    visited = set()
    g_scores = {(x, y): 0}
    came_from = {}

    # Heuristic: Euclidean distance to original point
    def heuristic(x, y):
        return lambda_prox * np.sqrt(
            (x - start_point[0]) ** 2 + (y - start_point[1]) ** 2
        )

    # 8-connected neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while open_set:
        _, curr_x, curr_y, g_score = heappop(open_set)

        if (curr_x, curr_y) in visited:
            continue
        visited.add((curr_x, curr_y))

        # Compute edge cost
        edge_cost = max_edge_value - edge_map[curr_y, curr_x]

        # Smoothness cost (if previous point exists)
        smoothness_cost = 0
        if prev_point is not None:
            prev_x, prev_y = prev_point
            smoothness_cost = lambda_smooth * np.sqrt(
                (curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2
            )

        # Total cost for current node
        total_cost = edge_cost + smoothness_cost

        # Explore neighbors
        for dx, dy in neighbors:
            next_x, next_y = curr_x + dx, curr_y + dy
            if not (x_min <= next_x < x_max and y_min <= next_y < y_max):
                continue

            # Compute tentative g_score
            edge_cost = max_edge_value - edge_map[next_y, next_x]
            smoothness_cost = lambda_smooth * np.sqrt(dx**2 + dy**2)
            tentative_g_score = g_score + edge_cost + smoothness_cost

            if (next_x, next_y) not in g_scores or tentative_g_score < g_scores[
                (next_x, next_y)
            ]:
                g_scores[(next_x, next_y)] = tentative_g_score
                f_score = tentative_g_score + heuristic(next_x, next_y)
                heappush(open_set, (f_score, next_x, next_y, tentative_g_score))
                came_from[(next_x, next_y)] = (curr_x, curr_y)

    # Select the best point (lowest cost)
    if not visited:
        return start_point
    best_point = min(visited, key=lambda p: g_scores[p] + heuristic(*p))
    return best_point


def snap_contour_to_edges(contour, edge_map, image, search_radius=10):
    """Snap all contour points to edges using A* search."""
    refined_contour = []
    prev_point = None

    # Simplify contour to reduce points (optional, for efficiency)
    epsilon = 0.01 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)

    for point in contour:
        x, y = point[0]
        new_point = a_star_snap_point(
            image, edge_map, (x, y), prev_point, search_radius
        )
        refined_contour.append([new_point])
        prev_point = new_point

    return np.array(refined_contour, dtype=np.int32)


def reconstruct_mask(contour, image_shape):
    """Reconstruct a binary mask from a contour."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour], 255)
    return mask


def apply_guided_filter(image, mask, radius=5, eps=0.1):
    """Apply guided filter to smooth the mask, preserving edges."""
    try:
        from cv2.ximgproc import guidedFilter

        # Convert mask to float [0, 1]
        mask_float = mask.astype(np.float32) / 255.0
        # Convert image to grayscale for guidance
        guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Apply guided filter
        refined_mask = guidedFilter(guide, mask_float, radius, eps)
        # Convert back to binary mask
        refined_mask = (refined_mask > 0.5).astype(np.uint8) * 255
        return refined_mask
    except ImportError:
        # Fallback to morphological operations if cv2.ximgproc is not available
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
        return refined_mask


def refine_mask(original_mask, image_path, search_radius=10):
    """Refine a Mask R-CNN mask using the automated magnetic lasso pipeline."""
    # Load image
    image = load_image(image_path)

    # Step 1: Extract contour
    contour = extract_mask_contour(original_mask)
    if contour is None:
        return original_mask.copy()  # Return original mask if no contour

    # Step 2: Compute edge map
    edge_map = compute_edge_map(image)

    # Step 3: Snap contour to edges using A*
    refined_contour = snap_contour_to_edges(contour, edge_map, image, search_radius)

    # Step 4: Reconstruct mask
    refined_mask = reconstruct_mask(refined_contour, image.shape)

    # Step 5: Post-process with guided filter
    refined_mask = apply_guided_filter(image, refined_mask)

    return refined_mask
