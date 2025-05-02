import cv2
import numpy as np
from typing import Optional, Tuple
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo


def load_image(image_path: str) -> np.ndarray:
    """Load an image from the given path in BGR format."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    return image


def setup_detectron2(
    model_config: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
) -> DefaultPredictor:
    """Set up Detectron2 predictor for Mask R-CNN."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
    return DefaultPredictor(cfg)


def run_mask_rcnn(image_path: str, index: int) -> Tuple[Optional[np.ndarray], int]:
    """
    Run Mask R-CNN inference for a specific instance in the input image using Detectron2.

    Args:
        image_path (str): Path to the input image file.
        index (int): Index of the instance to compute the mask for.

    Returns:
        tuple[Optional[np.ndarray], int]:
            - Binary mask (np.uint8 array of shape (height, width) with values 0 or 255) for the specified instance,
              or None if the index is invalid.
            - Total number of detected instances.
    """
    # Load image
    image = load_image(image_path)

    # Set up Detectron2 predictor
    predictor = setup_detectron2()

    # Run inference
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    masks = instances.pred_masks.numpy()  # Boolean masks [N, H, W]
    total_instances = len(masks)

    # Check if index is valid
    if index < 0 or index >= total_instances:
        return None, total_instances

    # Return the binary mask for the specified index
    mask = masks[index].astype(np.uint8) * 255  # Convert to binary (0 or 255)
    return mask, total_instances
