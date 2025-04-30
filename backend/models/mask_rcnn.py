import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo


def load_image(image_path):
    """Load an image from the given path in BGR format."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    return image


def setup_detectron2(
    model_config="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
):
    """Set up Detectron2 predictor for Mask R-CNN."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
    return DefaultPredictor(cfg)


def run_mask_rcnn(image_path):
    """Run Mask R-CNN inference on the image and return the most confident mask."""
    # Load image
    image = load_image(image_path)

    # Set up Detectron2 predictor
    predictor = setup_detectron2()

    # Run inference
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    masks = instances.pred_masks.numpy()  # Boolean masks [N, H, W]
    scores = instances.scores.numpy()  # Confidence scores [N]

    # If no masks are detected, return a zero mask
    if len(masks) == 0:
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Select the mask with the highest confidence score
    best_idx = np.argmax(scores)
    mask = masks[best_idx].astype(np.uint8) * 255  # Convert to binary (0 or 255)

    return mask
