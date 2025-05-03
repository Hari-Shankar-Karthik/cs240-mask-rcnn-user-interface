from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import base64
from PIL import Image
import io
import time
import json
import numpy as np
from typing import Optional
from models.mask_rcnn import run_mask_rcnn
from models.astar_refinement import refine_mask
from utils.image_utils import save_image, image_to_base64
import threading
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Thread lock for file operations
file_lock = threading.Lock()


def compute_metrics(
    original_mask: Optional[np.ndarray], custom_mask: Optional[np.ndarray]
) -> dict:
    """Compute IoU, Dice coefficient, and IoU improvement for masks."""
    if original_mask is None or custom_mask is None:
        return {"iou_improvement": 0.0, "dice_coefficient": 0.0}

    # Convert masks to boolean
    original_mask = original_mask > 0
    custom_mask = custom_mask > 0

    # Compute intersection and union
    intersection = np.logical_and(original_mask, custom_mask).sum()
    union = np.logical_or(original_mask, custom_mask).sum()

    # Compute IoU
    iou = intersection / union if union > 0 else 0.0

    # Compute Dice coefficient
    dice = (
        (2 * intersection) / (original_mask.sum() + custom_mask.sum())
        if (original_mask.sum() + custom_mask.sum()) > 0
        else 0.0
    )

    # IoU improvement (relative to original mask's IoU with itself, which is 1.0)
    iou_improvement = iou - 1.0  # Simplified; assumes original mask is baseline

    return {"iou_improvement": float(iou_improvement), "dice_coefficient": float(dice)}


def process_instance(
    image_path: str, image_id: str, index: int, total_instances: int
) -> bool:
    """Process a single instance and save results."""
    try:
        start_time = time.time()
        original_mask, _ = run_mask_rcnn(image_path, index)
        if original_mask is None:
            return False

        custom_mask = refine_mask(original_mask, image_path)
        metrics = compute_metrics(original_mask, custom_mask)
        metrics["processing_time"] = time.time() - start_time

        original_mask_path = os.path.join(
            RESULT_FOLDER, f"{image_id}_{index}_original.png"
        )
        custom_mask_path = os.path.join(RESULT_FOLDER, f"{image_id}_{index}_custom.png")
        metrics_path = os.path.join(RESULT_FOLDER, f"{image_id}_{index}_metrics.json")

        with file_lock:
            Image.fromarray(original_mask).save(original_mask_path)
            Image.fromarray(custom_mask).save(custom_mask_path)
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "metrics": metrics,
                        "original_mask_path": original_mask_path,
                        "custom_mask_path": custom_mask_path,
                        "total_instances": total_instances,
                    },
                    f,
                )
        return True
    except Exception as e:
        logger.error(
            f"Error processing index {index} for image_id {image_id}: {str(e)}"
        )
        return False


def background_process_all_instances(
    image_path: str, image_id: str, total_instances: int, skip_index: int
):
    """Compute masks for all instances in the background, skipping the provided index."""
    for index in range(total_instances):
        if index == skip_index:
            continue
        process_instance(image_path, image_id, index, total_instances)


@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files or "index" not in request.form:
        return jsonify({"error": "Image and index are required"}), 400

    file = request.files["image"]
    try:
        index = int(request.form["index"])
    except ValueError:
        return jsonify({"error": "Index must be an integer"}), 400

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    image_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_FOLDER, f"{image_id}.png")
    save_image(file, image_path)

    try:
        start_time = time.time()
        # Run Mask R-CNN for the specified index
        original_mask, total_instances = run_mask_rcnn(image_path, index)

        if original_mask is None:
            return (
                jsonify(
                    {
                        "error": f"Invalid index {index}",
                        "total_instances": total_instances,
                    }
                ),
                404,
            )

        # Run A* refinement
        custom_mask = refine_mask(original_mask, image_path)

        # Compute metrics
        metrics = compute_metrics(original_mask, custom_mask)
        processing_time = time.time() - start_time
        metrics["processing_time"] = processing_time

        # Save results
        original_mask_path = os.path.join(
            RESULT_FOLDER, f"{image_id}_{index}_original.png"
        )
        custom_mask_path = os.path.join(RESULT_FOLDER, f"{image_id}_{index}_custom.png")
        metrics_path = os.path.join(RESULT_FOLDER, f"{image_id}_{index}_metrics.json")

        with file_lock:
            Image.fromarray(original_mask).save(original_mask_path)
            Image.fromarray(custom_mask).save(custom_mask_path)
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "metrics": metrics,
                        "original_mask_path": original_mask_path,
                        "custom_mask_path": custom_mask_path,
                        "total_instances": total_instances,
                    },
                    f,
                )

        # Start background processing for all other indices
        threading.Thread(
            target=background_process_all_instances,
            args=(image_path, image_id, total_instances, index),
            daemon=True,
        ).start()

        # Prepare response
        results = {
            "original_mask": image_to_base64(original_mask_path),
            "custom_mask": image_to_base64(custom_mask_path),
            "metrics": metrics,
            "total_instances": total_instances,
        }

        return jsonify({"image_id": image_id, "results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/results/<image_id>/<int:index>", methods=["GET"])
def get_results(image_id: str, index: int):
    metrics_path = os.path.join(RESULT_FOLDER, f"{image_id}_{index}_metrics.json")
    image_path = os.path.join(UPLOAD_FOLDER, f"{image_id}.png")

    # If metrics file exists, load and return results
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                data = json.load(f)

            original_mask_path = data["original_mask_path"]
            custom_mask_path = data["original_mask_path"]
            total_instances = data["total_instances"]

            original_mask_b64 = (
                image_to_base64(original_mask_path) if original_mask_path else None
            )
            custom_mask_b64 = (
                image_to_base64(custom_mask_path) if custom_mask_path else None
            )

            return (
                jsonify(
                    {
                        "original_mask": original_mask_b64,
                        "custom_mask": custom_mask_b64,
                        "metrics": {
                            "iou_improvement": data["metrics"].get(
                                "iou_improvement", 0.0
                            ),
                            "dice_coefficient": data["metrics"].get(
                                "dice_coefficient", 0.0
                            ),
                            "processing_time": data["metrics"].get(
                                "processing_time", 0.0
                            ),
                        },
                        "total_instances": total_instances,
                    }
                ),
                200,
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # If metrics file doesn't exist, compute results on-demand
    if not os.path.exists(image_path):
        return jsonify({"error": "Image not found"}), 404

    try:
        start_time = time.time()
        # Run Mask R-CNN for the specified index
        original_mask, total_instances = run_mask_rcnn(image_path, index)

        if original_mask is None:
            return (
                jsonify(
                    {
                        "error": f"Invalid index {index}",
                        "total_instances": total_instances,
                    }
                ),
                404,
            )

        # Run A* refinement
        custom_mask = refine_mask(original_mask, image_path)

        # Compute metrics
        metrics = compute_metrics(original_mask, custom_mask)
        processing_time = time.time() - start_time
        metrics["processing_time"] = processing_time

        # Save results
        original_mask_path = os.path.join(
            RESULT_FOLDER, f"{image_id}_{index}_original.png"
        )
        custom_mask_path = os.path.join(RESULT_FOLDER, f"{image_id}_{index}_custom.png")
        metrics_path = os.path.join(RESULT_FOLDER, f"{image_id}_{index}_metrics.json")

        with file_lock:
            Image.fromarray(original_mask).save(original_mask_path)
            Image.fromarray(custom_mask).save(custom_mask_path)
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "metrics": metrics,
                        "original_mask_path": original_mask_path,
                        "custom_mask_path": custom_mask_path,
                        "total_instances": total_instances,
                    },
                    f,
                )

        # Prepare response
        results = {
            "original_mask": image_to_base64(original_mask_path),
            "custom_mask": image_to_base64(custom_mask_path),
            "metrics": metrics,
            "total_instances": total_instances,
        }

        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
