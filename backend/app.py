from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import time
from models.mask_rcnn import run_mask_rcnn
from models.astar_refinement import refine_mask
from utils import save_image, image_to_base64, compute_metrics

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "Uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store image paths and cached masks by image_id
image_storage = (
    {}
)  # {image_id: {"path": str, "masks": List[np.ndarray], "total_instances": int}}


@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files or "index" not in request.form:
        return jsonify({"error": "Missing image or index"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        index = int(request.form["index"])
    except ValueError:
        return jsonify({"error": "Invalid index format"}), 400

    # Generate unique image_id and save image
    image_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_FOLDER, f"{image_id}.png")
    save_image(file, image_path)

    try:
        start_time = time.time()
        # Run Mask R-CNN for all instances to cache masks
        image = load_image(image_path)  # Load image for inference
        predictor = setup_detectron2()  # From mask_rcnn.py
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        masks = instances.pred_masks.numpy()  # Boolean masks [N, H, W]
        total_instances = len(masks)
        cached_masks = [mask.astype(np.uint8) * 255 for mask in masks]

        # Store image path and masks
        image_storage[image_id] = {
            "path": image_path,
            "masks": cached_masks,
            "total_instances": total_instances,
        }

        # Process mask for the requested index
        if index < 0 or index >= total_instances:
            os.remove(image_path)
            del image_storage[image_id]
            return jsonify({"error": "Invalid index"}), 404

        original_mask = cached_masks[index]
        custom_mask = refine_mask(
            original_mask, image_path, use_guided_filter=False
        )  # Faster without guided filter
        if custom_mask is None:
            custom_mask = original_mask  # Fallback to original

        # Compute metrics
        metrics = compute_metrics(original_mask, custom_mask)
        metrics["processing_time"] = time.time() - start_time

        # Convert masks to base64
        original_mask_b64 = image_to_base64(original_mask)
        custom_mask_b64 = image_to_base64(custom_mask)

        response = {
            "image_id": image_id,
            "results": {
                "original_mask": original_mask_b64,
                "custom_mask": custom_mask_b64,
                "metrics": metrics,
                "total_instances": total_instances,
            },
        }

        return jsonify(response), 200
    except Exception as e:
        if os.path.exists(image_path):
            os.remove(image_path)
        if image_id in image_storage:
            del image_storage[image_id]
        return jsonify({"error": str(e)}), 500


@app.route("/results/<image_id>/<index>", methods=["GET"])
def get_results(image_id: str, index: str):
    if image_id not in image_storage:
        return jsonify({"error": "Image not found"}), 404

    try:
        index = int(index)
    except ValueError:
        return jsonify({"error": "Invalid index format"}), 400

    storage = image_storage[image_id]
    image_path = storage["path"]
    cached_masks = storage["masks"]
    total_instances = storage["total_instances"]

    try:
        start_time = time.time()
        # Check index validity
        if index < 0 or index >= total_instances:
            return jsonify({"error": "Invalid index"}), 404

        # Retrieve cached mask
        original_mask = cached_masks[index]
        custom_mask = refine_mask(
            original_mask, image_path, use_guided_filter=False
        )  # Faster without guided filter
        if custom_mask is None:
            custom_mask = original_mask  # Fallback to original

        # Compute metrics
        metrics = compute_metrics(original_mask, custom_mask)
        metrics["processing_time"] = time.time() - start_time

        # Convert masks to base64
        original_mask_b64 = image_to_base64(original_mask)
        custom_mask_b64 = image_to_base64(custom_mask)

        response = {
            "original_mask": original_mask_b64,
            "custom_mask": custom_mask_b64,
            "metrics": metrics,
            "total_instances": total_instances,
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.teardown_request
def cleanup(exception=None):
    # Optional: Clean up old images and masks periodically
    # Example: Remove entries older than 1 hour (implement as needed)
    pass


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
