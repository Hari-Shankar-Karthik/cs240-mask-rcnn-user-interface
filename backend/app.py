from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import base64
from PIL import Image
import io
import time
from models.mask_rcnn import run_mask_rcnn
from models.astar_refinement import refine_mask
from utils.image_utils import save_image, load_image, image_to_base64
from utils.metrics import compute_metrics
import json

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    image_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_FOLDER, f"{image_id}.png")
    save_image(file, image_path)

    # Process image asynchronously (simplified for template)
    try:
        start_time = time.time()
        # Run Mask R-CNN
        original_mask = run_mask_rcnn(image_path)
        # Run A* refinement
        custom_mask = refine_mask(original_mask, image_path)
        # Compute metrics
        metrics = compute_metrics(original_mask, custom_mask)
        processing_time = time.time() - start_time

        # Save results
        original_mask_path = os.path.join(RESULT_FOLDER, f"{image_id}_original.png")
        custom_mask_path = os.path.join(RESULT_FOLDER, f"{image_id}_custom.png")
        Image.fromarray(original_mask).save(original_mask_path)
        Image.fromarray(custom_mask).save(custom_mask_path)

        # Store metrics
        with open(os.path.join(RESULT_FOLDER, f"{image_id}_metrics.json"), "w") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "original_mask_path": original_mask_path,
                    "custom_mask_path": custom_mask_path,
                    "processing_time": processing_time,
                },
                f,
            )

        return jsonify({"image_id": image_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/results/<image_id>", methods=["GET"])
def get_results(image_id):
    metrics_path = os.path.join(RESULT_FOLDER, f"{image_id}_metrics.json")
    if not os.path.exists(metrics_path):
        return jsonify({"error": "Image not found or still processing"}), 404

    with open(metrics_path, "r") as f:
        data = json.load(f)

    original_mask_path = data["original_mask_path"]
    custom_mask_path = data["custom_mask_path"]

    original_mask_b64 = image_to_base64(original_mask_path)
    custom_mask_b64 = image_to_base64(custom_mask_path)

    return (
        jsonify(
            {
                "original_mask": original_mask_b64,
                "custom_mask": custom_mask_b64,
                "metrics": {
                    "iou_improvement": data["metrics"].get("iou_improvement", 0.0),
                    "dice_coefficient": data["metrics"].get("dice_coefficient", 0.0),
                    "processing_time": data["processing_time"],
                },
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
