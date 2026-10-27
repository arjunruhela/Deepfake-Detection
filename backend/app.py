import os
import io
import cv2
import tempfile
import hashlib
import numpy as np
import torch
import torch.nn as nn

from flask import Flask, jsonify, request
from flask_cors import CORS
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from database import (
    create_table,
    save_analysis,
    get_history,
    find_duplicate,
    find_similar_images,
)


# --------------------------------------------------
# Flask application setup
# --------------------------------------------------

app = Flask(__name__)
CORS(app)

# Maximum upload size: 200 MB
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

# Create the SQLite table if it doesn't exist.
create_table()


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "efficientnet_b0_finetuned_best.pth",
)

device = torch.device("cpu")

ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}

ALLOWED_FILE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


# --------------------------------------------------
# Load trained EfficientNet-B0 model
# --------------------------------------------------

model = models.efficientnet_b0(weights=None)
input_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(input_features, 2)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {value: key for key, value in class_to_idx.items()}


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

print("Deepfake detection model loaded successfully!")


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def calculate_file_hash(file_bytes):
    """Return the SHA-256 fingerprint of the exact uploaded bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def calculate_perceptual_hash(image):
    """Create a 64-bit pHash for an image using its low-frequency DCT values."""
    grayscale = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float32)
    dct_values = cv2.dct(pixels)
    low_frequency = dct_values[:8, :8].flatten()

    # Exclude the DC coefficient when calculating the threshold.
    median_value = float(np.median(low_frequency[1:]))
    bits = low_frequency > median_value

    hash_value = 0
    for bit in bits:
        hash_value = (hash_value << 1) | int(bit)

    return f"{hash_value:016x}"


def get_similar_image_info(perceptual_hash, max_distance=8):
    """Find previously stored images with similar pHashes."""
    if not perceptual_hash:
        return {
            "similar_images_detected": False,
            "similar_images": [],
            "similarity_message": "Visual similarity check was not available.",
        }

    matches = find_similar_images(perceptual_hash, max_distance=max_distance)
    return {
        "similar_images_detected": bool(matches),
        "similar_images": matches,
        "similarity_message": (
            "Potential visual match found using perceptual hashing. This indicates "
            "visual similarity, not proof that the files are identical or authentic."
            if matches else
            "No previously analyzed image matched the perceptual-hash threshold."
        ),
    }


def save_record(filename, file_type, file_hash, prediction, perceptual_hash=None):
    """Save an analysis record in SQLite."""
    save_analysis(
        filename,
        file_type,
        file_hash,
        prediction,
        perceptual_hash=perceptual_hash,
    )


def get_duplicate_info(file_hash):
    """Return the earliest previous record for this exact file hash."""
    record = find_duplicate(file_hash)
    if record is None:
        return {
            "duplicate_detected": False,
            "duplicate_of": None,
            "duplicate_message": "No previous file with identical bytes was found.",
        }

    return {
        "duplicate_detected": True,
        "duplicate_of": record,
        "duplicate_message": (
            "This file has the same SHA-256 fingerprint as a file previously analyzed."
        ),
    }


def analyze_image(image):
    """Run EfficientNet-B0 inference on a PIL image."""
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    prediction = idx_to_class[predicted_index]

    fake_score = round(
        float(probabilities[class_to_idx["fake"]]) * 100,
        2,
    )
    real_score = round(
        float(probabilities[class_to_idx["real"]]) * 100,
        2,
    )

    return prediction, fake_score, real_score


def get_upload_bytes(uploaded_file):
    """Read an upload and return its bytes, or None if empty."""
    file_bytes = uploaded_file.read()
    return file_bytes if file_bytes else None


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Deepfake Detection API is running!",
        "model": "EfficientNet-B0 Fine-Tuned",
        "status": "ready",
        "endpoints": {
            "image_detection": "POST /predict",
            "video_detection": "POST /predict-video",
            "analysis_history": "GET /history",
            "duplicate_check": "POST /check-duplicate",
        },
    })


# --------------------------------------------------
# Standalone duplicate-check endpoint
# --------------------------------------------------

@app.route("/check-duplicate", methods=["POST"])
def check_duplicate():
    """Check an uploaded file against history without running the model."""
    if "file" not in request.files:
        return jsonify({
            "error": "Upload a file using the 'file' field.",
        }), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(uploaded_file.filename) or "uploaded_file"
    extension = os.path.splitext(filename)[1].lower()

    if extension not in ALLOWED_FILE_EXTENSIONS:
        return jsonify({
            "error": "Unsupported file extension for duplicate checking.",
        }), 400

    file_bytes = get_upload_bytes(uploaded_file)
    if file_bytes is None:
        return jsonify({"error": "The uploaded file is empty."}), 400

    file_hash = calculate_file_hash(file_bytes)
    file_type = "video" if extension in ALLOWED_VIDEO_EXTENSIONS else "image"

    try:
        duplicate_info = get_duplicate_info(file_hash)
        response = {
            "filename": filename,
            "file_type": file_type,
            "sha256_hash": file_hash,
            **duplicate_info,
            "note": (
                "SHA-256 detects byte-identical files. Perceptual hashing can find "
                "some visually similar image versions, but does not prove authenticity."
            ),
        }

        if file_type == "image":
            try:
                image = Image.open(io.BytesIO(file_bytes))
                image.load()
                perceptual_hash = calculate_perceptual_hash(image)
                response["perceptual_hash"] = perceptual_hash
                response.update(get_similar_image_info(perceptual_hash))
            except (UnidentifiedImageError, OSError, ValueError):
                return jsonify({"error": "Could not process this image file."}), 400
        else:
            response.update({
                "similar_images_detected": False,
                "similar_images": [],
                "similarity_message": "Visual similarity checking is currently available for images only.",
            })

        return jsonify(response)
    except Exception:
        app.logger.exception("Duplicate check failed")
        return jsonify({"error": "Could not check for duplicate files."}), 500


# --------------------------------------------------
# Image detection endpoint
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "Please upload an image using the 'image' field.",
        }), 400

    uploaded_file = request.files["image"]
    if not uploaded_file.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = secure_filename(uploaded_file.filename) or "uploaded_image"
    file_bytes = get_upload_bytes(uploaded_file)

    if file_bytes is None:
        return jsonify({"error": "The uploaded image is empty."}), 400

    try:
        file_hash = calculate_file_hash(file_bytes)
        # Check before saving so this upload only matches earlier records.
        duplicate_info = get_duplicate_info(file_hash)

        image = Image.open(io.BytesIO(file_bytes))
        image.load()
        perceptual_hash = calculate_perceptual_hash(image)
        similar_image_info = get_similar_image_info(perceptual_hash)
        prediction, fake_score, real_score = analyze_image(image)

        save_record(
            filename,
            "image",
            file_hash,
            prediction.capitalize(),
            perceptual_hash=perceptual_hash,
        )

        return jsonify({
            "prediction": prediction.capitalize(),
            "class_scores": {
                "fake": fake_score,
                "real": real_score,
            },
            "sha256_hash": file_hash,
            "perceptual_hash": perceptual_hash,
            **duplicate_info,
            **similar_image_info,
            "warning": (
                "Experimental model output. This score is not proof "
                "of authenticity or manipulation."
            ),
        })

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "error": (
                "Could not process this image. Please upload a valid image file."
            ),
        }), 400
    except Exception:
        app.logger.exception("Image analysis failed")
        return jsonify({
            "error": "An unexpected error occurred during image analysis.",
        }), 500


# --------------------------------------------------
# Video detection endpoint
# --------------------------------------------------

@app.route("/predict-video", methods=["POST"])
def predict_video():
    if "video" not in request.files:
        return jsonify({
            "error": "Please upload a video using the 'video' field.",
        }), 400

    uploaded_file = request.files["video"]
    if not uploaded_file.filename:
        return jsonify({"error": "No video selected."}), 400

    original_filename = secure_filename(uploaded_file.filename) or "uploaded_video"
    suffix = os.path.splitext(original_filename)[1].lower()

    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({"error": "Unsupported video format."}), 400

    file_bytes = get_upload_bytes(uploaded_file)
    if file_bytes is None:
        return jsonify({"error": "The uploaded video is empty."}), 400

    temp_path = None
    cap = None

    try:
        file_hash = calculate_file_hash(file_bytes)
        # Check the database before this upload is saved as a new record.
        duplicate_info = get_duplicate_info(file_hash)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open the video."}), 400

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not fps or fps <= 0 or total_frames <= 0:
            return jsonify({"error": "Could not read video information."}), 400

        duration = total_frames / fps
        sampling_interval = max(1, int(fps))
        frame_predictions = []
        frame_details = []
        timestamp = 0

        # Sample about one frame per second, with a maximum of 120 frames.
        while len(frame_predictions) < 120:
            frame_number = timestamp * sampling_interval
            if frame_number >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = cap.read()
            if not success:
                timestamp += 1
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            prediction, fake_score, real_score = analyze_image(image)

            frame_predictions.append(prediction)
            frame_details.append({
                "timestamp_seconds": timestamp,
                "prediction": prediction.capitalize(),
                "fake_score": fake_score,
                "real_score": real_score,
            })
            timestamp += 1

        cap.release()
        cap = None

        if not frame_predictions:
            return jsonify({"error": "No video frames could be analyzed."}), 400

        fake_count = frame_predictions.count("fake")
        real_count = frame_predictions.count("real")
        analyzed_count = len(frame_predictions)

        if fake_count > real_count:
            majority_result = "Potentially Fake"
        elif real_count > fake_count:
            majority_result = "Potentially Real"
        else:
            majority_result = "Inconclusive"

        save_record(original_filename, "video", file_hash, majority_result)

        return jsonify({
            "model": "EfficientNet-B0 Fine-Tuned",
            "sha256_hash": file_hash,
            **duplicate_info,
            "video_duration_seconds": round(duration, 2),
            "total_frames_analyzed": analyzed_count,
            "fake_frames": fake_count,
            "real_frames": real_count,
            "fake_percentage": round(fake_count / analyzed_count * 100, 2),
            "real_percentage": round(real_count / analyzed_count * 100, 2),
            "majority_vote_result": majority_result,
            "frame_details": frame_details,
            "warning": (
                "Experimental frame-level predictions. The model was trained on images, "
                "not complete videos. This result does not prove video authenticity."
            ),
        })

    except Exception:
        app.logger.exception("Video analysis failed")
        return jsonify({"error": "Could not process this video."}), 500

    finally:
        if cap is not None:
            cap.release()
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                app.logger.warning("Could not remove temporary video file.")


# --------------------------------------------------
# Analysis history endpoint
# --------------------------------------------------

@app.route("/history", methods=["GET"])
def history():
    try:
        records = get_history()
        return jsonify({
            "total_records": len(records),
            "history": records,
        })
    except Exception:
        app.logger.exception("Could not retrieve analysis history")
        return jsonify({"error": "Could not retrieve analysis history."}), 500


# --------------------------------------------------
# Application entry point
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
