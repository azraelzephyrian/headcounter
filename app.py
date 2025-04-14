import io
import os
import base64
from flask import Flask, request, render_template
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerFeatureExtractor
import matplotlib.pyplot as plt

# Set device and image size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (512, 512)

# === Model Initialization ===
# Rebuild SegFormer configuration manually
config = SegformerConfig(
    num_labels=3,
    id2label={0: "background", 1: "face", 2: "hat"},
    label2id={"background": 0, "face": 1, "hat": 2}
)

# Initialize the SegFormer model and load weights
segformer = SegformerForSemanticSegmentation(config)
segformer.load_state_dict(torch.load("segformer_cihp.pth", map_location=DEVICE))
segformer = segformer.to(DEVICE)
segformer.eval()

# Initialize the processor (feature extractor) used for segmentation
processor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# === Utility Functions ===
def pil_to_base64(pil_img):
    """Convert a PIL image to a base64-encoded PNG."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def preprocess_resized_image(img_file):
    """
    Load an image from a file-like object, convert to RGB,
    and resize it to the expected IMG_SIZE.
    """
    image = Image.open(img_file).convert("RGB")
    image_resized = image.resize(IMG_SIZE, Image.BILINEAR)
    return image_resized, np.array(image_resized)

def predict_mask_segformer_with_tiles(
    model,
    processor,
    image_pil,
    tile_size=(512, 512),
    stride=256,
    detail_stride=128,
    detail_threshold=0.3,
    class_id=1
):
    """
    Run segmented prediction on an image using a combination of:
      1. Full image segmentation.
      2. A sliding window (tile) approach.
      3. An adaptive mask pass on high-detail regions.
    The final mask is computed by merging the masks from the three passes.
    """
    w, h = image_pil.size
    image_np = np.array(image_pil)

    # --- 1. Estimate Blob Size from Edge Map ---
    def estimate_average_blob_size(image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        edges = np.uint8(np.abs(sobel) > 30)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated)
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            return 64, dilated
        avg_area = np.mean(areas)
        est_size = int(np.sqrt(avg_area))
        return est_size, dilated

    blob_size, dilated = estimate_average_blob_size(image_np)
    detail_tile_size = max(64, min(512, 4 * blob_size))

    # --- 2. Full Image Segmentation ---
    full_inputs = processor(images=image_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        full_logits = model(**full_inputs).logits
    full_pred = torch.argmax(
        F.interpolate(full_logits, size=(h, w), mode="bilinear", align_corners=False),
        dim=1
    ).squeeze(0).cpu().numpy()

    # --- 3. Sliding Window Segmentation ---
    tile_mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h - tile_size[1] + 1, stride):
        for x in range(0, w - tile_size[0] + 1, stride):
            crop = image_pil.crop((x, y, x + tile_size[0], y + tile_size[1]))
            inputs = processor(images=crop, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs).logits
            pred = torch.argmax(
                F.interpolate(outputs, size=tile_size, mode="bilinear", align_corners=False),
                dim=1
            ).squeeze(0).cpu().numpy()
            tile_mask[y:y + tile_size[1], x:x + tile_size[0]] = np.maximum(
                tile_mask[y:y + tile_size[1], x:x + tile_size[0]], pred
            )

    # --- 4. Compute Detail Map ---
    def compute_detail_map(image_np):
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        return grad_mag / grad_mag.max()

    detail_map = compute_detail_map(image_np)

    # --- 5. Select High-Detail Tiles ---
    def get_high_detail_tiles(detail_map):
        tiles = []
        for y in range(0, h - detail_tile_size + 1, detail_stride):
            for x in range(0, w - detail_tile_size + 1, detail_stride):
                tile = detail_map[y:y + detail_tile_size, x:x + detail_tile_size]
                if tile.mean() > detail_threshold:
                    tiles.append((x, y, detail_tile_size, detail_tile_size))
        return tiles

    high_detail_tiles = get_high_detail_tiles(detail_map)

    # --- 6. Adaptive Mask Pass ---
    adaptive_mask = np.zeros((h, w), dtype=np.uint8)
    for x, y, tw, th in high_detail_tiles:
        crop = image_pil.crop((x, y, x + tw, y + th)).resize((512, 512), Image.BILINEAR)
        inputs = processor(images=crop, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs).logits
        pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
        pred_resized = cv2.resize(pred.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)
        adaptive_mask[y:y + th, x:x + tw] = np.maximum(adaptive_mask[y:y + th, x:x + tw], pred_resized)

    # --- 7. Merge Masks ---
    final_mask = np.where(
        (full_pred == class_id) | (tile_mask == class_id) | (adaptive_mask == class_id),
        class_id,
        0
    ).astype(np.uint8)

    return final_mask

def draw_centroids(mask):
    """
    Creates a black image with:
    - red mask regions (where mask == 1)
    - green dots at the centroids of connected mask regions
    """
    h, w = mask.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    # Paint red over the mask area
    output[mask == 1] = [255, 0, 0]

    # Add green centroids
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for i in range(1, num_labels):  # Skip background
        cX, cY = int(centroids[i][0]), int(centroids[i][1])
        cv2.circle(output, (cX, cY), 3, (0, 255, 0), -1)

    return output


# === Flask App Initialization ===
app = Flask(__name__)

@app.route('/')
def index():
    """Display the index page with an image upload form."""
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment():
    """
    Handles file upload, performs preprocessing and segmentation using the
    tiled SegFormer approach, then generates three visualizations:
      1. Original image.
      2. Mask overlaid with centroids.
      3. Image blended 50% with a red mask.
    Also counts the number of detected head blobs in the mask.
    """
    file = request.files['image']
    file_bytes = file.read()
    image_io = io.BytesIO(file_bytes)
    image_pil, _ = preprocess_resized_image(image_io)

    # Run segmentation (default: class_id=1 for "face")
    mask = predict_mask_segformer_with_tiles(segformer, processor, image_pil, class_id=1)

    # Count connected components (excluding background)
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    head_count = num_labels - 1  # Subtract 1 for background

    # Convert the original image to numpy format for visualization
    image_np = np.array(image_pil)

    # Generate a centroid overlay and blend
    mask_with_centroids = draw_centroids(mask)
    mask_color = np.zeros_like(image_np)
    mask_color[mask == 1] = [255, 0, 0]
    blended = cv2.addWeighted(image_np, 0.5, mask_color, 0.5, 0)

    # Convert to base64 for display
    original_b64 = pil_to_base64(image_pil)
    centroids_b64 = pil_to_base64(Image.fromarray(mask_with_centroids))
    blended_b64 = pil_to_base64(Image.fromarray(blended))

    return render_template('results.html',
                           original=original_b64,
                           centroids=centroids_b64,
                           blended=blended_b64,
                           count=head_count)


if __name__ == '__main__':
    app.run(debug=True)
