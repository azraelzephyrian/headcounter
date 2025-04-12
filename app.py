# app.py
import os
from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.transforms as T
from torchvision import models
import uuid

UPLOAD_FOLDER = 'uploads'
MASK_FOLDER = 'masks'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load Model ---
NUM_CLASSES = 3  # background, face, hat
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=False)
model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model.load_state_dict(torch.load('deeplabv3_cihp.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- Preprocessing ---
def preprocess_image(image_path, img_size=(512, 512)):
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image = image.resize(img_size, Image.BILINEAR)
    tensor = T.ToTensor()(image).unsqueeze(0).to(DEVICE)
    return tensor, original_size

# --- Inference ---
def predict_mask(model, input_tensor, original_size):
    with torch.no_grad():
        output = model(input_tensor)['out']
        logits = F.interpolate(output, size=original_size[::-1], mode='bilinear', align_corners=False)
        predicted = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    return predicted

# --- Count Faces and Draw Centroids ---
def count_faces(mask, face_class=1):
    face_mask = (mask == face_class).astype(np.uint8)
    num_labels, labels_im = cv2.connectedComponents(face_mask)

    # Convert to BGR for drawing
    mask_vis = cv2.cvtColor(face_mask * 255, cv2.COLOR_GRAY2BGR)

    for label in range(1, num_labels):
        ys, xs = np.where(labels_im == label)
        if len(xs) == 0:
            continue
        cx = int(xs.mean())
        cy = int(ys.mean())
        cv2.circle(mask_vis, (cx, cy), radius=5, color=(0, 0, 255), thickness=-1)

    return num_labels - 1, mask_vis

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        # Save file
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run model
        tensor, original_size = preprocess_image(filepath)
        mask = predict_mask(model, tensor, original_size)
        num_faces, mask_with_centroids = count_faces(mask)

        # Save mask as image
        mask_path = os.path.join(MASK_FOLDER, f"mask_{filename}")
        cv2.imwrite(mask_path, mask_with_centroids)

        return render_template('result.html', filename=filename, num_faces=num_faces)

    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/masks/<filename>')
def mask_file(filename):
    return send_from_directory(MASK_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)