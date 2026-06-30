# app/server.py
import os
import sys
import base64
from io import BytesIO
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image
import cv2

# Add parent directory to path so we can import models and data scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sr_model import SRNet
from models.color_model import UNetGenerator
from data.preprocess import load_band, convert_to_physical, downscale_image, normalize_data
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__, template_folder='templates', static_folder='static')

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models once at startup
sr_model = None
color_model = None
models_loaded = False

def load_models():
    global sr_model, color_model, models_loaded
    if models_loaded:
        return True
        
    sr_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "sr_best.pth")
    color_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "color_best.pth")
    
    if os.path.exists(sr_path) and os.path.exists(color_path):
        sr_model = SRNet().to(device)
        color_model = UNetGenerator().to(device)
        sr_model.load_state_dict(torch.load(sr_path, map_location=device))
        color_model.load_state_dict(torch.load(color_path, map_location=device))
        sr_model.eval()
        color_model.eval()
        models_loaded = True
        print("Models loaded successfully.")
        return True
    else:
        print("Model weights not found. Running in dummy mode or requiring training.")
        return False

def array_to_base64_png(arr, colormap=None):
    """Converts a numpy array to base64 encoded PNG string."""
    if arr.ndim == 3:  # RGB
        # If in range [-1, 1], scale to [0, 1]
        if arr.min() < -0.1:
            arr = (arr + 1.0) / 2.0
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    else:  # Grayscale / Single band
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        if colormap == 'thermal':
            # Apply JET colormap for pseudo-color thermal representation
            colored = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(colored)
        else:
            img = Image.fromarray(arr)
            
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/scenes')
def get_scenes():
    scenes = [
        {"id": "urban", "name": "Urban Heat Island", "description": "High thermal concrete, cold rivers, distinct street patterns."},
        {"id": "agriculture", "name": "Agricultural Fields", "description": "Cool vegetated plots next to hot bare-soil fields."},
        {"id": "forest", "name": "Forest Canopy", "description": "Uniformly cool dense canopy, forest clearing, and lake."},
        {"id": "coastal", "name": "Coastal Zone", "description": "Uniform cold sea water, warm beach sand, moderate land vegetation."}
    ]
    return jsonify(scenes)

@app.route('/api/inference', methods=['POST'])
def run_inference():
    scene_id = request.json.get('scene', 'urban')
    
    # Check if models are loaded
    has_models = load_models()
    
    # Paths for raw bands
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(project_root, "data", "raw")
    
    b2_path = os.path.join(raw_dir, f"{scene_id}_B2.tif")
    b3_path = os.path.join(raw_dir, f"{scene_id}_B3.tif")
    b4_path = os.path.join(raw_dir, f"{scene_id}_B4.tif")
    b10_path = os.path.join(raw_dir, f"{scene_id}_B10.tif")
    
    if not os.path.exists(b10_path):
        return jsonify({"error": "Raw scene data not found. Please run generate_synthetic.py first."}), 404
        
    # Load and process raw bands
    b2, _ = load_band(b2_path)
    b3, _ = load_band(b3_path)
    b4, _ = load_band(b4_path)
    b10, _ = load_band(b10_path)
    
    rgb, tir = convert_to_physical(b2, b3, b4, b10)
    
    # Downscale
    rgb_100m = downscale_image(rgb, 3.3333)
    tir_100m = downscale_image(tir, 3.3333)
    tir_200m = downscale_image(tir, 6.6667)
    
    # Normalize
    rgb_100m_norm, tir_100m_norm = normalize_data(rgb_100m, tir_100m)
    _, tir_200m_norm = normalize_data(np.zeros_like(tir_200m), tir_200m)
    
    # Crop central region to display
    # 200m: 64x64 region (from 153x153)
    # 100m: 128x128 region (from 307x307)
    lr_h, lr_w = tir_200m_norm.shape
    lr_cy, lr_cx = lr_h // 2, lr_w // 2
    
    tir_lr_patch = tir_200m_norm[lr_cy-32 : lr_cy+32, lr_cx-32 : lr_cx+32]
    tir_hr_patch = tir_100m_norm[lr_cy*2-64 : lr_cy*2+64, lr_cx*2-64 : lr_cx*2+64]
    rgb_hr_patch = rgb_100m_norm[lr_cy*2-64 : lr_cy*2+64, lr_cx*2-64 : lr_cx*2+64]
    
    # Run pipeline
    import time
    start_time = time.time()
    
    # Baselines
    # Interpolated TIR baseline (64x64 -> 128x128)
    tir_lr_patch_t = torch.from_numpy(tir_lr_patch).unsqueeze(0).unsqueeze(0)
    tir_sr_base = F.interpolate(tir_lr_patch_t, size=(128, 128), mode='bilinear', align_corners=False).squeeze().numpy()
    
    if has_models:
        # Run deep learning model
        tir_lr_patch_dev = tir_lr_patch_t.to(device)
        with torch.no_grad():
            tir_sr_pred_t = sr_model(tir_lr_patch_dev)
            rgb_pred_t = color_model(tir_sr_pred_t)
            
        tir_sr_pred = tir_sr_pred_t.squeeze().cpu().numpy()
        rgb_pred = rgb_pred_t.squeeze().permute(1, 2, 0).cpu().numpy()
    else:
        # Dummy fallback if models are not trained yet
        tir_sr_pred = tir_sr_base * 0.95 + 0.05 * np.random.rand(128, 128)
        # Simple colorization mapping for preview
        rgb_pred = np.zeros((128, 128, 3))
        rgb_pred[:, :, 0] = tir_sr_pred * 0.8  # Red
        rgb_pred[:, :, 1] = tir_sr_pred * 0.5  # Green
        rgb_pred[:, :, 2] = (1 - tir_sr_pred) * 0.7  # Blue
        rgb_pred = rgb_pred * 2.0 - 1.0 # scale to [-1, 1]
        
    inference_time = (time.time() - start_time) * 1000.0 # in ms
    
    # Scale variables for metrics comparison
    tir_sr_base_scaled = tir_sr_base
    tir_sr_pred_scaled = tir_sr_pred
    tir_hr_scaled = tir_hr_patch
    
    rgb_pred_scaled = (rgb_pred + 1.0) / 2.0
    rgb_hr_scaled = (rgb_hr_patch + 1.0) / 2.0
    
    # Compute Metrics
    # 1. SR Baseline (Bilinear)
    base_psnr = psnr(tir_hr_scaled, tir_sr_base_scaled, data_range=1.0)
    base_ssim = ssim(tir_hr_scaled, tir_sr_base_scaled, data_range=1.0)
    
    # 2. SR Model
    model_psnr = psnr(tir_hr_scaled, tir_sr_pred_scaled, data_range=1.0)
    model_ssim = ssim(tir_hr_scaled, tir_sr_pred_scaled, data_range=1.0)
    
    # 3. Colorization Model
    color_psnr = psnr(rgb_hr_scaled, rgb_pred_scaled, data_range=1.0)
    color_ssim = ssim(rgb_hr_scaled, rgb_pred_scaled, data_range=1.0, channel_axis=2)
    
    # Convert all to base64 images for sending to frontend
    response_data = {
        # Input TIR
        "lr_tir_gray": array_to_base64_png(tir_lr_patch),
        "lr_tir_thermal": array_to_base64_png(tir_lr_patch, colormap='thermal'),
        
        # Ground Truth TIR
        "gt_tir_gray": array_to_base64_png(tir_hr_patch),
        "gt_tir_thermal": array_to_base64_png(tir_hr_patch, colormap='thermal'),
        
        # Baseline SR TIR
        "base_tir_gray": array_to_base64_png(tir_sr_base),
        "base_tir_thermal": array_to_base64_png(tir_sr_base, colormap='thermal'),
        
        # Predicted SR TIR
        "pred_tir_gray": array_to_base64_png(tir_sr_pred),
        "pred_tir_thermal": array_to_base64_png(tir_sr_pred, colormap='thermal'),
        
        # Colorization output
        "pred_rgb": array_to_base64_png(rgb_pred),
        "gt_rgb": array_to_base64_png(rgb_hr_patch),
        
        # Metrics
        "metrics": {
            "sr_base": {"psnr": round(float(base_psnr), 2), "ssim": round(float(base_ssim), 4)},
            "sr_model": {"psnr": round(float(model_psnr), 2), "ssim": round(float(model_ssim), 4)},
            "color_model": {"psnr": round(float(color_psnr), 2), "ssim": round(float(color_ssim), 4)},
            "inference_time_ms": round(float(inference_time), 1)
        },
        "using_trained_weights": has_models
    }
    
    return jsonify(response_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
