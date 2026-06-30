# data/preprocess.py
import os
import numpy as np
import rasterio
from skimage.transform import resize

def load_band(path):
    """Loads a single band from a GeoTIFF."""
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), src.profile

def convert_to_physical(b2, b3, b4, b10):
    """Converts Landsat-9 raw Digital Numbers (DN) to physical units."""
    # RGB reflectance: DN * 0.0000275 - 0.2
    # Clip to [0.0, 1.0]
    r = np.clip(b4 * 0.0000275 - 0.2, 0.0, 1.0)
    g = np.clip(b3 * 0.0000275 - 0.2, 0.0, 1.0)
    b = np.clip(b2 * 0.0000275 - 0.2, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    
    # TIR Temperature in Kelvin: DN * 0.00341802 + 149.0
    tir_kelvin = b10 * 0.00341802 + 149.0
    return rgb, tir_kelvin

def downscale_image(img, factor):
    """Downscales image by a factor using anti-aliasing to mimic optical blur."""
    h, w = img.shape[:2]
    new_h, new_w = int(round(h / factor)), int(round(w / factor))
    
    # anti_aliasing=True applies a Gaussian filter before downsampling
    if img.ndim == 3:
        return resize(img, (new_h, new_w, img.shape[2]), anti_aliasing=True, preserve_range=True)
    else:
        return resize(img, (new_h, new_w), anti_aliasing=True, preserve_range=True)

def extract_patches(img, patch_size, stride):
    """Extracts overlapping patches from an image."""
    patches = []
    h, w = img.shape[:2]
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(img[y:y+patch_size, x:x+patch_size])
            
    return np.array(patches)

def normalize_data(rgb, tir):
    """Normalizes RGB to [-1, 1] and TIR Kelvin to [0, 1] (based on range [270, 320])."""
    # RGB to [-1, 1] for Pix2Pix Tanh generator
    rgb_norm = rgb * 2.0 - 1.0
    
    # TIR Kelvin to [0, 1] based on standard Earth temperatures [270K, 320K]
    tir_norm = (tir - 270.0) / (320.0 - 270.0)
    tir_norm = np.clip(tir_norm, 0.0, 1.0)
    
    return rgb_norm, tir_norm

def main():
    scenes = ["urban", "agriculture", "forest", "coastal"]
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Patches placeholders
    all_tir_lr = []  # 200m (64x64)
    all_tir_hr = []  # 100m (128x128)
    all_rgb_hr = []  # 100m (128x128)
    
    print("Preprocessing scenes...")
    for scene in scenes:
        print(f"Processing scene: {scene}...")
        
        # Load raw bands
        b2, _ = load_band(os.path.join(raw_dir, f"{scene}_B2.tif"))
        b3, _ = load_band(os.path.join(raw_dir, f"{scene}_B3.tif"))
        b4, _ = load_band(os.path.join(raw_dir, f"{scene}_B4.tif"))
        b10, _ = load_band(os.path.join(raw_dir, f"{scene}_B10.tif"))
        
        # Convert to physical units
        rgb, tir = convert_to_physical(b2, b3, b4, b10)
        
        # Downscale: Original is 30m resolution.
        # Downscaling to 100m represents 3.333x downscaling
        # Downscaling to 200m represents 6.667x downscaling
        rgb_100m = downscale_image(rgb, 3.3333)
        tir_100m = downscale_image(tir, 3.3333)
        tir_200m = downscale_image(tir, 6.6667)
        
        # Normalize
        rgb_100m_norm, tir_100m_norm = normalize_data(rgb_100m, tir_100m)
        _, tir_200m_norm = normalize_data(np.zeros_like(tir_200m), tir_200m) # only normalize tir
        
        # Extract paired patches:
        # 100m target patches: 128x128, stride 32
        # 200m input patches: 64x64, stride 16 (since it's exactly 2x coarser)
        # This keeps them perfectly aligned spatially!
        
        scene_rgb_patches = extract_patches(rgb_100m_norm, 128, 32)
        scene_tir_hr_patches = extract_patches(tir_100m_norm, 128, 32)
        scene_tir_lr_patches = extract_patches(tir_200m_norm, 64, 16)
        
        # Ensure number of patches is equal
        min_patches = min(len(scene_rgb_patches), len(scene_tir_hr_patches), len(scene_tir_lr_patches))
        
        all_rgb_hr.append(scene_rgb_patches[:min_patches])
        all_tir_hr.append(scene_tir_hr_patches[:min_patches])
        all_tir_lr.append(scene_tir_lr_patches[:min_patches])
        
        print(f"Scene {scene}: extracted {min_patches} patches.")
        
    # Concatenate and save
    all_rgb_hr = np.concatenate(all_rgb_hr, axis=0)
    all_tir_hr = np.concatenate(all_tir_hr, axis=0)
    all_tir_lr = np.concatenate(all_tir_lr, axis=0)
    
    # Save as compressed numpy files
    np.savez_compressed(
        os.path.join(processed_dir, "dataset.npz"),
        tir_lr=all_tir_lr,
        tir_hr=all_tir_hr,
        rgb_hr=all_rgb_hr
    )
    
    print(f"Preprocessed dataset saved at {os.path.join(processed_dir, 'dataset.npz')}")
    print(f"Total patches: {len(all_rgb_hr)}")
    print(f"TIR LR shape: {all_tir_lr.shape}")
    print(f"TIR HR shape: {all_tir_hr.shape}")
    print(f"RGB HR shape: {all_rgb_hr.shape}")

if __name__ == "__main__":
    main()
