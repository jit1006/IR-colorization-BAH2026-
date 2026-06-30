# data/generate_synthetic.py
import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

def generate_noise_map(shape, scale=100.0, octaves=4):
    """Generates a fractal-like noise map using numpy."""
    ny, nx = shape
    grid_y, grid_x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    noise = np.zeros(shape, dtype=np.float32)
    
    for i in range(octaves):
        freq = 2 ** i
        amp = 0.5 ** i
        y_scaled = (grid_y / scale) * freq
        x_scaled = (grid_x / scale) * freq
        # Use sine/cosine combinations to approximate natural noise patterns
        layer = (np.sin(y_scaled) * np.cos(x_scaled) + np.sin(x_scaled * 0.5) * np.cos(y_scaled * 1.5))
        noise += amp * layer
        
    # Normalize to 0-1
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise

def save_geotiff(path, data, crs="EPSG:32643", transform=None, scale_factor=30.0):
    """Saves a 2D numpy array as a GeoTIFF using rasterio."""
    h, w = data.shape
    if transform is None:
        # Standard default origin: UTM zone 43N (India region)
        transform = from_origin(700000, 2500000, scale_factor, scale_factor)
        
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=h,
        width=w,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)

def build_scene(scene_type, shape=(1024, 1024)):
    """
    Builds synthetic B2, B3, B4, and B10 bands with physical correlations.
    Returns: B2, B3, B4 (reflectance), B10 (Kelvin)
    """
    h, w = shape
    ny, nx = shape
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    
    # Base masks
    water_mask = np.zeros(shape, dtype=np.float32)
    urban_mask = np.zeros(shape, dtype=np.float32)
    veg_mask = np.zeros(shape, dtype=np.float32)
    soil_mask = np.zeros(shape, dtype=np.float32)
    
    # 1. Base topography noise
    topo = generate_noise_map(shape, scale=300.0, octaves=3)
    
    if scene_type == "urban":
        # A river running through the city
        water_mask = (np.abs(y - (x * 0.4 + 200 + 50 * np.sin(x/50.0))) < 45).astype(np.float32)
        # Urban grids (streets and buildings)
        urban_mask = (((y % 60 < 20) & (x % 60 < 20)) | ((y % 120 < 10) | (x % 120 < 10))).astype(np.float32)
        # Avoid putting buildings in the river
        urban_mask = np.clip(urban_mask - water_mask, 0, 1)
        # Some parks/vegetation
        veg_mask = ((topo > 0.6) & (water_mask == 0) & (urban_mask == 0)).astype(np.float32)
        # Default soil/background
        soil_mask = (1 - (water_mask + urban_mask + veg_mask))
        
    elif scene_type == "agriculture":
        # A grid of farming plots
        plot_y = (y // 128) % 2
        plot_x = (x // 128) % 2
        # Different crop densities/stages
        veg_mask = ((plot_y == 0) & (plot_x == 0)).astype(np.float32) * 0.9 + \
                   ((plot_y == 1) & (plot_x == 0)).astype(np.float32) * 0.4
        # Bare soil fields
        soil_mask = ((plot_y == 0) & (plot_x == 1)).astype(np.float32) * 0.8 + \
                    ((plot_y == 1) & (plot_x == 1)).astype(np.float32) * 0.9
        # Irrigation canal
        water_mask = (np.abs(x - 512 - 20 * np.sin(y/100.0)) < 15).astype(np.float32)
        # Clear canal from fields
        veg_mask = np.clip(veg_mask - water_mask, 0, 1)
        soil_mask = np.clip(soil_mask - water_mask, 0, 1)
        # Remaining background is dry soil
        bg_mask = 1 - (veg_mask + soil_mask + water_mask)
        soil_mask += np.clip(bg_mask, 0, 1)

    elif scene_type == "forest":
        # Dense forest canopy with a lake
        lake_center_y, lake_center_x = 400, 600
        dist_lake = np.sqrt((y - lake_center_y)**2 + (x - lake_center_x)**2)
        water_mask = (dist_lake < 180).astype(np.float32)
        
        # Dense forest vegetation cover
        veg_noise = generate_noise_map(shape, scale=80.0, octaves=5)
        veg_mask = np.clip(veg_noise * 1.2 - water_mask, 0, 1)
        
        # Clearings/bare ground
        soil_mask = 1 - (veg_mask + water_mask)
        
    elif scene_type == "coastal":
        # Ocean/coast boundary
        coast_line = x * 0.5 + 400 + 80 * np.sin(y / 150.0)
        water_mask = (x < coast_line).astype(np.float32)
        # Beach/sand
        soil_mask = ((x >= coast_line) & (x < coast_line + 40)).astype(np.float32)
        # Coastal scrub / vegetation
        veg_noise = generate_noise_map(shape, scale=120.0, octaves=4)
        veg_mask = np.clip((x >= coast_line + 40).astype(np.float32) * veg_noise * 1.1, 0, 1)
        # Rest is bare ground
        soil_mask += np.clip(1 - (water_mask + soil_mask + veg_mask), 0, 1)

    # 2. Add fine-grained texture noise to reflectances
    fine_noise = generate_noise_map(shape, scale=15.0, octaves=2) * 0.05
    
    # 3. Compute Band Values based on physical properties
    # Landsat-9 reflectance range (scaled 0-1 here, standard DN scale is 0-65535, we model direct physical values for simplicity, then convert to DN)
    # Water: low reflectance in all bands, blue is slightly higher
    # Vegetation: high green, low red, low blue
    # Urban: bright grey, similar across RGB
    # Bare soil: high red, moderate green, lower blue
    
    b2 = water_mask * 0.08 + veg_mask * 0.03 + urban_mask * 0.18 + soil_mask * 0.10 + fine_noise
    b3 = water_mask * 0.05 + veg_mask * 0.16 + urban_mask * 0.18 + soil_mask * 0.14 + fine_noise
    b4 = water_mask * 0.02 + veg_mask * 0.04 + urban_mask * 0.19 + soil_mask * 0.22 + fine_noise
    
    # Clip reflectance to realistic bounds [0.0, 1.0]
    b2 = np.clip(b2, 0.0, 1.0)
    b3 = np.clip(b3, 0.0, 1.0)
    b4 = np.clip(b4, 0.0, 1.0)
    
    # 4. Thermal Infrared Band B10 (Kelvin, typically ranges from 275K to 315K on Earth)
    # Water: cool & stable (282K - 286K)
    # Vegetation: evapotranspiration cooling (288K - 294K)
    # Bare soil: warm/hot (302K - 308K)
    # Urban: very hot roofs & roads (305K - 314K)
    
    thermal_noise = generate_noise_map(shape, scale=40.0, octaves=3) * 2.0
    b10 = water_mask * 284.0 + \
          veg_mask * 291.0 + \
          urban_mask * 309.0 + \
          soil_mask * 304.0 + \
          thermal_noise
          
    # 5. Convert Reflectance and Temperature to standard Landsat-9 Digital Numbers (DN)
    # Landsat 9 SR (B2, B3, B4) scale factor: DN = (Reflectance + 0.2) / 0.0000275
    # Landsat 9 ST (B10) scale factor: DN = (Temperature - 149.0) / 0.00341802
    
    dn_b2 = np.round((b2 + 0.2) / 0.0000275).astype(np.uint16)
    dn_b3 = np.round((b3 + 0.2) / 0.0000275).astype(np.uint16)
    dn_b4 = np.round((b4 + 0.2) / 0.0000275).astype(np.uint16)
    dn_b10 = np.round((b10 - 149.0) / 0.00341802).astype(np.uint16)
    
    return dn_b2, dn_b3, dn_b4, dn_b10

def main():
    scenes = ["urban", "agriculture", "forest", "coastal"]
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    print("Generating synthetic Landsat-9 bands...")
    for scene in scenes:
        print(f"Generating scene: {scene}...")
        b2, b3, b4, b10 = build_scene(scene)
        
        save_geotiff(os.path.join(raw_dir, f"{scene}_B2.tif"), b2)
        save_geotiff(os.path.join(raw_dir, f"{scene}_B3.tif"), b3)
        save_geotiff(os.path.join(raw_dir, f"{scene}_B4.tif"), b4)
        save_geotiff(os.path.join(raw_dir, f"{scene}_B10.tif"), b10)
        
    print("All synthetic bands generated successfully.")

if __name__ == "__main__":
    main()
