import os
import rasterio
from rasterio.enums import Resampling
import torch
import torch.nn.functional as F
import numpy as np

def load_geotiff(file_path):
    with rasterio.open(file_path) as src:
        data = src.read()
        profile = src.profile
    return data, profile

def save_geotiff(data, profile, output_path):
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data)

def interpolate_tensor(tensor, scale_factor):
    # tensor shape: (bands, height, width)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    resized = F.interpolate(tensor, scale_factor=scale_factor, mode='bicubic', align_corners=True)
    return resized.squeeze(0)  # Remove batch dimension

def resample_geotiff(input_path, output_path, scale_factor=4.0):
    data, profile = load_geotiff(input_path)

    # Convert to torch tensor
    tensor = torch.from_numpy(data).float()

    # Perform bicubic interpolation
    resized_tensor = interpolate_tensor(tensor, scale_factor)

    # Convert back to numpy
    resized_data = resized_tensor.numpy()

    # Update profile
    profile.update({
        'height': resized_data.shape[1],
        'width': resized_data.shape[2],
        'transform': rasterio.Affine(
            profile['transform'].a / scale_factor, profile['transform'].b, profile['transform'].c,
            profile['transform'].d, profile['transform'].e / scale_factor, profile['transform'].f
        )
    })

    # Save the new raster
    save_geotiff(resized_data, profile, output_path)

def process_directory(input_dir, output_dir, scale_factor=4.0):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_resampled.tif")
            print(f"Processing: {filename}")
            resample_geotiff(input_path, output_path, scale_factor)

# Example usage
if __name__ == "__main__":
    input_directory = "datasets/PS_UM2020_128/test_dams/inputs/"
    output_directory = "results/bicubic/"
    process_directory(input_directory, output_directory, scale_factor=4.0)
