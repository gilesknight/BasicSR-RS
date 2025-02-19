import os
import rasterio
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from multiprocessing import Pool, cpu_count

DS_DIR = "datasets/UM2018_UM2018_128/valid/inputs/"
OUTPUT_DIR = "datasets/misc/UM2018_UM2018_128/"
OUTPUT_GPKG = OUTPUT_DIR + "UM2018_UM2018_128_valid_points.gpkg"
CRS = "EPSG:28350"
NUM_WORKERS = min(20, cpu_count())

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_patch(patch_file):
    """Extracts midpoint and metadata from a raster file"""
    patch_path = os.path.join(DS_DIR, patch_file)
    try:
        with rasterio.open(patch_path) as src:
            bounds = src.bounds
            mid_x = (bounds.left + bounds.right) / 2
            mid_y = (bounds.bottom + bounds.top) / 2
            point = Point(mid_x, mid_y)
            group_name = "_".join(os.path.splitext(patch_file)[0].split("_")[:-2])
            return {'group': group_name, 'geometry': point}
    except Exception as e:
        print(f"Error processing {patch_file}: {e}")
        return None

def main():
    patch_files = [entry.name for entry in os.scandir(DS_DIR) if entry.is_file()]

    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(process_patch, patch_files, chunksize=5000)

    # Filter out None values (errors)
    results = [r for r in results if r]

    # Create GeoDataFrame in bulk
    ds = gpd.GeoDataFrame(results, crs=CRS)

    # Save to file
    ds.to_file(OUTPUT_GPKG, driver="GPKG")

if __name__ == "__main__":
    main()
