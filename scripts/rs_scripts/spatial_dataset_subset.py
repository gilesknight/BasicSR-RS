import os
import shutil
import rasterio
import geopandas as gpd

from shapely.geometry import box

data_dir = "datasets/PS_UM2020_128/test/"
input_dir = data_dir + "inputs/"
dam_points_path = "datasets/misc/dam_points_UM2020_test_32750.gpkg"

test_dams_dir = "datasets/PS_UM2020_128/test_dams/"
test_dams_inputs_dir = "datasets/PS_UM2020_128/test_dams/inputs/"
test_dams_targets_dir = "datasets/PS_UM2020_128/test_dams/targets/"
os.makedirs(test_dams_dir, exist_ok=True)
os.makedirs(test_dams_inputs_dir, exist_ok=True)
os.makedirs(test_dams_targets_dir, exist_ok=True)

dam_points = gpd.read_file(dam_points_path)
inputs = sorted(
    [os.path.join(input_dir, fname)
     for fname in os.listdir(input_dir)
     if fname.endswith(".tif") and not fname.startswith(".")]
)

input_bounds = {inp: box(*rasterio.open(inp).bounds) for inp in inputs}

for index, row in dam_points.iterrows():
    print(f"Dam {index}")
    for input, bbox in input_bounds.items():
        if bbox.contains(row["geometry"]):
            print(f"Enveloped by patch {input}")
            target = input.replace("/inputs/", "/targets/")
            shutil.copyfile(
                input,
                os.path.join(test_dams_inputs_dir, os.path.basename(input))
            )
            shutil.copyfile(
                target,
                os.path.join(test_dams_targets_dir, os.path.basename(target))
            )
            del input_bounds[input]
            break