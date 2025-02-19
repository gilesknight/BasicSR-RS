import os
import rasterio
import numpy as np


train_patch_dir = "datasets/UM2018_UM2018_128/train/inputs/"

train_paths = sorted(
    [os.path.join(train_patch_dir, fname)
     for fname in os.listdir(train_patch_dir)
     if fname.endswith(".tif")])

b_total = 0.0
g_total = 0.0
r_total = 0.0
nir_total = 0.0
num_imgs = len(train_paths)

train_pixel_total = num_imgs * 32 * 32

for i in range(0, num_imgs):
    with rasterio.open(train_paths[i]) as src:
        img = src.read()
        if i == 0: print(img.dtype)
        img = img.astype(np.float32) / 10000
        img = np.clip(img, 0, 1)
    b_total += np.sum(img[0, :, :])
    g_total += np.sum(img[1, :, :])
    r_total += np.sum(img[2, :, :])
    nir_total += np.sum(img[3, :, :])
    if i % 1000 == 999:
        print(f"{i + 1}/{num_imgs} images summed")

r_mean = (r_total / train_pixel_total)
b_mean = (b_total / train_pixel_total)
g_mean = (g_total / train_pixel_total)
nir_mean = (nir_total / train_pixel_total)

print("\n")
print(f"B band mean: {b_mean}")
print(f"G band mean: {g_mean}")
print(f"R band mean: {r_mean}")
print(f"NIR band mean: {nir_mean}")


# PS_UM2020_128
# B band mean: 0.06144250033747977 (0.0614)
# G band mean: 0.09135907478994558 (0.0914)
# R band mean: 0.12646609250927754 (0.1265)
# NIR band mean: 0.24840672442695283 (0.2484)

# UM2018_UM2018_128
# B band mean: 0.2017289462367599
# G band mean: 0.1875601688839594
# R band mean: 0.18222907919617828
# NIR band mean: 0.45329034328102