import torch
import rasterio
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


def read_raster(path):
    with rasterio.open(path) as src:
        raster = src.read()
        raster = raster.astype(np.float32)
    return torch.from_numpy(raster)

def plot_raster(
        ax,
        img,
        nir=False,
        band_order=[0, 1, 2, 3],
        bilinear=False,
        param_dict={}
    ):
        img = img[band_order, :, :]

        if bilinear:
            img = img.unsqueeze(0)
            img =  nn.functional.interpolate(
                input=img, scale_factor=4, mode="bilinear"
            )
            img = img.squeeze(0)
        if nir:
            img = img[[3, 1, 2], :, :].permute(1, 2, 0).numpy()
        else:
            img = img[:3, :, :].permute(1, 2, 0).numpy()
        img = (img - img.min()) / \
            (img.max() - img.min())
        out = ax.imshow(img, **param_dict)
        ax.axis('off')
        return out

