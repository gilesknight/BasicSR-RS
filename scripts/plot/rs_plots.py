import torch
import rasterio
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def read_raster(path):
    with rasterio.open(path) as src:
        raster = src.read()
        raster = raster.astype(np.float32)
    return raster


def bicubic(img):
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = nn.functional.interpolate(
            input=img, scale_factor=4, mode="bicubic"
    )
    img = img.squeeze(0)
    return img.numpy()


def plot_raster(
        ax,
        img,
        nir=False,
        band_order=[0, 1, 2, 3],
        bilinear=False,
        bicubic=False,
        scalebar=False,
        scalebar_px_len=0,
        scalebar_px_label="",
        scalebar_font_size=None,
        param_dict={}
    ):
        img = torch.from_numpy(img)
        img = img[band_order, :, :]
        if scalebar_font_size is None:
            font_properties = scalebar_font_size
        else:
            font_properties = fm.FontProperties(size=scalebar_font_size)
        sbar = AnchoredSizeBar(
            ax.transData,
            scalebar_px_len,
            scalebar_px_label,
            'lower center',
            pad=0.1,
            color='white',
            frameon=False,
            size_vertical=5,
            fontproperties=font_properties
        )
        if bilinear:
            img = img.unsqueeze(0)
            img =  nn.functional.interpolate(
                input=img, scale_factor=4, mode="bilinear"
            )
            img = img.squeeze(0)
        if bicubic:
            img = img.unsqueeze(0)
            img = nn.functional.interpolate(
                  input=img, scale_factor=4, mode="bicubic"
            )
            img = img.squeeze(0)
        if nir:
            img = img[[3, 0, 1], :, :].permute(1, 2, 0).numpy()
        else:
            img = img[:3, :, :].permute(1, 2, 0).numpy()
        for i in range(3):
            band = img[:, :, i]
            img[:, :, i] = (band - band.min()) / (band.max() - band.min())
        # img = (img - img.min()) / \
        #     (img.max() - img.min())
        out = ax.imshow(img, **param_dict)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if scalebar:
            ax.add_artist(sbar)
        #ax.axis('off')
        # ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        return out

