import argparse
import cv2
import glob
import numpy as np
import os
import torch
import rasterio

from basicsr.archs.edsr_arch import EDSRRS
from rasterio.transform import Affine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'experiments/004_EDSRRS_Lx4_f256b32_PS_UM2020_128_100000k_B16G1/models/net_g_100000.pth'  # noqa: E501
    )
    parser.add_argument('--input', type=str, default='datasets/PS_NM_timeseries/dam_115412/inputs', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/PS_NM_timeseries/dam_115412/EDSR', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = EDSRRS(
        num_in_ch=4,
        num_out_ch=4,
        num_feat=256,
        num_block=32,
        upscale=4,
        res_scale=0.1,
        img_range=10000.,
        bgrnir_mean=[0.0614, 0.0914, 0.1265, 0.2484]
    )
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(args.output, exist_ok=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        with rasterio.open(path) as src:
            meta = src.meta.copy()
            transform = src.transform
            img = src.read()
            img = np.moveaxis(img, 0, -1)
            img = img.astype(np.float32) / 10000
            img = np.clip(img, 0, 1)
        img = torch.from_numpy(img.transpose(2, 0, 1).copy())
        img = img.float()
        img = img.unsqueeze(0).to(device)
        # inference
        try:
            with torch.no_grad():
                output = model(img)
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = (output * 10000).round().astype(np.int16)
            sr_meta = meta
            sr_meta.update({
                "height": meta["height"] * 4,
                "width":  meta["width"] * 4,
                "transform": transform * Affine.scale(1 / 4)
            })
            with rasterio.open(os.path.join(args.output, f'{imgname}_EDSR.tif'), 'w', **sr_meta) as src:
                src.write(output)


if __name__ == '__main__':
    main()
