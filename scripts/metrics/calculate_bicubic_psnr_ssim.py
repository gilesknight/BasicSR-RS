import argparse
import cv2
import torch
import torch.nn as nn
import rasterio
import numpy as np
from os import path as osp

from basicsr.metrics import calculate_psnr_rs, calculate_ssim_rs
from basicsr.utils import bgr2ycbcr, scandir


def main(args):
    """Calculate PSNR and SSIM for images.
    """
    psnr_all = []
    ssim_all = []
    img_list_gt = sorted(list(scandir(args.gt, recursive=True, full_path=True)))
    img_list_lq = sorted(list(scandir(args.lq, recursive=True, full_path=True)))

    for i, img_path in enumerate(img_list_gt):
        basename, ext = osp.splitext(osp.basename(img_path))

        with rasterio.open(img_path) as src:
            img_gt = src.read()
            img_gt = img_gt.astype(np.float32) / 10000
            img_gt = np.clip(img_gt, 0, 1)
            img_gt = img_gt.transpose(1, 2, 0)
        img_gt = (img_gt * 10000).round()
        img_gt = img_gt.astype(np.int16)

        img_path_lq = img_list_lq[i]
        with rasterio.open(img_path_lq) as src:
            img_lq = src.read()
            img_lq = img_lq.astype(np.float32) / 10000
            img_lq = np.clip(img_lq, 0, 1)
            img_lq = torch.from_numpy(img_lq)
        img_interp = nn.functional.interpolate(
            input=img_lq.unsqueeze(0), scale_factor=4, mode="bicubic"
        )
        img_interp = img_interp.squeeze(0).numpy()
        img_interp = img_interp.transpose(1, 2, 0)
        img_interp = (img_interp * 10000).round()
        img_interp = img_interp.astype(np.int16)

        # calculate PSNR and SSIM
        psnr = calculate_psnr_rs(img_gt, img_interp, crop_border=args.crop_border, input_order='HWC', max_pixel=10000)
        ssim = calculate_ssim_rs(img_gt, img_interp, crop_border=args.crop_border, input_order='HWC', max_pixel=10000)
        print(f'{i+1:3d}: {basename:25}. \tPSNR: {psnr:.6f} dB, \tSSIM: {ssim:.6f}')
        psnr_all.append(psnr)
        ssim_all.append(ssim)
    print(args.gt)
    print(args.lq)
    print(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f} dB, SSIM: {sum(ssim_all) / len(ssim_all):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='datasets/PS_UM2020_128/test_dams_only/targets', help='Path to gt (Ground-Truth)')
    parser.add_argument('--lq', type=str, default='datasets/PS_UM2020_128/test_dams_only/inputs', help='Path to lq')
    parser.add_argument('--crop_border', type=int, default=4, help='Crop border for each side')
    args = parser.parse_args()
    main(args)
