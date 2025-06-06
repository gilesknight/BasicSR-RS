import rasterio
from os import path as osp
from PIL import Image

from basicsr.utils import scandir


def generate_meta_info_div2k():
    """Generate meta info for DIV2K dataset.
    """

    gt_folder = 'datasets/DIV2K/DIV2K_train_HR_sub/'
    meta_info_txt = 'basicsr/data/meta_info/meta_info_DIV2K800sub_GT.txt'

    img_list = sorted(list(scandir(gt_folder)))

    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            img = Image.open(osp.join(gt_folder, img_path))  # lazy load
            width, height = img.size
            mode = img.mode
            if mode == 'RGB':
                n_channel = 3
            elif mode == 'L':
                n_channel = 1
            else:
                raise ValueError(f'Unsupported mode {mode}.')

            info = f'{img_path} ({height},{width},{n_channel})'
            print(idx + 1, info)
            f.write(f'{info}\n')

def generate_meta_info_UM2018_UM2018_128():

    # gt_folder = 'datasets/UM2018_UM2018_128/train/targets/'
    # meta_info_txt = 'basicsr/data/meta_info/meta_info_UM2018_UM2018_128.txt'

    gt_folder = "datasets/UM2018_UM2018_128_div8/valid/targets/"
    meta_info_txt = 'basicsr/data/meta_info/meta_info_UM2018_UM2018_128_div8_valid.txt'

    img_list = sorted(list(scandir(gt_folder)))
    with open(meta_info_txt, 'w') as f:
        for idx, img_path in enumerate(img_list):
            with rasterio.open(gt_folder + img_path) as src:
                info = f'{img_path} ({src.height},{src.width},{src.count})'
            print(idx + 1, info)
            f.write(f'{info}\n')

if __name__ == '__main__':
    #generate_meta_info_div2k()
    generate_meta_info_UM2018_UM2018_128()
