import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop, paired_random_water_crop, rs_augment, paired_centred_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor, rs_imfrombytes, rs_img2tensor, RasterioReader
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class RSWaterSegmentationDataset(data.Dataset):
    """Remote Sensing  dataset for image segmentation.

    Read HR image and mask pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt mask.
        dataroot_lq (str): Data root path for lq image.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(RSWaterSegmentationDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_rescale_val = opt['gt_rescale_val']
        self.lq_rescale_val = opt['lq_rescale_val']
        self.gt_clip = opt["gt_clip"]
        self.lq_clip = opt["lq_clip"]
        self.patch_size = 128
        self.min_water_percent = 0.25

        self.rasterio_reader = RasterioReader()

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt']
            )
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'],
                self.filename_tmpl
            )
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder],
                ['lq', 'gt'], self.filename_tmpl
            )

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt
            )

        img_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(img_path, 'gt')
        img = rs_imfrombytes(
            img_bytes, self.rasterio_reader,
            float32=True,
            rescale_val=self.gt_rescale_val,
            clip=self.gt_clip
        )
        mask_path = self.paths[index]['lq_path']
        mask_bytes = self.file_client.get(mask_path, 'lq')
        mask = rs_imfrombytes(mask_bytes, self.rasterio_reader)

        if self.opt['phase'] == 'train':
            mask, img = paired_random_water_crop(
                mask, img, self.patch_size, self.min_water_percent, mask_path
            )
            mask, img = rs_augment(
                [mask, img], self.opt['use_hflip'], self.opt['use_rot']
            )

        if self.opt['phase'] != 'train':
            mask, img = paired_centred_crop(mask, img, 128, mask_path)

        mask = rs_img2tensor(mask, float32=False, bf16=False)
        img = rs_img2tensor(img, float32=True, bf16=False)

        if self.mean is not None and self.std is not None:
            normalize(img, self.mean, self.std, inplace=True)

        return {
            'lq': img, 'gt': mask, 'lq_path': img_path, 'gt_path': mask_path
        }

    def __len__(self):
        return len(self.paths)