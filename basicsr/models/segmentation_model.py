import torch
import copy
import rasterio
import numpy as np
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img, rs_tensor2img, rs_imwrite
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


class SegModelRS(BaseModel):

    def __init__(self, opt):
        super(SegModelRS, self).__init__(opt)

        self.net = build_network(opt["network_seg"])
        self.net = self.model_to_device(self.net)
        self.print_network(self.net)

        load_path = self.opt['path'].get('pretrain_network_seg')
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_s', 'params')
            self.load_network(
                self.net,
                load_path,
                self.opt['path'].get('strict_load_s', True),
                param_key
            )

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}'
            )
            self.net_ema = build_network(self.opt['network_seg']).to(self.device)
            load_path = self.opt['path'].get('pretain_network_seg', None)
            if load_path is not None:
                self.load_network(
                    self.net_ema,
                    load_path,
                    self.opt['path'].get('strict_load_g', True),
                    'params_ema'
                )
            else:
                self.model_ema(0)
            self.net_ema.eval()

        if train_opt.get('seg_loss'):
            self.seg_loss = build_loss(train_opt['seg_loss']).to(self.device)
        else:
            raise ValueError("Segmentation loss is None.")

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f"Params {k} will not be optimized.")

        optim_type = train_opt['optim'].pop('type')
        self.optimizer = self.get_optimizer(
            optim_type,
            optim_params,
            **train_opt['optim']
        )
        self.optimizers.append(self.optimizer)

    def feed_data(self, data):
        self.img = data['img'].to(self.device)
        if 'mask' in data:
            self.mask = data['mask'].to(self.device)

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.output = self.net(self.img)

        l_total = 0
        loss_dict = OrderedDict()

        if self.seg_loss:
            l_seg = self.seg_loss(self.output, self.mask)
            l_total += l_seg
            loss_dict['l_seg'] = l_seg

        l_total.backward()
        self.optimizer.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_ema'):
            self.net_ema.eval()
            with torch.no_grad():
                self.output = self.net_ema(self.img)
        else:
            self.net.eval()
            with torch.no_grad():
                self.output = self.net(self.img)
            self.net.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img
            )

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {
                    metric: 0 for metric in self.opt['val']['metrics'].keys()
                }
            self._initialize_best_metric_results(dataset_name)
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_path = val_data('img_path')[0]
            mask_dtype = self.opt['rs_options']['datasets']['mask_dtype']
            mask_dtype = np.dtype(mask_dtype)

            mask_rescale = self.opt['rs_options']['datasets']['mask_rescale_val']
            img_name = osp.splitext(osp.basename(img_path))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            pred_mask = rs_tensor2img(
                [visuals['result']], False, out_type=mask_dtype
            )
            metric_data['img'] = pred_mask
            if 'mask' in visuals:
                mask = rs_tensor2img(
                    [visuals['mask']], False, out_type=mask_dtype
                )
                metric_data['img2'] = mask
                del self.mask

            del self.img
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                with rasterio.open(img_path) as src:
                    new_profile = copy.deepcopy(src.profile)
                new_profile['dtype'] = mask_dtype
                new_profile['count'] = 1
                if self.opt['is_train']:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'],
                        img_name,
                        f'{img_name}_{current_iter}.tif'
                    )
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'],
                            dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.tif'
                        )
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'],
                            dataset_name,
                            f'{img_name}_{self.opt["name"]}.tif'
                        )
                rs_imwrite(pred_mask, save_img_path, new_profile)

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(
                        metric_data, opt_
                    )
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f"Test {img_name}")
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(
                    dataset_name,
                    metric,
                    self.metric_results[metric],
                    current_iter
                )
            self._log_validation_metric_values(
                current_iter, dataset_name, tb_logger
            )

    def _log_validation_metric_values(
            self, current_iter, dataset_name, tb_logger
        ):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(
                    f'metrics/{dataset_name}/{metric}', value, current_iter
                )

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['img'] = self.img.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'mask'):
            out_dict['mask'] = self.mask.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_ema'):
            self.save_network(
                [self.net, self.net_ema],
                'net',
                current_iter,
                param_key=['params', 'params_ema']
            )
        else:
            self.save_network(self.net, 'net', current_iter)
        self.save_training_state(epoch, current_iter)