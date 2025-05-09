import os
import matplotlib.pyplot as plt

from scripts.plot import rs_plots
from basicsr.metrics import calculate_psnr_rs, calculate_ssim_rs


plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.sans-serif"] = ["Helvetica"]

def plot_column(
        ax, nrows, colnum, fnames, f_dir, nir, title,
        order=[2, 1, 0, 3], bicubic=False, model=None,
        metrics=False, target_dir=None
    ):
    for i in range(0, nrows):
        if model is not None:
            fname = fnames[i] + f"_{model}.tif"
        else:
            fname = fnames[i] + ".tif"
        raster = rs_plots.read_raster(f_dir + fname)
        rs_plots.plot_raster(ax[i, colnum], raster, nir, order, bicubic=bicubic)
        if metrics:
            if target_dir is None:
                ax[i, colnum].set_xlabel("(PSNR / SSIM)", fontsize=12)
            else:
                if bicubic:
                    raster = rs_plots.bicubic(raster)
                target = rs_plots.read_raster(target_dir + fnames[i] + ".tif")
                psnr = calculate_psnr_rs(target, raster, 4, 10000, 'CHW')
                ssim = calculate_ssim_rs(target, raster, 4, 10000, 'CHW')
                ax[i, colnum].set_xlabel(
                    f"({'{:.2f}'.format(round(psnr, 2))} dB / " +
                    f"{'{:.4f}'.format(round(ssim, 4))})",
                    fontsize=12
                )
        if i == 0:
            ax[0, colnum].set_title(title, fontsize=14)

def plot_spacer_column(ax, nrows, colnum):
    for i in range(0, nrows):
        ax[i, colnum].axhline(0, color='white', lw=1)
        ax[i, colnum].axis('off')


edsr_model = "003_EDSRRS_Lx4_f256b32_UM2018_UM2018_128_100000k_B16G1"
rrdbnet_model = "006_RRDBNetRS_PSNR_x4_f64b23_UM2018_UM2018_128_100000k_B16G1"

lr_inputs_dir = f"datasets/UM2018_UM2018_128/test/inputs/"
hr_targets_dir = f"datasets/UM2018_UM2018_128/test/targets/"
edsr_preds_dir = f"results/{edsr_model}/visualization/UM2018_UM2018_128_test/"
rrdbnet_preds_dir = f"results/{rrdbnet_model}/visualization/UM2018_UM2018_128_test/"

out_dir = f"plots/sample_results/UM2018_UM2018_128/"
os.makedirs(out_dir, exist_ok=True)
order = [2, 1, 0, 3]
nrows = 6
ncols = 5

width_ratios = [1, 1, 1, 1, 1]
fig, ax = plt.subplots(
    nrows, ncols, figsize=(10, 13.25), gridspec_kw={'width_ratios': width_ratios}
)

fnames = [
    "2032NE_pinjSE_5089", "2035SW_moorSE_4936",
    "2032NE_pinjNE_5809", "2134NW_jumpSW_2935",
    #"2032NE_pinjNE_17183",
    "2134SE_chidNW_15039",
    #"2035SW_moorSE_7820",
    "2032NE_pinjSE_18249",
]
plot_column(ax, nrows, 0, fnames, lr_inputs_dir, False, "LR")
plot_column(ax, nrows, 1, fnames, hr_targets_dir, False, "HR", metrics=True)
plot_column(
    ax, nrows, 2, fnames, lr_inputs_dir, False, "Bicubic", bicubic=True,
    metrics=True, target_dir=hr_targets_dir
)
plot_column(
    ax, nrows, 3, fnames, edsr_preds_dir, False, "EDSR", model=edsr_model,
    metrics=True, target_dir=hr_targets_dir
)
plot_column(
    ax, nrows, 4, fnames, rrdbnet_preds_dir, False, "RRDBNet", model=rrdbnet_model,
    metrics=True, target_dir=hr_targets_dir
)



plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)

fig.savefig(f"{out_dir}sample_results.pdf", dpi=400)

plt.close()

