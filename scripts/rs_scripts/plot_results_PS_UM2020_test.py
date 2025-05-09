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


edsr_model = "004_EDSRRS_Lx4_f256b32_PS_UM2020_128_100000k_B16G1"
esrgan_model = "008_ESRGANRS_x4_f64b23_PS_UM2020_128_400000k_B16G1"

lr_inputs_dir = f"datasets/PS_UM2020_128/test/inputs/"
hr_targets_dir = f"datasets/PS_UM2020_128/test/targets/"
edsr_preds_dir = f"results/{edsr_model}/visualization/PS_UM2020_128_test/"
esrgan_preds_dir = f"results/{esrgan_model}/visualization/PS_UM2020_128_test/"

out_dir = f"plots/sample_results/PS_UM2020_128/"
os.makedirs(out_dir, exist_ok=True)
order = [3, 2, 1, 0]
#order = [2, 1, 0, 3]
#         r g b nir
# [3, 0, 1, 2]
nir = False
nrows = 6
ncols = 5

width_ratios = [1, 1, 1, 1, 1]
fig, ax = plt.subplots(
    nrows, ncols, figsize=(10, 13.25), gridspec_kw={'width_ratios': width_ratios}
)

fnames = [
    #"2134NW_jumpSW_20200202_020252_1013_3B_4015_0",
    #"2134SE_chidNW_20200202_020056_1040_3B_3493_0",
    #"2035SW_moorSE_20200202_014211_1054_3B_3628_0",
    #"2032NE_pinjSE_20200129_033718_08_106c_3092_0",
    #"2032NE_pinjSE_20200129_033718_08_106c_3092_0",
    #"2035SW_moorSE_20200202_014211_1054_3B_5678_0",
    #"2134NW_jumpSW_20200202_020252_1013_3B_4841_0",
    #"2032NE_pinjNE_20200129_033722_18_106c_3402_0",
    #"2032NE_pinjSE_20200129_033720_13_106c_3499_0",
    #"2032NE_pinjNE_20200129_033722_18_106c_2958_0",
    #"2134NW_jumpSW_20200202_020252_1013_3B_1227_0",
    #"2134SE_chidNW_20200202_020058_1040_3B_271_0",
    #"2035SW_moorSE_20200202_014211_1054_3B_5124_0",
    #"2032NE_pinjNE_20200129_033720_13_106c_2091_0",
    #"2134NW_jumpSW_20200202_020252_1013_3B_3052_0",
    #"2032NE_pinjSE_20200129_033718_08_106c_2550_0",
    #"2032NE_pinjSE_20200129_033720_13_106c_3810_0",
    #"2134SE_chidNW_20200202_020056_1040_3B_4703_0",
    #"2134NW_jumpSW_20200202_020252_1013_3B_3132_0",
    #"2134NW_jumpSW_20200202_020252_1013_3B_4204_0",
    "2032NE_pinjSE_20200129_033720_13_106c_4282_0",
    "2035SW_moorSE_20200202_014211_1054_3B_5396_0",
    "2032NE_pinjSE_20200129_033718_08_106c_7084_0",
    "2134NW_jumpSW_20200202_020252_1013_3B_2550_0",
    "2134NW_jumpSW_20200202_020252_1013_3B_7560_0",
    "2032NE_pinjSE_20200129_033720_13_106c_3927_0"
    #"2032NE_pinjSE_20200129_033720_13_106c_3343_0",
]
plot_column(ax, nrows, 0, fnames, lr_inputs_dir, nir, "LR")
plot_column(ax, nrows, 1, fnames, hr_targets_dir, nir, "HR", metrics=True)
plot_column(
    ax, nrows, 2, fnames, lr_inputs_dir, nir, "Bicubic", bicubic=True,
    metrics=True, target_dir=hr_targets_dir
)
plot_column(
    ax, nrows, 3, fnames, edsr_preds_dir, nir, "EDSR", model=edsr_model,
    metrics=True, target_dir=hr_targets_dir
)
plot_column(
    ax, nrows, 4, fnames, esrgan_preds_dir, nir, "ESRGAN", model=esrgan_model,
    metrics=True, target_dir=hr_targets_dir
)



plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)

fig.savefig(f"{out_dir}sample_results_test.pdf", dpi=400)

plt.close()

