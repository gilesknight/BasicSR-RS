import os
import matplotlib.pyplot as plt

from scripts.plot import rs_plots

import matplotlib.font_manager as fm


plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.sans-serif"] = ["Helvetica"]

def plot_column(
        ax, nrows, colnum, fnames, f_dir, nir, title, order=[2, 1, 0, 3]
    ):
    for i in range(0, nrows):
        raster = rs_plots.read_raster(f_dir + fnames[i])
        rs_plots.plot_raster(ax[i, colnum], raster, nir, order)
        if i == 0:
            ax[0, colnum].set_title(title, fontsize=16)

def plot_spacer_column(ax, nrows, colnum):
    for i in range(0, nrows):
        ax[i, colnum].axhline(0, color='white', lw=1)
        ax[i, colnum].axis('off')

um_inputs_dir = f"datasets/UM2018_UM2018_128/train/inputs/"
um_targets_dir = f"datasets/UM2018_UM2018_128/train/targets/"
ps_inputs_dir = f"datasets/PS_UM2020_128/train/inputs/"
ps_targets_dir = f"datasets/PS_UM2020_128/train/targets/"

out_dir = f"plots/training_sample/train_samples/"
os.makedirs(out_dir, exist_ok=True)
order = [2, 1, 0, 3]
nrows = 4
ncols = 9

width_ratios = [1, 1, 1, 1, 0.05, 1, 1, 1, 1]
fig, ax = plt.subplots(
    nrows, ncols, figsize=(18.85, 10), gridspec_kw={'width_ratios': width_ratios}
)
fig.text(0.25, 0.975, 'Urban Monitor 2018 Dataset', ha='center', fontsize=18)
fig.text(0.75, 0.975, 'PlanetScope Urban Monitor 2020 Dataset', ha='center', fontsize=18)

fnames = [
    "2034NW_yancSE_1835.tif", "2132SW_nangNW_8902.tif",
    "2033NE_fremNE_69.tif", "2034NW_yancNE_9463.tif"
]
plot_column(ax, 4, 0, fnames, um_inputs_dir, False, "LR (RGB)")
plot_column(ax, 4, 1, fnames, um_targets_dir, False, "HR (RGB)")

fnames = [
    "2033SW_rockNE_8624.tif", "2034SE_pertNW_14692.tif",
    "2034NW_yancNE_14983.tif", "2032NE_pinjNW_2590.tif"

]
plot_column(ax, 4, 2, fnames, um_inputs_dir, True, "LR (NIR-RG)")
plot_column(ax, 4, 3, fnames, um_targets_dir, True, "HR (NIR-RG)")

plot_spacer_column(ax, 4, 4)

fnames = [
    "2032SE_hameSE_20200129_033713_98_106c_116_0.tif",
    "2032SE_hameNW_20200209_020302_1039_3B_3096_0.tif",
    "2034NE_muchSE_20200202_020252_1013_3B_2324_0.tif",
    "2134SW_mundNW_20200202_020255_1013_3B_6643_0.tif"
]
plot_column(ax, 4, 5, fnames, ps_inputs_dir, False, "LR (RGB)")
plot_column(ax, 4, 6, fnames, ps_targets_dir, False, "HR (RGB)")

fnames = [
    "2134SW_mundNW_20200202_020255_1013_3B_2251_0.tif",
    "2134NE_toodSW_20200202_020055_1040_3B_6944_0.tif",
    "2134SW_mundNE_20200202_020254_1013_3B_4100_0.tif",
    "2032NE_pinjNW_20200209_022415_80_105d_1974_0.tif"
]
plot_column(ax, 4, 7, fnames, ps_inputs_dir, True, "LR (NIR-RG)")
plot_column(ax, 4, 8, fnames, ps_targets_dir, True, "HR (NIR-RG)")

plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5, rect=[0, 0, 1, 0.95])

fig.savefig(f"{out_dir}dataset_samples.pdf", dpi=400)

plt.close()

