import os
import matplotlib.pyplot as plt

from scripts.plot import rs_plots

DATASET = "PS_UM2020_128"
SUBSET = "test_dams"
EXPERIMENT_1 = "004_EDSRRS_Lx4_f256b32_PS_UM2020_128_100000k_B16G1"
EXPERIMENT_2 = "008_ESRGANRS_x4_f64b23_PS_UM2020_128_400000k_B16G1"
NIR=True
OUT_NAME = f"{DATASET}_{SUBSET}"

inputs_dir = f"datasets/{DATASET}/{SUBSET}/inputs/"
targets_dir = f"datasets/{DATASET}/{SUBSET}/targets/"
preds_1_dir = f"results/{EXPERIMENT_1}/visualization/{DATASET}_{SUBSET}/"
preds_2_dir = f"results/{EXPERIMENT_2}/visualization/{DATASET}_{SUBSET}/"
# preds_dir = f"results/006_RRDBNetRS_PSNR_x4_f64b23_UM2018_UM2018_128_100000k_B16G1/visualization/UM2018_UM2018_128_test_dams"
inputs = sorted(
    [os.path.join(inputs_dir, fname)
     for fname in os.listdir(inputs_dir)
     if fname.endswith(".tif") and not fname.startswith(".")]
)
targets = sorted(
    [os.path.join(targets_dir, fname)
     for fname in os.listdir(targets_dir)
     if fname.endswith(".tif") and not fname.startswith(".")]
)
preds_1 = sorted(
    [os.path.join(preds_1_dir, fname)
     for fname in os.listdir(preds_1_dir)
     if fname.endswith(".tif") and not fname.startswith(".")]
)
preds_2 = sorted(
    [os.path.join(preds_2_dir, fname)
     for fname in os.listdir(preds_2_dir)
     if fname.endswith(".tif") and not fname.startswith(".")]
)

assert len(inputs) == len(targets) == len(preds_1)
assert len(inputs) == len(targets) == len(preds_2)



nir=False
order = [2, 1, 0, 3]

nrows = 4
ncols = 5
page = int((len(inputs) / nrows)) + int((len(inputs) % nrows > 0))

if nir:
    out_dir = f"plots/{OUT_NAME}/nir/"
else:
    out_dir = f"plots/{OUT_NAME}/rgb/"
os.makedirs(out_dir, exist_ok=True)

start_indx = 0
img_indx = 0
for i in range(0, page):
    fig, ax = plt.subplots(nrows, ncols, figsize=(12, 10))
    fig.suptitle(OUT_NAME)
    plot_row = 0
    for j in range(img_indx, img_indx + 4):
        input = rs_plots.read_raster(inputs[j])
        target = rs_plots.read_raster(targets[j])
        pred_1 = rs_plots.read_raster(preds_1[j])
        pred_2 = rs_plots.read_raster(preds_2[j])

        rs_plots.plot_raster(ax[plot_row, 0], input, nir, order)
        rs_plots.plot_raster(ax[plot_row, 1], target, nir, order)
        rs_plots.plot_raster(ax[plot_row, 2], input, nir, order, bilinear=True)
        rs_plots.plot_raster(ax[plot_row, 3], pred_1, nir, order)
        rs_plots.plot_raster(ax[plot_row, 4], pred_2, nir, order)

        ax[plot_row, 2].set_title(os.path.basename(inputs[j]), fontsize=10)

        # if plot_row == 0:
        #     ax[plot_row, 0].set_title('Input')
        #     ax[plot_row, 1].set_title('Target')
        #     ax[plot_row, 2].set_title('Bilinear')
        #     ax[plot_row, 3].set_title('EDSR')
        #     ax[plot_row, 4].set_title('ESRGAN')
        plot_row = plot_row + 1
    img_indx = img_indx + 4
    plt.tight_layout()
    if nir:
        fig.savefig(f"{out_dir}{DATASET}_{SUBSET}_nir_{i}.pdf", dpi=400)
    else:
        fig.savefig(f"{out_dir}{DATASET}_{SUBSET}_rgb_{i}.pdf", dpi=400)
    plt.close()






# fig, ax = plt.subplots(1, 4, figsize=(20, 15))


# ax[0].set_title('Input')
# ax[1].set_title('Target')
# ax[2].set_title('Bilinear')
# ax[3].set_title('Prediction')

# fig.savefig("foo.png", dpi=400)