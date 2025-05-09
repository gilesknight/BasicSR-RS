import os
import random
import matplotlib.pyplot as plt

from scripts.plot import rs_plots

random.seed(0)

random.randint
DATASET = "PS_UM2020_128"
SUBSET = "test_dams"
OUT_NAME = f"{DATASET}_{SUBSET}"

inputs_dir = f"datasets/{DATASET}/{SUBSET}/inputs/"
targets_dir = f"datasets/{DATASET}/{SUBSET}/targets/"
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

order = [2, 1, 0, 3]
nrows = 4
ncols = 4
num_inputs = len(inputs)
page = int((num_inputs / nrows)) + int((num_inputs % nrows > 0))

out_dir = f"plots/training_samples/{OUT_NAME}/"

os.makedirs(out_dir, exist_ok=True)

start_indx = 0
img_indx = 0
for i in range(0, page):
    fig, ax = plt.subplots(nrows, ncols, figsize=(10, 10))

    for j in range(0, 4):
        idx = random.randint(0, num_inputs)
        input = rs_plots.read_raster(inputs[idx])
        target = rs_plots.read_raster(targets[idx])
        rs_plots.plot_raster(ax[j, 0], input, False, order)
        rs_plots.plot_raster(ax[j, 1], target, False, order)
        ax[j, 0].set_title(os.path.basename(inputs[idx]), fontsize=6)
        ax[j, 1].set_title(os.path.basename(targets[idx]), fontsize=6)


    for j in range(0, 4):
        idx = random.randint(0, num_inputs)
        input = rs_plots.read_raster(inputs[idx])
        target = rs_plots.read_raster(targets[idx])
        rs_plots.plot_raster(ax[j, 2], input, True, order)
        rs_plots.plot_raster(ax[j, 3], target, True, order)
        ax[j, 2].set_title(os.path.basename(inputs[idx]), fontsize=6)
        ax[j, 3].set_title(os.path.basename(targets[idx]), fontsize=6)
        # if j == 0:
        #     ax[0, 2].set_title('LR (NIRGB)')
        #     ax[0, 3].set_title('HR (NIRGB)')

    plt.tight_layout()

    fig.savefig(f"{out_dir}{DATASET}_{SUBSET}_{i}.pdf", dpi=400)

    plt.close()

