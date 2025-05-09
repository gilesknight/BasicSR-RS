import os
import matplotlib.pyplot as plt

from datetime import datetime
from scripts.plot import rs_plots

DATASET = "PS_NM_timeseries"
SUBSET = "dam_113992"
MODELS = ["ESRGAN", "EDSR"]

inputs_dir = f"datasets/{DATASET}/{SUBSET}/inputs/"
targets_dir = f"datasets/{DATASET}/{SUBSET}/targets/"
preds_dir = [f"results/{DATASET}/{SUBSET}/{model}/" for model in MODELS]

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
preds = [
    sorted(
        [os.path.join(pred_dir, fname)
        for fname in os.listdir(pred_dir)
        if fname.endswith(".tif") and not fname.startswith(".")]
    )
    for pred_dir in preds_dir
]

nir=False
order = [2, 1, 0, 3]
num_models = len(MODELS)

nrows = 2 + num_models
ncols = 6

if nir:
    out_dir = f"plots/{DATASET}/{SUBSET}/nir/"
else:
    out_dir = f"plots/{DATASET}/{SUBSET}/rgb/"
os.makedirs(out_dir, exist_ok=True)


width_ratios = [1 for i in range(0, ncols)]
fig, ax = plt.subplots(
    nrows, ncols, figsize=(ncols * 2, nrows * 2),
    gridspec_kw={'width_ratios': width_ratios}
)
title_font_size = plt.rcParams['axes.titlesize']
for j in range(0, ncols):
    target_date = os.path.basename(targets[j]).split(".")[0][3:]
    input_date = os.path.basename(inputs[j]).split(".")[0][3:]
    target_date = datetime.strptime(target_date, "%Y%m%d")
    input_date = datetime.strptime(input_date, "%Y%m%d")
    if target_date == input_date:
        date_label = target_date.strftime("%d/%m/%Y")
    else:
        date_label = target_date.strftime("%d/%m/%Y") + "*"
    target = rs_plots.read_raster(targets[j])
    input = rs_plots.read_raster(inputs[j])
    rs_plots.plot_raster(
        ax[0, j],
        target,
        False,
        [2,1,0],
        scalebar=True,
        scalebar_px_len=200,
        scalebar_px_label="15 m",
        scalebar_font_size=None
    )
    if j == 0: ax[0, 0].set_ylabel("Nearmap", fontsize=title_font_size)
    ax[0, j].set_title(date_label)
    for i in range(0, num_models):
        pred = rs_plots.read_raster(preds[i][j])
        rs_plots.plot_raster(ax[i+1, j], pred, nir, order)
        if j == 0: ax[i+1, 0].set_ylabel(f"{MODELS[i]}", fontsize=title_font_size)
    #rs_plots.plot_raster(ax[num_models + 1, j], input, nir, order, bicubic=True)
    #if j == 0: ax[num_models + 1, 0].set_ylabel("Bicubic", fontsize=title_font_size)
    rs_plots.plot_raster(ax[num_models + 1, j], input, nir, order)
    if j == 0: ax[num_models + 1, 0].set_ylabel("PlanetScope", fontsize=title_font_size)
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
if nir:
    fig.savefig(f"{out_dir}{DATASET}_{SUBSET}_nir.pdf", dpi=400)
else:
    fig.savefig(f"{out_dir}{DATASET}_{SUBSET}_rgb.pdf", dpi=100)
plt.close()


