import os
import matplotlib.pyplot as plt
from datetime import datetime
from scripts.plot import rs_plots

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.sans-serif"] = ["Helvetica"]

def plot_spacer_row(ax, ncols, rownum):
    """Create a blank spacer row in the plot grid."""
    for i in range(ncols):
        ax[rownum, i].axhline(0, color='white', lw=1)
        ax[rownum, i].axis('off')


def get_sorted_files(directory, extension=".tif"):
    """Get sorted list of files with the given extension from directory."""
    return sorted([
        os.path.join(directory, fname)
        for fname in os.listdir(directory)
        if fname.endswith(extension) and not fname.startswith(".")
    ])


def format_date_label(target_path, input_path):
    """Format the date label for a pair of target and input images."""
    target_date = os.path.basename(target_path).split(".")[0][3:]
    input_date = os.path.basename(input_path).split(".")[0][3:]

    target_date = datetime.strptime(target_date, "%Y%m%d")
    input_date = datetime.strptime(input_date, "%Y%m%d")

    if target_date == input_date:
        return target_date.strftime("%d/%m/%Y")
    else:
        return target_date.strftime("%d/%m/%Y") + "*"


def plot_imagery_row(ax, row_start, dataset_index, inputs, targets, all_preds, models, nir=False, order=[2, 1, 0, 3]):
    """Plot a row of imagery data for a specific dataset."""
    num_models = len(models)
    ncols = len(inputs)

    for j in range(ncols):
        date_label = format_date_label(targets[j], inputs[j])

        # Plot target (Nearmap)
        target = rs_plots.read_raster(targets[j])
        rs_plots.plot_raster(
            ax[row_start, j],
            target,
            False,
            [2, 1, 0],
            scalebar=True,
            scalebar_px_len=400,
            scalebar_px_label="30 m",
            scalebar_font_size=plt.rcParams['axes.titlesize']
        )
        if j == 0:
            ax[row_start, 0].set_ylabel("Nearmap", fontsize=plt.rcParams['axes.titlesize'])
        ax[0, j].set_title(date_label)

        # Plot model predictions
        for i in range(num_models):
            pred = rs_plots.read_raster(all_preds[dataset_index][i][j])
            rs_plots.plot_raster(ax[row_start + i + 1, j], pred, nir, order)
            if j == 0:
                ax[row_start + i + 1, 0].set_ylabel(f"{models[i]}", fontsize=plt.rcParams['axes.titlesize'])

        # Plot input (PlanetScope)
        input_img = rs_plots.read_raster(inputs[j])
        rs_plots.plot_raster(ax[row_start + num_models + 1, j], input_img, nir, order)
        if j == 0:
            ax[row_start + num_models + 1, 0].set_ylabel("PlanetScope", fontsize=plt.rcParams['axes.titlesize'])


def main():
    # Configuration
    dataset = "PS_NM_timeseries"
    subset = ["dam_115263", "dam_113992"]
    models = ["ESRGAN", "EDSR"]
    ncols = 6
    nir = False
    order = [2, 1, 0, 3]
    num_models = len(models)

    # Calculate layout dimensions
    rows_per_dataset = num_models + 2  # Target + models + input
    nrows = (rows_per_dataset * 2) + 1  # Two datasets with spacer row

    # Create output directory
    out_dir = f"plots/{dataset}/"
    os.makedirs(out_dir, exist_ok=True)

    # Load all data
    all_inputs = []
    all_targets = []
    all_preds = []

    for i in range(len(subset)):
        inputs_dir = f"datasets/{dataset}/{subset[i]}/inputs/"
        targets_dir = f"datasets/{dataset}/{subset[i]}/targets/"
        preds_dirs = [f"results/{dataset}/{subset[i]}/{model}/" for model in models]

        inputs = get_sorted_files(inputs_dir)[:ncols]
        targets = get_sorted_files(targets_dir)[:ncols]
        preds = [get_sorted_files(pred_dir)[:ncols] for pred_dir in preds_dirs]

        all_inputs.append(inputs)
        all_targets.append(targets)
        all_preds.append(preds)

    # Create plot
    width_ratios = [1] * ncols
    height_ratios = [1, 1, 1, 1, 0.05, 1, 1, 1, 1]

    fig, ax = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 1.5, nrows * 1.34),
        gridspec_kw={'width_ratios': width_ratios, 'height_ratios': height_ratios}
    )

    # Plot first dataset
    plot_imagery_row(ax, 0, 0, all_inputs[0], all_targets[0], all_preds, models, nir, order)

    # Add spacer row
    plot_spacer_row(ax, ncols=ncols, rownum=4)

    # Plot second dataset
    plot_imagery_row(ax, 5, 1, all_inputs[1], all_targets[1], all_preds, models, nir, order)

    # Finalize and save plot
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    output_filename = f"{out_dir}{dataset}_{'nir' if nir else 'rgb'}.pdf"
    dpi = 400 if nir else 100
    fig.savefig(output_filename, dpi=dpi)
    plt.close()


if __name__ == "__main__":
    main()