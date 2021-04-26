#
# This is the code for plotting the figures for RQ1. It is optimized towards plotting exactly those figures
# Use data_analysis.py for explorative data analysis
#

from plotting_utils import load_probability_maps, generate_average_probability_maps, set_probability_maps_axes, store_figure_to_paper_folder
import matplotlib.pyplot as plt


def preprare_the_figure(probability_df, features_df , feature_list):
    fontsize = 24
    min_fontsize = 20

    #  Create the figure 3 x 1 + ColorBar ,
    fig, axs = plt.subplots(ncols=4, gridspec_kw=dict(width_ratios=[5, 5, 5, 0.2]), figsize=(30, 10))

    assert len(feature_list) == 3, "Too many combinations to plot"

    for idx, features in enumerate(feature_list):
        axs[idx] = generate_average_probability_maps(axs[idx], probability_df, features)
        set_probability_maps_axes(axs[idx], features_df, features, "rescaled", fontsize=fontsize, min_fontsize=min_fontsize)

    # This returns a cbar object not its axis
    cbar = fig.colorbar(axs[2].collections[0], cax=axs[3])
    cbar.ax.tick_params(labelsize=min_fontsize)
    cbar.ax.set_xlabel('')
    return fig

def main():
    # Load all the data and select the required feature combinations

    probability_df, features_df = load_probability_maps("./data/mnist")
    mnist_fig = preprare_the_figure(probability_df, features_df, [('moves', 'bitmaps'),
                                                                  ('orientation', 'bitmaps'),
                                                                  ('orientation', 'moves')])
    store_figure_to_paper_folder(mnist_fig, file_name="RQ3-MNIST")

    probability_df, features_df = load_probability_maps("./data/beamng")
    beamng_fig = preprare_the_figure(probability_df, features_df,[('mean_lateral_position', 'sd_steering'),
                                                                  ('segment_count', 'min_radius'),
                                                                  ('mean_lateral_position', 'min_radius')])
    store_figure_to_paper_folder(beamng_fig, file_name="RQ3-BeamNG")





if __name__ == "__main__":
    main()
