#
# This is the code for plotting the figures for RQ1. It is optimized towards plotting exactly those figures
# Use data_analysis.py for explorative data analysis
#
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from plotting_utils import load_data_from_folder, create_custom_palette, \
    filter_data_and_plot_as_boxplots, filter_data_by_tag, store_figure_to_paper_folder
import matplotlib.pyplot as plt

def preprare_the_figure(plot_data):

    fontsize = 30
    min_fontsize = 24
    color_palette = create_custom_palette()

    # Create the figure
    fig = plt.figure(figsize=(15, 10))
    # Set the figure to be a grid with 1 column and 2 rows without space between them
    gs = fig.add_gridspec(2, hspace=0)
    # Get the axes objects
    axs = gs.subplots(sharex=True)
    # Plot the top plot
    axs[0] = filter_data_and_plot_as_boxplots(axs[0], "Filled Cells", plot_data, color_palette)
    # Plot the bottom plot
    axs[1] = filter_data_and_plot_as_boxplots(axs[1], "Coverage Sparseness", plot_data, color_palette)

    # Adjust the plots

    # Increase font for y-label and y-ticks
    axs[0].set_ylabel(axs[0].get_ylabel(), fontsize=fontsize)
    axs[0].tick_params(axis='y', which='major', labelsize=min_fontsize)

    # ax.tick_params(axis='both', which='minor', labelsize=8)
    axs[0].legend(fontsize=fontsize)

    # Remove the legend from the bottom plot
    axs[1].legend([], [], frameon=False)
    # Remove the x - label
    axs[1].set_xlabel('')
    # Increase only the size of x-ticks, but split the combinations in two lines
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[1] = 'Testing'
    # https://stackoverflow.com/questions/11244514/modify-tick-label-text
    axs[1].set_xticklabels([l.get_text().replace("-", "\n") for l in axs[1].get_xticklabels()], fontsize=fontsize)
    # Increase label y-label and y-ticks
    axs[1].set_ylabel("Cov. Sparseness", fontsize=fontsize)
    axs[1].tick_params(axis='y', which='major', labelsize=min_fontsize)

    # Align the y labels: -0.1 moves it a bit to the left, 0.5 move it in the middle of y-axis
    axs[0].get_yaxis().set_label_coords(-0.08, 0.5)
    axs[1].get_yaxis().set_label_coords(-0.08, 0.5)

    return fig

def main():
    # Load all the data and select the required feature combinations

    mnist_data = load_data_from_folder("./data/mnist")
    mnist_data = filter_data_by_tag(mnist_data, ["black-box", "rescaled"])
    mnist_figure = preprare_the_figure(mnist_data)

    # Store
    store_figure_to_paper_folder(mnist_figure, file_name="RQ2-MNIST")

    beamng_data = load_data_from_folder("./data/beamng")
    beamng_data = filter_data_by_tag(beamng_data, ["black-box", "rescaled"])
    beamng_figure = preprare_the_figure(beamng_data)

    # Store
    store_figure_to_paper_folder(beamng_figure, file_name="RQ2-BeamNG")

if __name__ == "__main__":
    main()
