#
# This is the code for plotting the figures for RQ1. It is optimized towards plotting exactly those figures
# Use data_analysis.py for explorative data analysis
#
import matplotlib
import numpy as np
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from plotting_utils import  load_auc_mnist,\
    store_figure_to_paper_folder, load_auc_beamng
import matplotlib.pyplot as plt


def preprare_the_plots_mnist(auc_filled, auc_misbehaviour, feature_list):

    fontsize = 24
    min_fontsize = 20
    markers = [".", ".", ".", "."]
    # x axis
    time_intervals = np.linspace(0,3600, 360)


    #  Create the figure 3 x 1 
    fig1, axs1 = plt.subplots(ncols=3, sharex='col', sharey='row', gridspec_kw=dict(width_ratios=[5, 5, 5]), figsize=(30, 10))

    assert len(feature_list) == 3, "Too many combinations to plot"

    for idx, features in enumerate(feature_list):
        i = 0
        for auc in auc_filled:
            if auc["features"] == features:
                auc_data = auc["auc"]
                auc_std = auc["auc_std"]
                axs1[idx].plot(time_intervals, auc_data , marker=markers[i],label=auc["tool"])
                axs1[idx].tick_params(axis='both', which='major', labelsize=min_fontsize)
                axs1[idx].fill_between(time_intervals, auc_data-auc_std, auc_data+auc_std, alpha=0.3)
                i += 1
        title  = features.replace("-", ",").replace("Moves", "Mov").replace("Bitmaps", "Lum").replace("Orientation", "Or")
        axs1[idx].set_title(title, fontsize=fontsize)
        axs1[0].set_ylabel("Filled Cells", fontsize=min_fontsize)
        axs1[idx].set_xlabel("Time (sec)", fontsize=min_fontsize)

    # add legend to the right and remove from axes
    axs1[2].legend(bbox_to_anchor=(1.04,1), borderaxespad=0,fontsize=fontsize)
    axs1[1].legend([], [], frameon=False)
    axs1[0].legend([], [], frameon=False)


    #  Create the figure 3 x 1 + ColorBar ,
    fig2, axs2 = plt.subplots(ncols=3, sharex='col', sharey='row', gridspec_kw=dict(width_ratios=[5, 5, 5]), figsize=(30, 10))

    assert len(feature_list) == 3, "Too many combinations to plot"

    for idx, features in enumerate(feature_list):
        i = 0
        for auc in auc_misbehaviour:
            if auc["features"] == features:
                auc_data = auc["auc"]
                auc_std = auc["auc_std"]
                axs2[idx].plot(time_intervals, auc_data , marker=markers[i],label=auc["tool"])
                axs2[idx].tick_params(axis='both', which='major', labelsize=min_fontsize)
                axs2[idx].fill_between(time_intervals, auc_data-auc_std, auc_data+auc_std, alpha=0.3)
                i += 1
    
        title  = features.replace("-", ",").replace("Moves", "Mov").replace("Bitmaps", "Lum").replace("Orientation", "Or")
        axs2[idx].set_title(title, fontsize=fontsize)
        axs2[0].set_ylabel("Mapped Misb.", fontsize=min_fontsize)
        axs2[idx].set_xlabel("Time (sec)", fontsize=min_fontsize)
        axs2[idx].tick_params(axis='both', which='major', labelsize=min_fontsize)
    
    axs2[2].legend(bbox_to_anchor=(1.04,1), borderaxespad=0,fontsize=fontsize)
    axs2[1].legend([], [], frameon=False)
    axs2[0].legend([], [], frameon=False)

    return fig1, fig2


def preprare_the_plots_beamng(auc_filled, auc_misbehaviour, feature_list):

    fontsize = 24
    min_fontsize = 20
    markers = [".", ".", ".", "."]

    #  Create the figure 3 x 1 
    fig1, axs1 = plt.subplots(ncols=3, sharex='col', sharey='row', gridspec_kw=dict(width_ratios=[5, 5, 5]), figsize=(30, 10))

    assert len(feature_list) == 3, "Too many combinations to plot"

    for idx, features in enumerate(feature_list):
        i = 0
        for auc in auc_filled:
            if auc["features"] == features:
                auc_data = auc["auc"]
                tool = auc["tool"].replace("DeepHyperionBeamNG", "DeepHyperion").replace("DeepJanusBeamNG", "DeepJanus").replace("DeepHyperion-CSBeamNG", "DeepHyperion-CS")
                auc_std = auc["auc_std"]
                time_intervals = auc["time"]
                axs1[idx].plot(time_intervals, auc_data , marker=markers[i],label=tool)
                axs1[idx].tick_params(axis='both', which='major', labelsize=min_fontsize)
                axs1[idx].fill_between(time_intervals, auc_data-auc_std, auc_data+auc_std, alpha=0.3)
                i += 1
        title  = features.replace("-", ",").replace("MeanLateralPosition", "MLP").replace("SDSteeringAngle", "StdSA").replace("Curvature", "Curv").replace("SegmentCount", "TurnCnt")
        axs1[idx].set_title(title, fontsize=fontsize)
        axs1[0].set_ylabel("Filled Cells", fontsize=min_fontsize)
        axs1[idx].set_xlabel("Time (sec)", fontsize=min_fontsize)

    # add legend to the right and remove from axes
    axs1[2].legend(bbox_to_anchor=(1.04,1), borderaxespad=0,fontsize=fontsize)
    axs1[1].legend([], [], frameon=False)
    axs1[0].legend([], [], frameon=False)


    #  Create the figure 3 x 1 + ColorBar ,
    fig2, axs2 = plt.subplots(ncols=3, sharex='col', sharey='row', gridspec_kw=dict(width_ratios=[5, 5, 5]), figsize=(30, 10))

    assert len(feature_list) == 3, "Too many combinations to plot"

    for idx, features in enumerate(feature_list):
        i = 0
        for auc in auc_misbehaviour:
            if auc["features"] == features: 
                auc_data = auc["auc"]
                tool = auc["tool"].replace("DeepHyperionBeamNG", "DeepHyperion").replace("DeepJanusBeamNG", "DeepJanus").replace("DeepHyperion-CSBeamNG", "DeepHyperion-CS")
                auc_std = auc["auc_std"]
                time_intervals = auc["time"]
                axs2[idx].plot(time_intervals, auc_data , marker=markers[i],label=tool)
                axs2[idx].tick_params(axis='both', which='major', labelsize=min_fontsize)
                axs2[idx].fill_between(time_intervals, auc_data-auc_std, auc_data+auc_std, alpha=0.3)
                i += 1
    
        title  = features.replace("-", ",").replace("MeanLateralPosition", "MLP").replace("SDSteeringAngle", "StdSA").replace("Curvature", "Curv").replace("SegmentCount", "TurnCnt")
        axs2[idx].set_title(title, fontsize=fontsize)
        axs2[0].set_ylabel("Mapped Misb.", fontsize=min_fontsize)
        axs2[idx].set_xlabel("Time (sec)", fontsize=min_fontsize)
        axs2[idx].tick_params(axis='both', which='major', labelsize=min_fontsize)
    
    axs2[2].legend(bbox_to_anchor=(1.58,1), borderaxespad=0,fontsize=fontsize)
    axs2[1].legend([], [], frameon=False)
    axs2[0].legend([], [], frameon=False)

    return fig1, fig2


def main():
    # Load all the data and select the required feature combinations
    print("Preparing MNIST plot")
    auc_filled, auc_misbehaviour = load_auc_mnist("./data/mnist")
    mnist_fig_filled, mnist_fig_misbehavior = preprare_the_plots_mnist(auc_filled, auc_misbehaviour, ["Moves-Bitmaps",
                                                                   "Orientation-Bitmaps",
                                                                   "Orientation-Moves"])
    print("Storing MNIST plot")
    store_figure_to_paper_folder(mnist_fig_filled, file_name="RQ3-MNIST-2")
    store_figure_to_paper_folder(mnist_fig_misbehavior, file_name="RQ3-MNIST-1")

    print("Preparing BeamNG plot")
    auc_filled, auc_misbehaviour = load_auc_beamng("./data/beamng")
    beamng_fig_filled, beamng_fig_misbehavior = preprare_the_plots_beamng(auc_filled, auc_misbehaviour, [ "MeanLateralPosition-SDSteeringAngle",
    "MeanLateralPosition-SegmentCount", 
    "SDSteeringAngle-Curvature"])

    print("Storing BeamNG plot")
    store_figure_to_paper_folder(beamng_fig_filled, file_name="RQ3-BeamNG-2")
    store_figure_to_paper_folder(beamng_fig_misbehavior, file_name="RQ3-BeamNG-1")

if __name__ == "__main__":
    main()
