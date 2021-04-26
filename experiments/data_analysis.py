# Entry file for data analysis and reporting

import os
import errno
import sys

from matplotlib.patches import Rectangle


import statsmodels.stats.proportion as smp

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import itertools as it
import logging as log

import click
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pandas as pd
import numpy as np

from numpy import mean
from numpy import var
from math import sqrt

from scipy.stats import pearsonr
import scipy.stats as ss
from bisect import bisect_left
from pandas import Categorical

PAPER_FOLDER="./plots"


# calculate Pearson's correlation
def correlation(d1, d2):
    corr, _ = pearsonr(d1, d2)
    print('Pearsons correlation: %.3f' % corr)
    return corr

# https://gist.github.com/jacksonpradolima/f9b19d65b7f16603c837024d5f8c8a65
# https://machinelearningmastery.com/effect-size-measures-in-python/
# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    return (u1 - u2) / s

# https://gist.github.com/jacksonpradolima/f9b19d65b7f16603c837024d5f8c8a65
def VD_A(treatment, control):
    """
    Computes Vargha and Delaney A index
    A. Vargha and H. D. Delaney.
    A critique and improvement of the CL common language
    effect size statistics of McGraw and Wong.
    Journal of Educational and Behavioral Statistics, 25(2):101-132, 2000
    The formula to compute A has been transformed to minimize accuracy errors
    See: http://mtorchiano.wordpress.com/2014/05/19/effect-size-of-r-precision/
    :param treatment: a numeric list
    :param control: another numeric list
    :returns the value estimate and the magnitude
    """
    m = len(treatment)
    n = len(control)

    if m != n:
        raise ValueError("Data must have the same length")

    r = ss.rankdata(treatment + control)
    r1 = sum(r[0:m])

    # Compute the measure
    # A = (r1/m - (m+1)/2)/n # formula (14) in Vargha and Delaney, 2000
    A = (2 * r1 - m * (m + 1)) / (2 * n * m)  # equivalent formula to avoid accuracy errors

    levels = [0.147, 0.33, 0.474]  # effect sizes from Hess and Kromrey, 2004
    magnitude = ["negligible", "small", "medium", "large"]
    scaled_A = (A - 0.5) * 2

    magnitude = magnitude[bisect_left(levels, abs(scaled_A))]
    estimate = A

    return estimate, magnitude


def VD_A_DF(data, val_col: str = None, group_col: str = None, sort=True):
    """
    :param data: pandas DataFrame object
        An array, any object exposing the array interface or a pandas DataFrame.
        Array must be two-dimensional. Second dimension may vary,
        i.e. groups may have different lengths.
    :param val_col: str, optional
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains values.
    :param group_col: str, optional
        Must be specified if `a` is a pandas DataFrame object.
        Name of the column that contains group names.
    :param sort : bool, optional
        Specifies whether to sort DataFrame by group_col or not. Recommended
        unless you sort your data manually.
    :return: stats : pandas DataFrame of effect sizes
    Stats summary ::
    'A' : Name of first measurement
    'B' : Name of second measurement
    'estimate' : effect sizes
    'magnitude' : magnitude
    """

    x = data.copy()
    if sort:
        x[group_col] = Categorical(x[group_col], categories=x[group_col].unique(), ordered=True)
        x.sort_values(by=[group_col, val_col], ascending=True, inplace=True)

    groups = x[group_col].unique()

    # Pairwise combinations
    g1, g2 = np.array(list(it.combinations(np.arange(groups.size), 2))).T

    # Compute effect size for each combination
    ef = np.array([VD_A(list(x[val_col][x[group_col] == groups[i]].values),
                        list(x[val_col][x[group_col] == groups[j]].values)) for i, j in zip(g1, g2)])

    return pd.DataFrame({
        'A': np.unique(data[group_col])[g1],
        'B': np.unique(data[group_col])[g2],
        'estimate': ef[:, 0],
        'magnitude': ef[:, 1]
    })


def _log_raw_statistics(treatment, treatment_name, control, control_name):
    # Compute p : In statistics, the Mann–Whitney U test (also called the Mann–Whitney–Wilcoxon (MWW),
    # Wilcoxon rank-sum test, or Wilcoxon–Mann–Whitney test) is a nonparametric test of the null hypothesis that,
    # for randomly selected values X and Y from two populations, the probability of X being greater than Y is
    # equal to the probability of Y being greater than X.

    statistics, p_value = ss.mannwhitneyu(treatment, control)
    # Compute A12
    estimate, magnitude = VD_A(treatment, control)

    # Print them
    print("Comparing: %s,%s.\n \t p-Value %s - %s \n \t A12 %f - %s" %(
             treatment_name.replace("\n", " "), control_name.replace("\n", " "),
             statistics, p_value,
             estimate, magnitude))


def _log_statistics(data, column_name):

    print("Log Statistics for: %s" % (column_name))
    # Generate all the pairwise combinations
    for treatment_name, control_name in it.combinations(data["Tool"].unique(), 2):
        try:
            treatment = list(data[data["Tool"] == treatment_name][column_name])
            control = list(data[data["Tool"] == control_name][column_name])

            # Compute the statistics
            _log_raw_statistics(treatment, treatment_name, control, control_name)
        except:
            print("*    Cannot compare %s (%d) and %s (%d)" % (treatment_name, len(treatment), control_name, len(control)))


def _log_exception(extype, value, trace):
    log.exception('Uncaught exception:', exc_info=(extype, value, trace))


def _set_up_logging(debug):
    # Disable annoyng messages from matplot lib.
    # See: https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python
    log.getLogger('matplotlib.font_manager').disabled = True

    term_handler = log.StreamHandler()
    log_handlers = [term_handler]
    start_msg = "Process Started"

    log_level = log.DEBUG if debug else log.INFO

    log.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=log_level, handlers=log_handlers)

    # Configure default logging for uncaught exceptions
    sys.excepthook = _log_exception

    log.info(start_msg)


def _adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def _store_figure_to_paper_folder(figure, file_name):
    import os
    try:
        os.makedirs(PAPER_FOLDER)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    file_format = 'pdf'
    figure_file_name = "".join([file_name, ".", file_format])
    figure_file = os.path.join(PAPER_FOLDER, figure_file_name)

    # https://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
    figure.tight_layout()
    figure.savefig(figure_file, format=file_format, bbox_inches='tight')


def _create_custom_palette(data):
    # Todo Ensures DeepHyperion is always bright
    # tools = ["", "", "DLFuzz", "", "DeepJanus", "", "DeepHyperion"]
    # tool_colors = dict(zip(tools, sns.color_palette("gray", len(tools))))
    
    tool_colors = {
        "DeepHyperion": "#ffffff",
        "DeepJanus" : "#d3d3d3", #C0C0C0 - #DCDCDC
        "DLFuzz": "#a9a9a9" # #808080
    }
    # tool_colors = dict(zip(tools, sns.color_palette("cubehelix", len(tools))))
    # https://colorbrewer2.org/#type=sequential&scheme=OrRd&n=3
    # tool_colors = {
    #     "DeepHyperion" : "#fee8c8",
    #     "DeepJanus" : "#fdbb84",
    #     "DLFuzz" : "#e34a33",
    # }
    return tool_colors


def rename_features(features):
    return [rename_feature(f) for f in features]


def rename_feature(feature):
    if "Bitmaps" == feature or "bitmaps" == feature:
        return "Lum"
    elif "Moves" == feature or "moves" == feature:
        return "Mov"
    elif "Orientation" == feature or "orientation" == feature:
        return "Or"
    ##
    elif "Segment Count" == feature or "segment_count" == feature:
        return "TurnCnt"
    elif "MinRadius" == feature or "min_radius" == feature:
        return "MinRad"
    elif "MeanLateralPosition" == feature or "mean_lateral_position" == feature:
        return "MLP"
    elif "SDSteeringAngle" == feature or "sd_steering" == feature:
        return "StdSA"

def load_data_from_folder(dataset_folder):
    """
    Returns: Panda DF with the data about the experiments from the data folder, data/mnist or data/beamng. Merge the configurations of DH together
    -------
    """

    the_data = None

    for subdir, dirs, files in os.walk(dataset_folder, followlinks=False):

        # Consider only the files that match the pattern
        for json_data_file in [os.path.join(subdir, f) for f in files if f.endswith("stats.json")]:

            with open(json_data_file, 'r') as input_file:
                # Get the JSON
                map_dict = json.load(input_file)

                # Introduce a value to uniquely identify the tool combinations (Important fo DH)
                map_dict["Tool"] = map_dict["Tool"].replace("BeamNG", "")

                # "Expose" the reports by flattening the JSON
                # TODO We rely on the fact that there's ONLY one report here
                assert len(map_dict["Reports"]) == 1, "Too many reports to plot !"
                skip = False
                for report_idx, report_dict in enumerate(map_dict["Reports"]):

                    target_feature_combination = "-".join(rename_features(report_dict["Features"]))
                    allowed_features_combination = None
                    if allowed_features_combination is not None and not target_feature_combination in allowed_features_combination:
                        print("SKIP Feature Combination ", target_feature_combination)
                        skip = True
                        continue
                    else:
                        map_dict["Features Combination"] = target_feature_combination
                        for item in [i for i in report_dict.items() if i[0] != "Features"]:
                            map_dict[item[0]] = item[1]
                    # Add a label to uniquely identify the reports by their features combinations
                    # report_dict["Features Combination"] = "-".join(report_dict["Features"])
                    # Add the The exploration / exploitation trade - off
                    # report_dict["Exploration vs Exploitation"] = report_dict["Filled Cells"]/map_dict["Total Samples"]
                    # Add it to the main data structure
                    # map_dict[str("Report") + str(report_idx)] = report_dict
                    # "Features": [
                    #     "orientation",
                    #     "moves"
                    # ],
                    map_dict["Features Combination"] = "-".join(rename_features(report_dict["Features"]))
                    for item in [i for i in report_dict.items() if i[0] != "Features"]:
                        map_dict[item[0]] = item[1]


                # Patch: Do not any data if there's no feature combination
                if skip:
                    continue

                if the_data is None:
                    # Creates the DataFrame
                    the_data = pd.json_normalize(map_dict)
                else:
                    # Maybe better to concatenate only once
                    the_data = pd.concat([the_data, pd.json_normalize(map_dict)])

    # make sure that DH is reported per each configuration
    # https://stackoverflow.com/questions/26886653/pandas-create-new-column-based-on-values-from-other-columns-apply-a-function-o

    # # TODO Improve the labeling - Let's do this directly at the level of JSON
    # fn = lambda row: row.Tool + '-'.join(row.Tags) if row.Tool == "DeepHyperionBeamNG" else row.Tool  # define a function for the new column
    # col = beamng_data.apply(fn, axis=1)  # get column data with an index
    # beamng_data = beamng_data.assign(**{'Tool Configuration': col.values})

    # Fix data type
    the_data['Tags'] = the_data['Tags'].astype(str)
    print("Features Combinations:", the_data["Features Combination"].unique())
    return the_data


def filter_data_by_tag(ctx, tags):
    # Keep only the data which contain this tags. Ideally one should simply check for containment in the Tags column,
    # but this somehow gets the d64 type instead of string...
    # Load data and store that into the context for the next commands
    # This is how we filter white-box and black-box data
    mnist_data = ctx.obj['mnist-data-full']

    assert len(tags) > 0, "Specify a tag to filter by"

    if mnist_data is not None:
        # result =  all(elem in list1  for elem in list2)
        for tag in tags:
            mnist_data = mnist_data[mnist_data['Tags'].str.contains(tag)]

        ctx.obj['mnist-data'] = mnist_data


    beamng_data = ctx.obj['beamng-data-full']
    if beamng_data is not None:
        for tag in tags:
            print("Debug: Size before filtering: %d " % len(beamng_data.index))
            beamng_data = beamng_data[beamng_data['Tags'].str.contains(tag)]
            print("Debug: Size before filtering: %d " % len(beamng_data.index))
        ctx.obj['beamng-data'] = beamng_data


@click.group()
@click.option('--debug', required=False, is_flag=True, default=False, help="Activate debugging (more logging)")
@click.option('--visualize', required=False, is_flag=True, default=False, help="Visualize the generated plots")
@click.pass_context
def cli(ctx, debug, visualize):
    """
    Main entry point for the CLI. This is mostly to setup general configurations such as the logging
    """
    # See: https://click.palletsprojects.com/en/7.x/commands/
    # Nested Commands
    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)
    ctx.obj['debug'] = debug
    ctx.obj['visualize'] = visualize

    _set_up_logging(debug)

    # Load data and store that into the context for the next commands
    mnist_data = load_data_from_folder("./data/mnist")
    ctx.obj['mnist-data-full'] = mnist_data

    beamng_data = load_data_from_folder("./data/beamng")
    ctx.obj['beamng-data-full'] = beamng_data

    if mnist_data is not None:
        mnist_color_palette = _create_custom_palette(mnist_data)
        ctx.obj['mnist-palette'] = mnist_color_palette

    if beamng_data is not None:
        beamng_color_palette = _create_custom_palette(beamng_data)
        ctx.obj['beamng-palette'] = beamng_color_palette


@cli.resultcallback()
@click.pass_context
def process_result(ctx, result, **kwargs):
    if ctx.obj["visualize"]:
        plt.show()


# Utility to plot maps data
def _filter_data_and_plot_as_boxplots(rq_id, data_set_id, we_plot, raw_data, palette, store_to):

    assert type(we_plot) is str, "we_plot not a string !"

    # Select only the data we need to plot
    plot_axis_and_grouping = [
        "Tool",  # Test Subjects
        "Features Combination"  # Features that define this map
    ]
    # Filter the data
    we_need = plot_axis_and_grouping[:]
    we_need.append(we_plot)
    plot_data = raw_data[we_need]

    if plot_data.empty:
        print("WARINING: Empty plot for %s %s %s" % (rq_id, data_set_id, we_plot))
        return

    # Prepare the figure. TODO we_plot must be a single dimension here !

    hue_order = []
    for tool_name in ["DeepHyperion", "DeepJanus", "DLFuzz"]:
        if tool_name in plot_data["Tool"].unique():
            hue_order.append(tool_name)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.boxplot(x="Features Combination",
                     y=we_plot,
                     hue="Tool",
                     data=plot_data,
                     palette=palette,
                     hue_order=hue_order)
    #                 order=)
    # https://python-graph-gallery.com/35-control-order-of-boxplot/
    # Only for Debug
    # ax = sns.stripplot(x="Features Combination",
    #                    y=we_plot,
    #                    hue="Tool",
    #                    data=plot_data,
    #                    color = 'black',
    #                    jitter=0.25,
    #                    dodge=True)

    # TODO Replace the feature combination with the human readable names
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = [label.replace("-", " - ") for label in labels]
    ax.set_xticklabels(labels)

    # Remove y Label
    # ax.set(ylabel=None)
    ax.tick_params(labelsize=12)
    # the_title = " ".join([rq_id, "-", data_set_id])
    # fig.suptitle(the_title, fontsize=16)

    # Store the plot
    _store_figure_to_paper_folder(fig, store_to)

    # TODO We can probably create a latex table with this data...
    # TODO This does not work if the series do not have the same size
    # Log the statistics
    for the_map in plot_data["Features Combination"].unique():
        # Filter the maps first by "Features Combination" and the invoke the regular _log_statistics !
        print("============================================================================")
        print("DATASET %s Showing comparisons for MAP %s : " %(rq_id, the_map))
        print("============================================================================")

        stats_data = plot_data[plot_data["Features Combination"] == the_map]

        _log_statistics(stats_data, we_plot)
#


def plot_mapped_misbehaviour_distribution(rq_id, ctx):
    we_plot = "Mapped Misbehaviors"
    store_to = "-".join([str(rq_id), "mapped_misbheavior"])

    beamng_raw_data = ctx.obj["beamng-data"]
    if beamng_raw_data is not None:
        palette = ctx.obj["beamng-palette"]
        _filter_data_and_plot_as_boxplots(rq_id, "BeamNG", we_plot, beamng_raw_data, palette, store_to+"-BeamNG")


    mnist_raw_data = ctx.obj["mnist-data"]
    if mnist_raw_data is not None:
        palette = ctx.obj["mnist-palette"]
        _filter_data_and_plot_as_boxplots(rq_id, "MNIST", we_plot, mnist_raw_data, palette, store_to+"-MNIST")


def plot_misbehaviour_sparseness_distribution(rq_id, ctx):
    we_plot = "Misbehavior Sparseness"
    store_to = "-".join([str(rq_id), "misbheavior-sparseness"])

    beamng_raw_data = ctx.obj["beamng-data"]
    if beamng_raw_data is not None:
        palette = ctx.obj["beamng-palette"]
        _filter_data_and_plot_as_boxplots(rq_id, "BeamNG", we_plot, beamng_raw_data, palette, store_to+"-BeamNG")


    mnist_raw_data = ctx.obj["mnist-data"]
    if mnist_raw_data is not None:
        palette = ctx.obj["mnist-palette"]
        _filter_data_and_plot_as_boxplots(rq_id, "MNIST", we_plot, mnist_raw_data, palette, store_to+"-MNIST")


def plot_misbehaviour_collision_ratio_distribution(rq_id, ctx):
    we_plot = "Misbehavior Collision Ratio"

    store_to = "-".join([str(rq_id), "misbheavior-collision-ratio"])

    beamng_raw_data = ctx.obj["beamng-data"]
    if beamng_raw_data is not None:
        palette = ctx.obj["beamng-palette"]
        _filter_data_and_plot_as_boxplots(rq_id, "BeamNG", we_plot, beamng_raw_data, palette, store_to + "-BeamNG")

    mnist_raw_data = ctx.obj["mnist-data"]
    if mnist_raw_data is not None:
        palette = ctx.obj["mnist-palette"]
        _filter_data_and_plot_as_boxplots(rq_id, "MNIST", we_plot, mnist_raw_data, palette, store_to + "-MNIST")


def plot_filled_cell_distribution(rq_id, ctx):
    we_plot = "Filled Cells"

    store_to = "-".join([str(rq_id), "map-coverage"])

    beamng_raw_data = ctx.obj["beamng-data"]
    if beamng_raw_data is not None:
        palette = ctx.obj["beamng-palette"]
        _filter_data_and_plot_as_boxplots(rq_id, "BeamNG", we_plot, beamng_raw_data, palette, store_to + "-BeamNG")

    mnist_raw_data = ctx.obj["mnist-data"]
    if mnist_raw_data is not None:
        palette = ctx.obj["mnist-palette"]
        _filter_data_and_plot_as_boxplots(rq_id, "MNIST", we_plot, mnist_raw_data, palette, store_to + "-MNIST")

def plot_filled_cell_sparseness_distribution(rq_id, ctx):
    we_plot = "Coverage Sparseness"

    store_to = "-".join([str(rq_id), "coverage-sparseness"])

    beamng_raw_data = ctx.obj["beamng-data"]
    if beamng_raw_data is not None:
        palette = ctx.obj["beamng-palette"]
        _filter_data_and_plot_as_boxplots(rq_id, "BeamNG", we_plot, beamng_raw_data, palette, store_to + "-BeamNG")

    mnist_raw_data = ctx.obj["mnist-data"]
    if mnist_raw_data is not None:
        palette = ctx.obj["mnist-palette"]
        _filter_data_and_plot_as_boxplots(rq_id, "MNIST", we_plot, mnist_raw_data, palette, store_to + "-MNIST")


def plot_filled_cell_collision_ratio_distribution(rq_id, ctx):
    we_plot = "Collision Ratio"

    store_to = "-".join([str(rq_id), "collision-ratio"])

    beamng_raw_data = ctx.obj["beamng-data"]
    if beamng_raw_data is not None:
        palette = ctx.obj["beamng-palette"]
        _filter_data_and_plot_as_boxplots(rq_id, "BeamNG", we_plot, beamng_raw_data, palette, store_to + "-BeamNG")

    mnist_raw_data = ctx.obj["mnist-data"]
    if mnist_raw_data is not None:
        palette = ctx.obj["mnist-palette"]
        _filter_data_and_plot_as_boxplots(rq_id, "MNIST", we_plot, mnist_raw_data, palette, store_to + "-MNIST")


############################## Research Questions ##############################
@cli.command()
@click.pass_context
def rq1(ctx):
    """
    RQ1: Failure Diversity
        Context: Generating tests that reveal faults is useful only if the faults revealed by the tests are different.
        In other words, a test generator that repeatedly exposes the same problem is not optimal, as it wastes
        computational resources.
        Question: Can DH generate tests (inputs) that expose ``behaviourally'' diverse failures? How much and in which
        regards the exposed failures differ?
        Metrics:
            For each map/feature combination
                Total cells in the map that contains Misbehaviors
                Misbehaviour sparseness: mean of max manhattan distances
                    For each sample take the one at the max distance, then mean of the max distances
    """
    id = "RQ1"
    # ONLY RESCALED AND BLACK BOX
    for origin, map_transform in it.product(["black-box"], ["rescaled"]):
        # Filter the plot data
        filter_data_by_tag(ctx, [origin, map_transform])
        prefix = "-".join([id, origin, map_transform])
        # Plot and store
        plot_mapped_misbehaviour_distribution(prefix, ctx)
        plot_misbehaviour_sparseness_distribution(prefix, ctx)
        plot_misbehaviour_collision_ratio_distribution(prefix, ctx)


@cli.command()
@click.pass_context
def rq2(ctx):
    """
    RQ2: Search Exploration
        Context: While generating tests, automatic test generation should stress many behaviors of the systems under
        test. This can be achieved by suitably exploring the test/input space and the output space.
        Question: Can DH cover a substantial area of the feature space, i.e., feature map?
        Metrics:
            For each map/feature combination
                Map coverage (Filled Cells/Map size)
                Sparseness
                Collisions: Total # generated samples/ Filled Cells
    """

    id = "RQ2"
    # ONLY RESCALED AND BLACK BOX
    for origin, map_transform in it.product(["black-box"], ["rescaled"]):
        # Filter the plot data
        filter_data_by_tag(ctx, [origin, map_transform])
        prefix = "-".join([id, origin, map_transform])
        # Plot and store
        plot_filled_cell_distribution(prefix, ctx)
        plot_filled_cell_sparseness_distribution(prefix, ctx)
        plot_filled_cell_collision_ratio_distribution(prefix, ctx)


# TODO For the moment look only at white box data
def _load_probability_maps(dataset_folder):

    rows_list = []
    for subdir, dirs, files in os.walk(dataset_folder, followlinks=False):

        # Extract metadata about features
        for json_data_file in [os.path.join(subdir, f) for f in files if
                         f.startswith("DeepHyperion") and
                         (f.endswith("-white-box-rescaled-stats.json") or f.endswith("-white-box-relative-stats.json"))]:

            with open(json_data_file, 'r') as input_file:
                # Get the JSON
                map_dict = json.load(input_file)

            # TODO Read those from the json maybe?
            # DLFuzz-017-Orientation-Moves-white-box-rescaled-stats.json
            attrs = json_data_file.split("-")

            run = attrs[1]
            map_type = attrs[6].replace("-stats.npy", "")

            # Store the features data for this run - Is this a tuple ?!
            features = tuple(map_dict["Features"].keys())

            for feature_name, f in map_dict["Features"].items():
                rows_list.append({
                    'bins': [np.linspace(f["meta"]["min-value"], f["meta"]["max-value"], f["meta"]["num-cells"])],
                    'feature': feature_name,
                    'features': features,
                    'map type': map_type,
                    'run': int(run)
                })

    # Feature Map
    features_data = pd.DataFrame(rows_list, columns={'bins': pd.Series([], dtype='float'),
                                                     'features': pd.Series([], dtype='str'),
                                                     'feature': str(),
                                                     'map type': str(),
                                                     'run': int()})

    rows_list = []
    for subdir, dirs, files in os.walk(dataset_folder, followlinks=False):
        # Consider only the files that match the pattern
        for npy_file in [os.path.join(subdir, f) for f in files if
                         f.startswith("probability-DeepHyperion") and
                         (f.endswith("-white-box-rescaled.npy") or f.endswith("-white-box-relative.npy"))]:

            probabilities = np.load(npy_file)
            attrs = npy_file.split("-")
            # probability-DeepJanusBeamNG-001-segment_count-sd_steering-SegmentCount-SDSteeringAngle-white-box-rescaled.npy
            features = (attrs[3], attrs[4])
            map_type = attrs[9].replace(".npy", "")
            run = attrs[2]

            rows_list.append({
                'probabilities': probabilities,
                'features': features,
                'map type': map_type,
                'run': int(run)
            })

    probability_data = pd.DataFrame(rows_list, columns={'probabilities': pd.Series([], dtype='float'),
                                          'features': pd.Series([], dtype='str'),
                                          'map type': str(),
                                          'run': int()})

    rows_list = []
    for subdir, dirs, files in os.walk(dataset_folder, followlinks=False):
        # Consider only the files that match the pattern
        for npy_file in [os.path.join(subdir, f) for f in files if
                                              f.startswith("misbehaviour-DeepHyperion") and
                                              (f.endswith("-white-box-rescaled.npy") or f.endswith(
                                                  "-white-box-relative.npy"))]:
            misbehaviors = np.load(npy_file)
            attrs = npy_file.split("-")

            features = (attrs[3], attrs[4])
            map_type = attrs[9].replace(".npy", "")
            run = attrs[2]


            rows_list.append({
                'misbehaviors': misbehaviors,
                'features': features,
                'map type': map_type,
                'run': int(run)
            })

    misbehavior_data = pd.DataFrame(rows_list, columns={'misbehaviors': pd.Series([], dtype='float'),
                                                            'features': pd.Series([], dtype='str'),
                                                            'map type': str(),
                                                            'run': int()})

    rows_list = []
    for subdir, dirs, files in os.walk(dataset_folder, followlinks=False):
        # Consider only the files that match the pattern
        for npy_file in [os.path.join(subdir, f) for f in files if
                      f.startswith("coverage-DeepHyperion") and
                      (f.endswith("-white-box-rescaled.npy") or f.endswith(
                          "-white-box-relative.npy"))]:
            coverage = np.load(npy_file)
            attrs = npy_file.split("-")

            features = (attrs[3], attrs[4])
            map_type = attrs[9].replace(".npy", "")
            run = attrs[2]

            rows_list.append({
                'coverage': coverage,
                'features': features,
                'map type': map_type,
                'run': int(run)
            })

    # merge all the DF to obtain the last one
    coverage_data = pd.DataFrame(rows_list, columns={'coverage': pd.Series([], dtype='float'),
                                                        'features': pd.Series([], dtype='str'),
                                                        'map type': str(),
                                                        'run': int()})
    df = probability_data.merge(misbehavior_data, on=['features', 'map type', 'run'])
    df = df.merge(coverage_data, on=['features', 'map type', 'run'])

    return df, features_data

def _set_probability_maps_axes(ax, features_df, features, map_type):
    try:
        # Prepare the labels and ticks (reused across main map and supporting maps)
        f1_bins = list(features_df[(features_df["features"] == features) & (features_df["map type"] == map_type)
                                   & (features_df["feature"] == features[0])]["bins"].array[0][0])
        f2_bins = list(features_df[(features_df["features"] == features) & (features_df["map type"] == map_type)
                                   & (features_df["feature"] == features[1])]["bins"].array[0][0])

        # Stop at first digit after comma
        xtickslabel = [round(the_bin, 1) for the_bin in f1_bins]
        ytickslabel = [round(the_bin, 1) for the_bin in f2_bins]
        ax.set_xticklabels(xtickslabel, fontsize=10)
        plt.xticks(rotation=45)
        ax.set_yticklabels(ytickslabel, fontsize=10)
        plt.yticks(rotation=0)

        # We need this to have the y axis start from zero at the bottom
        ax.invert_yaxis()

        # axis labels
        plt.xlabel(rename_feature(features[0]), fontsize=14)
        plt.ylabel(rename_feature(features[1]), fontsize=14)
    except:
        print("Error in setting axes for", features, map_type)


def generate_average_probability_maps(dataset_id, dataset_folder, map_type="rescaled",
                                      min_avg_prob=0.7999, min_low_ci = 0.64999):
    """
    Generate the map of average misb probability and annotate/highlight the cells which have low-level confidence
     interval above the parameter. Alonside this map we plot the following "supporting" maps:
        - low-conf interval -> useful for MISB cells
        - high-conf interval -> useful for NON-MISB cells
        - total number of samples
        - total number of misb

    Parameters
    ----------
    dataset_id
    map_type
    dataset_folder

    Returns
    -------

    """
    probability_df, features_df = _load_probability_maps(dataset_folder)

    for features in probability_df["features"].unique():

        # Take all the probability maps for rescaled
        all_probabilities = list(probability_df[(probability_df["features"] == features) &
                                                (probability_df["map type"] == map_type)]["probabilities"])
        # Compute the mean ignoring Nan over the cells
        avg_probabilities = np.nanmean(all_probabilities, axis=0)


        # Take misbhevaing
        all_misbehaviors = list(probability_df[(probability_df["features"] == features) & (
                probability_df["map type"] == map_type)]["misbehaviors"])
        all_coverage = list(probability_df[(probability_df["features"] == features) & (
                    probability_df["map type"] == map_type)]["coverage"])
        # Sum per each cell
        total_misb = np.nansum(all_misbehaviors, axis=0)
        total_inputs = np.nansum(all_coverage, axis=0)

        confident_data_high = np.empty(shape=total_misb.shape, dtype=float)
        confident_data_high[:] = np.NaN
        confident_data_low = np.empty(shape=total_misb.shape, dtype=float)
        confident_data_low[:] = np.NaN

        for (i, j), value in np.ndenumerate(total_misb):

            if np.isnan(value):
                continue

            (low, high) = smp.proportion_confint(value, total_inputs[i][j], method='wilson')
            confident_data_high[i][j] = high
            confident_data_low[i][j] = low

            # Transpose to have first axis over x
        avg_probabilities = np.transpose(avg_probabilities)
        confident_data_high = np.transpose(confident_data_high)
        confident_data_low = np.transpose(confident_data_low)
        total_inputs = np.transpose(total_inputs)

        # Create the main figure
        fig, ax = plt.subplots(figsize=(8, 8))
        # Create the color map
        cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0.5, as_cmap=True)
        # Set WHITE for the cells for which we do not have observations (NaN)
        cmap.set_bad(color='white')
        # Show the average probabilities in the map
        ax = sns.heatmap(avg_probabilities, square=True,
                         vmin=0.0, vmax=1.0, cmap=cmap,
                         cbar_kws={"shrink": .6},  # reduce size of the color bar
                         # annot_kws={"size": 5},
                         # linewidths=2,
                         # annot=total_inputs, # Do not annotate the map
                         # fmt='d'
        )

        # Plot the highlighted cells: each cell is an empty square with a tick border.
        # Highlight the cells. Note that we have j, i not i, j because we transposed the original data while here
        # we use the ax reference system and not the heatmap one
        for (j, i), value in np.ndenumerate(confident_data_low):
            if value > min_low_ci:
                ax.add_patch(Rectangle((i, j), 1, 1, fill=False, edgecolor='black', lw=2))

        _set_probability_maps_axes(ax, features_df, features, map_type)

        store_to = "-".join(["RQ3", "misbehaviour-probability", "DeepHyperion", features[0], features[1], dataset_id, map_type])
        _store_figure_to_paper_folder(fig, store_to)

        # now plot the supporting maps for high. use different colors to avoid confusions !




        # Create the confident_data_low figure
        fig, ax = plt.subplots(figsize=(8, 8))
        # Create the color map - Greenish
        cmap = sns.color_palette("crest", as_cmap=True)
        # Set WHITE for the cells for which we do not have observations (NaN)
        cmap.set_bad(color='white')
        # Show the average probabilities in the map
        ax = sns.heatmap(confident_data_low, square=True,
                         vmin=0.0, vmax=1.0, cmap=cmap,
                         cbar_kws={"shrink": .6},  # reduce size of the color bar
                         annot_kws={"size": 5},
                         linewidths=1,
                         annot=True,
                         fmt='.2f'
                         )

        _set_probability_maps_axes(ax, features_df, features, map_type)

        the_title = " ".join(["confidence low", "-", dataset_id])
        fig.suptitle(the_title, fontsize=16)

        store_to = "-".join(
            ["RQ3", "confidence-low", "DeepHyperion", features[0], features[1], dataset_id, map_type])
        _store_figure_to_paper_folder(fig, store_to)




        # Create the confidence_data_high figure
        fig, ax = plt.subplots(figsize=(8, 8))
        # Create the color map - ??? COLOR?
        cmap = sns.color_palette("rocket", as_cmap=True)
        # Set WHITE for the cells for which we do not have observations (NaN)
        cmap.set_bad(color='white')
        # Show the average probabilities in the map
        ax = sns.heatmap(confident_data_high, square=True,
                         vmin=0.0, vmax=1.0, cmap=cmap,
                         cbar_kws={"shrink": .6},  # reduce size of the color bar
                         annot_kws={"size": 5},
                         linewidths=1,
                         annot=True,
                         fmt='.2f'
                         )

        _set_probability_maps_axes(ax, features_df, features, map_type)

        the_title = " ".join(["confidence high", "-", dataset_id])
        fig.suptitle(the_title, fontsize=16)

        store_to = "-".join(
            ["RQ3", "confidence-high", "DeepHyperion", features[0], features[1], dataset_id, map_type])
        _store_figure_to_paper_folder(fig, store_to)




        # Create the Total Samples Maps
        fig, ax = plt.subplots(figsize=(8, 8))
        # Create the color map - ??? COLOR?
        cmap = sns.color_palette("viridis", as_cmap=True)

        # Set WHITE for the cells for which we do not have observations (NaN)
        cmap.set_bad(color='white')
        # Show the average probabilities in the map
        ax = sns.heatmap(total_inputs, square=True,
                         vmin=0.0, vmax=1.0, cmap=cmap,
                         cbar_kws={"shrink": .6},  # reduce size of the color bar
                         annot_kws={"size": 5},
                         linewidths=1,
                         annot=True,
                         fmt='d'
                         )

        _set_probability_maps_axes(ax, features_df, features, map_type)

        the_title = " ".join(["total_inputs", "-", dataset_id])
        fig.suptitle(the_title, fontsize=16)

        store_to = "-".join(
            ["RQ3", "total_inputs", "DeepHyperion", features[0], features[1], dataset_id, map_type])
        _store_figure_to_paper_folder(fig, store_to)



        # Create the Total MISB
        fig, ax = plt.subplots(figsize=(8, 8))
        # Create the color map - ??? COLOR?
        cmap = sns.color_palette("flare", as_cmap=True)
        # Set WHITE for the cells for which we do not have observations (NaN)
        cmap.set_bad(color='white')
        # Show the average probabilities in the map
        ax = sns.heatmap(total_misb, square=True,
                         vmin=0.0, vmax=1.0, cmap=cmap,
                         cbar_kws={"shrink": .6},  # reduce size of the color bar
                         annot_kws={"size": 5},
                         linewidths=1,
                         annot=True,
                         fmt='d'
                         )

        _set_probability_maps_axes(ax, features_df, features, map_type)

        the_title = " ".join(["total_misb", "-", dataset_id])
        fig.suptitle(the_title, fontsize=16)

        store_to = "-".join(
            ["RQ3", "total_misb", "DeepHyperion", features[0], features[1], dataset_id, map_type])
        _store_figure_to_paper_folder(fig, store_to)






@cli.command()
@click.pass_context
def rq3(ctx):
    """
    RQ3: Feature Discrimination
        Context: The map of elites can be a useful tool to gain insights about the combinations of feature values that
        are likely to expose a failure. The regions of the map where there is a high probability of misbehaviours
        indicate that the corresponding feature value combinations are very likely to expose faults.

        Question: How do combinations of features discriminate failure-inducing inputs?
        Metrics:
            Limited to DH configurations:
                Probability of misbehaviour for each cell = (#misbehaving samples/ #generated samples) for each cell
                Qualitative discussion of combination of features
                    Note: This is a combination of previous RQ4 and RQ5
                    Note #2: For now, we show the probability in a Figure
                    Note #3: If we do not have enough data, we can use the maps from other tools
    """

    generate_average_probability_maps("MNIST", "data/mnist")

    generate_average_probability_maps("BEAMNG", "data/beamng")

if __name__ == '__main__':
    cli()