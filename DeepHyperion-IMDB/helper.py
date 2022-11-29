
from features import compute_sentiment, compute_sentiment2
import csv
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.proportion as smp
from matplotlib.patches import Rectangle
import os
import json
from datasets import load_dataset

def compute_polarities(words, file):
    fw = open(file, 'w')
    cf = csv.writer(fw, lineterminator='\n')

    for word in words:
        cf.writerow([word, compute_sentiment(word)])

def enumerate2D(array1, array2):
    """
    https://stackoverflow.com/questions/44117612/enumerate-over-2-arrays-of-same-shape
    """
    assert array1.shape == array2.shape, "Error - dimensions."
    for indexes, data in np.ndenumerate(array1):
        yield indexes, data, array2[indexes]

# TODO For the moment look only at white box data
def load_probability_maps(dataset_folder, type="white-box"):

    rows_list = []
    for subdir, dirs, files in os.walk(dataset_folder, followlinks=False):

        # Extract metadata about features
        for json_data_file in [os.path.join(subdir, f) for f in files if
                         f.startswith("DeepHyperion") and
                        f.endswith("stats.json")]:

            with open(json_data_file, 'r') as input_file:
                # Get the JSON
                map_dict = json.load(input_file)

            # TODO Read those from the json maybe?
            # DLFuzz-017-Orientation-Moves-"+type+"-rescaled-stats.json
            attrs = json_data_file.split("-")

            run = attrs[1]
            # map_type = attrs[6].replace("-stats.npy", "")

            # Store the features data for this run - Is this a tuple ?!
            features = tuple(map_dict["Features"].keys())

            for feature_name, f in map_dict["Features"].items():
                rows_list.append({
                    'bins': [np.linspace(f["meta"]["min-value"], f["meta"]["max-value"], f["meta"]["num-cells"])],
                    'feature': feature_name,
                    'features': features,
                    # 'map type': map_type,
                    'run': int(run)
                })

    # Feature Map
    features_data = pd.DataFrame(rows_list, columns={'bins': pd.Series([], dtype='float'),
                                                     'features': pd.Series([], dtype='str'),
                                                     'feature': str(),
                                                    #  'map type': str(),
                                                     'run': int()})

    rows_list = []
    for subdir, dirs, files in os.walk(dataset_folder, followlinks=False):
        # Consider only the files that match the pattern
        for npy_file in [os.path.join(subdir, f) for f in files if
                         f.startswith("probability-DeepHyperion") and
                         (f.endswith(".npy") )]:

            probabilities = np.load(npy_file)
            attrs = npy_file.split("-")
            # probability-DeepJanusBeamNG-001-segment_count-sd_steering-SegmentCount-SDSteeringAngle-"+type+"-rescaled.npy
            features = (attrs[3].replace(".npy", ""), attrs[4].replace(".npy", ""))
            # map_type = attrs[9].replace(".npy", "")
            run = attrs[2]

            rows_list.append({
                'probabilities': probabilities,
                'features': features,
                # 'map type': map_type,
                'run': int(run)
            })

    probability_data = pd.DataFrame(rows_list, columns={'probabilities': pd.Series([], dtype='float'),
                                          'features': pd.Series([], dtype='str'),
                                        #   'map type': str(),
                                          'run': int()})

    rows_list = []
    for subdir, dirs, files in os.walk(dataset_folder, followlinks=False):
        # Consider only the files that match the pattern
        for npy_file in [os.path.join(subdir, f) for f in files if
                                              f.startswith("misbehaviour-DeepHyperion") and
                                               f.endswith(".npy")]:
            misbehaviors = np.load(npy_file)
            attrs = npy_file.split("-")

            features = (attrs[3].replace(".npy", ""), attrs[4].replace(".npy", ""))
            # map_type = attrs[9].replace(".npy", "")
            run = attrs[2]


            rows_list.append({
                'misbehaviors': misbehaviors,
                'features': features,
                # 'map type': map_type,
                'run': int(run)
            })

    misbehavior_data = pd.DataFrame(rows_list, columns={'misbehaviors': pd.Series([], dtype='float'),
                                                            'features': pd.Series([], dtype='str'),
                                                            # 'map type': str(),
                                                            'run': int()})

    rows_list = []
    for subdir, dirs, files in os.walk(dataset_folder, followlinks=False):
        # Consider only the files that match the pattern
        for npy_file in [os.path.join(subdir, f) for f in files if
                      f.startswith("coverage-DeepHyperion") and
                      (f.endswith(".npy"))]:
            coverage = np.load(npy_file)
            attrs = npy_file.split("-")

            features = (attrs[3].replace(".npy", ""), attrs[4].replace(".npy", ""))
            # map_type = attrs[9].replace(".npy", "")
            run = attrs[2]

            rows_list.append({
                'coverage': coverage,
                'features': features,
                # 'map type': map_type,
                'run': int(run)
            })

    # merge all the DF to obtain the last one
    coverage_data = pd.DataFrame(rows_list, columns={'coverage': pd.Series([], dtype='float'),
                                                        'features': pd.Series([], dtype='str'),
                                                        # 'map type': str(),
                                                        'run': int()})
    df = probability_data.merge(misbehavior_data, on=['features', 'run'])
    df = df.merge(coverage_data, on=['features', 'run'])

    return df, features_data



def generate_average_probability_maps(use_ax, probability_df, features,
                                      min_avg_prob=0.7999, min_low_ci=0.64999):

    # Take all the probability maps for rescaled
    # map_type = "rescaled"

    all_probabilities = list(probability_df[(probability_df["features"] == features)]["probabilities"])
    # Compute the mean ignoring Nan over the cells
    avg_probabilities = np.nanmean(all_probabilities, axis=0)

    # Load misb and coverage
    all_misbehaviors = list(probability_df[(probability_df["features"] == features)]["misbehaviors"])
    all_coverage = list(probability_df[(probability_df["features"] == features)]["coverage"])

    # Sum per each cell
    total_misb = np.nansum(all_misbehaviors, axis=0)
    total_inputs = np.nansum(all_coverage, axis=0)

    # Compute the confidence intervals per cell
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
    total_misb = np.transpose(total_misb)

    # Create the color map
    cmap = sns.cubehelix_palette(light=0.9, dark=0.5, as_cmap=True)
    # Set WHITE for the cells for which we do not have observations (NaN)
    cmap.set_bad(color='white')
    # Show the average probabilities in the map


    use_ax = sns.heatmap(avg_probabilities,
                         square=True,
                         vmin=0.0, vmax=1.0,
                         cmap=cmap,
                         cbar=None,
                         linewidths=1,
                         ax=use_ax
                         )

    # Highlighted cells: each cell is an empty square with a tick border.
    # Highlight the cells that value above 0.8 and low_ci above 0.65.
    # Note that we have j, i not i, j because we transposed the original data while here
    # we use the ax reference system and not the heatmap one
    for (j, i), prob_value, low_ci_value in enumerate2D(avg_probabilities, confident_data_low):
        if prob_value > min_avg_prob and low_ci_value > min_low_ci:
            use_ax.add_patch(Rectangle((i, j), 1, 1, fill=False, edgecolor='black', lw=2))

    return use_ax


def set_probability_maps_axes(ax, features_df, features, map_type, fontsize=24, min_fontsize=20):

    try:
        # Prepare the labels and ticks (reused across main map and supporting maps)
        f1_bins = list(features_df[(features_df["features"] == features) 
                                   & (features_df["feature"] == features[0])]["bins"].array[0][0])
        f2_bins = list(features_df[(features_df["features"] == features) 
                                   & (features_df["feature"] == features[1])]["bins"].array[0][0])

        ax.set_xticks(np.linspace(0, len(f1_bins)-1, len(f1_bins)))
        ax.set_yticks(np.linspace(0, len(f2_bins)-1, len(f2_bins)))

        # [unicode(x.strip()) if x is not None else '' for x in row]
        xtickslabel = [round(the_bin, 1) if idx % 2 == 0 else '' for idx, the_bin in enumerate(f1_bins)]
        ytickslabel = [round(the_bin, 1) if idx %2 == 0 else '' for idx, the_bin in enumerate(f2_bins)]

        ax.set_xticklabels(xtickslabel, fontsize=min_fontsize, rotation=45)
        ax.set_yticklabels(ytickslabel, fontsize=min_fontsize, rotation=0)


        ax.set_xlabel(features[0], fontsize=fontsize)
        ax.set_ylabel(features[1], fontsize=fontsize)

        # Add rotation


        # We need this to have the y axis start from zero at the bottom
        ax.invert_yaxis()

        # axis labels
        plt.xlabel(features[0], fontsize=14)
        plt.ylabel(features[1], fontsize=14)
    except Exception as e:
        print("Error in setting axes for", features, map_type)
        print(e)


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


PAPER_FOLDER="./plots"

def store_figure_to_paper_folder(figure, file_name):

    os.makedirs(PAPER_FOLDER)

    file_format = 'pdf'
    figure_file_name = "".join([file_name, ".", file_format])
    figure_file = os.path.join(PAPER_FOLDER, figure_file_name)

    # https://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
    figure.tight_layout()
    figure.savefig(figure_file, format=file_format, bbox_inches='tight')

    print("Plot stored to ", figure_file)


import predictor

if __name__ == "__main__":

    # file = open('opinion-lexicon-English/negative-words.txt', 'r')
    # neg_words = file.read().split()

    # file_name = 'opinion-lexicon-English/negative-words-pol.csv'
    # compute_polarities(neg_words, file_name)

    # file = open('opinion-lexicon-English/positive-words.txt', 'r')
    # pos_words = file.read().split()
    # file_name = 'opinion-lexicon-English/positive-words-pol.csv'
    # compute_polarities(pos_words, file_name)
    # from datasets import load_dataset

    # DATASET_DIR = "data"
    # test_ds = load_dataset('imdb', cache_dir=f"{DATASET_DIR}/imdb", split='test')
    # x_test, y_test = test_ds['text'], test_ds['label']

    # with open("starting_seeds_pos.txt", "w") as file:
    #     for i in range(len(x_test)):
    #         if y_test[i] == 1:
    #             print(".", end='', flush=True)
    #             # prediction, _ = predictor.Predictor.predict(x_test[i]) 

    #             # if prediction == 0:
    #             file.write(str(i)+",")

    print("Preparing IMDB plot")
    probability_df, features_df = load_probability_maps("./data/experiments/")
    mnist_fig = preprare_the_figure(probability_df, features_df, [('poscount', 'negcount'),
                                                                  ('verbcount', 'poscount'),
                                                                  ('verbcount', 'negcount')])
    print("Storing IMDB plot")
    store_figure_to_paper_folder(mnist_fig, file_name="RQ3-IMDB")