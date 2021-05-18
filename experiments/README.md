# Experimental Evaluation: Data and Scripts #

## General Information ##

This folder contains the data we obtained by conducting the experimental procedure described in the paper (Section 4). We used this data to generate the plots reported in the paper.

## Dependencies ##

The scripts require `make` and `python 3.7`

## (Re)Generate the plots ##

To regenerate the plots, run the following command from the **current** folder:

```
make plot-all
```

This command sets up a python virtual environment in a predefined location (`./.venv`), extracts into the `data/mnist` and `data/beamng` folders the raw data (feature-maps, probability-maps, etc.), and creates the various plots.

The command checks that the expected version of python is installed and fails otherwise.

The command produces a verbose output (sorry) and a few warnings like the following one:

```
...
plotting_utils.py:354: RuntimeWarning: Mean of empty slice
  avg_probabilities = np.nanmean(all_probabilities, axis=0)
...
```
 
Those warnings are expected, so do not worry.

If everything worked as expected you'll find the following plots under `./plots`:

* `RQ1-MNIST.pdf` (Figure 3: RQ1: Misbehaviours found by DeepHyperion, DeepJanus and DLFuzz on MNIST)
* `RQ1-BeamNG.pdf` (Figure 4: RQ1: Misbehaviours found by DeepHyperion and DeepJanus on BeamNG)  
* `RQ2-MNIST.pdf` (Figure 5: RQ2: Map cells filled by DeepHyperion, DeepJanus and DLFuzz on MNIST)
* `RQ2-BeamNG.pdf` (Figure 6: RQ2: Map cells filled by DeepHyperion and DeepJanus on BeamNG)
* `RQ3-MNIST.pdf`  (Figure 7: RQ3: Probability maps and feature discrimination for MNIST)
* `RQ3-BeamNG.pdf`  (Figure 8: RQ3: Probability maps and feature discrimination for BeamNG)

Those plots corresponds to the ones reported in Figures 3 -- 8 of the (pre-print) version of the paper.
