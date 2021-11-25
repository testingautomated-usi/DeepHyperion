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
import statsmodels.stats.power as pw
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
    :returns the value estimate and the magnitude and power and number of runs to have power>0.8
    """
    m = len(treatment)
    n = len(control)

    # if m != n:
    #     raise ValueError("Data must have the same length")

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

    return estimate, magnitude#, power, nruns


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
    print("Comparing: %s,%s.\n \t p-Value %s - %s \n \t A12 %f - %s " %(
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
