import matplotlib.pyplot as plt
import seaborn as sns

import re
import numpy as np
import glob
import os
import pandas as pd

def plot_heatmap(data, 
                ylabel,
                xlabel,                                   
                minimization=False,
                savefig_path=None,
                 ):
    plt.clf()
    plt.cla()

    ser = pd.Series(list(data.values()),
                  index=pd.MultiIndex.from_tuples(data.keys()))
    df = ser.unstack().fillna(0)
    df = ser.unstack().fillna(np.inf)

    # figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # cmap = sns.cubehelix_palette(as_cmap=True)

    # Set the color for the under the limit to be white (0.0) so empty cells are not visualized
    # cmap.set_under('-1.0')
    # Plot NaN in white
    # cmap.set_bad(color='white')    
    
    ax = sns.heatmap(df)
    ax.invert_yaxis()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # get figure to save to file
    if savefig_path:
        ht_figure = ax.get_figure()
        fig_name = savefig_path+"/heatmap_"+xlabel+"_"+ylabel
        print(os.path.abspath(fig_name))
        ht_figure.savefig(fig_name)


    plt.clf()
    plt.cla()
    plt.close()
    
def plot_heatmap_rescaled(data, 
                ylabel,
                xlabel,       
                min_value_y=0,
                max_value_y=25,
                min_value_x=0,
                max_value_x=25,                            
                minimization=False,
                savefig_path=None,
                 ):
    plt.clf()
    plt.cla()

    plt.clf()
    plt.cla()

    ser = pd.Series(list(data.values()),
                  index=pd.MultiIndex.from_tuples(data.keys()))
    df = ser.unstack().fillna(0)
    df = ser.unstack().fillna(np.inf)

    # figure
    fig, ax = plt.subplots(figsize=(8, 8))

    cmap = sns.cubehelix_palette(as_cmap=True)

    # Set the color for the under the limit to be white (0.0) so empty cells are not visualized
    # cmap.set_under('-1.0')
    # Plot NaN in white
    cmap.set_bad(color='white')    
    
    ax = sns.heatmap(df)
    ax.invert_yaxis()
    plt.xlabel(xlabel.name)
    plt.ylabel(ylabel.name)

    
    num_cells_x = 25
    num_cells_y = 25


    # if max_value_x > 25:
    #     num_cells_x = 25
    # else:
    #     num_cells_x = max_value_x + 1
    
    # if max_value_y > 25:
    #     num_cells_y = 25
    # else:
    #     num_cells_y = max_value_y + 1



    xtickslabel = [round(the_bin, 1) for the_bin in np.linspace(min_value_x, max_value_x, num_cells_x)]
    ytickslabel = [round(the_bin, 1) for the_bin in np.linspace(min_value_y, max_value_y, num_cells_y)]

    ax.set_xticklabels(xtickslabel)
    plt.xticks(rotation=45)
    ax.set_yticklabels(ytickslabel)
    plt.yticks(rotation=0)

    # get figure to save to file
    if savefig_path:
        ht_figure = ax.get_figure()
        fig_name = savefig_path+"/heatmap_"+xlabel.name+"_"+ylabel.name
        print(os.path.abspath(fig_name))
        ht_figure.savefig(fig_name)

    plt.clf()
    plt.cla()
    plt.close()
 

