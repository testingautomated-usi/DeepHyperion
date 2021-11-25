import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Rectangle
from svgpathtools import svg2paths, wsvg
import os
import re
import numpy as np
import glob
import cv2
import pandas as pd

NAMESPACE = '{http://www.w3.org/2000/svg}'

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

    cmap = sns.cubehelix_palette(as_cmap=True)

    # Set the color for the under the limit to be white (0.0) so empty cells are not visualized
    # cmap.set_under('-1.0')
    # Plot NaN in white
    cmap.set_bad(color='white')    
    
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
                xlabel,
                ylabel,       
                min_value_x=0,
                max_value_x=25,
                min_value_y=0,
                max_value_y=25,                            
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


    if max_value_x > 25:
        num_cells_x = 25
    else:
        num_cells_x = max_value_x + 1
    
    if max_value_y > 25:
        num_cells_y = 25
    else:
        num_cells_y = max_value_y + 1



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
 

def plot_archive_rescaled(data,
                labels,
                xlabel,
                ylabel,       
                min_value_x=0,
                max_value_x=25,
                min_value_y=0,
                max_value_y=25,                            
                minimization=False,
                savefig_path=None,
                 ):
    plt.clf()
    plt.cla()

    str_labels = np.full(labels.shape, "", dtype=(str))

    for (i, j), value in np.ndenumerate(labels):
        if value > 0:
            str_labels[i][j] = value

    annotations = str_labels

    ax = sns.heatmap(data,  annot=annotations, fmt="s")
    ax.invert_yaxis()
    plt.xlabel(xlabel.name)
    plt.ylabel(ylabel.name)

    
    num_cells_x = 25
    num_cells_y = 25


    if max_value_x > 25:
        num_cells_x = 25
    else:
        num_cells_x = max_value_x + 1
    
    if max_value_y > 25:
        num_cells_y = 25
    else:
        num_cells_y = max_value_y + 1



    xtickslabel = [round(the_bin, 1) for the_bin in np.linspace(min_value_x, max_value_x, num_cells_x)]
    ytickslabel = [round(the_bin, 1) for the_bin in np.linspace(min_value_y, max_value_y, num_cells_y)]

    ax.set_xticklabels(xtickslabel)
    plt.xticks(rotation=30)
    ax.set_yticklabels(ytickslabel)
    plt.yticks(rotation=0)

    #ax.set_aspect("equal")
    # get figure to save to file
    if savefig_path:
        ht_figure = ax.get_figure()
        fig_name = savefig_path+"/archive_"+xlabel.name+"_"+ylabel.name
        print(os.path.abspath(fig_name))
        ht_figure.savefig(fig_name, dpi=400)

    plt.clf()
    plt.cla()
    plt.close()


def plot_heatmap_rescaled_expansion(data, expansion,
                xlabel,
                ylabel,       
                min_value_x=0,
                max_value_x=25,
                min_value_y=0,
                max_value_y=25,                            
                minimization=False,
                savefig_path=None,
                 ):
    plt.clf()
    plt.cla()
    ax = sns.heatmap(data)
    ax.invert_yaxis()
    plt.xlabel(xlabel.name)
    plt.ylabel(ylabel.name)

    
    num_cells_x = 25
    num_cells_y = 25


    if max_value_x > 25:
        num_cells_x = 25
    else:
        num_cells_x = max_value_x + 1
    
    if max_value_y > 25:
        num_cells_y = 25
    else:
        num_cells_y = max_value_y + 1



    xtickslabel = [round(the_bin, 1) for the_bin in np.linspace(min_value_x, max_value_x, num_cells_x)]
    ytickslabel = [round(the_bin, 1) for the_bin in np.linspace(min_value_y, max_value_y, num_cells_y)]

    ax.set_xticklabels(xtickslabel)
    plt.xticks(rotation=30)
    ax.set_yticklabels(ytickslabel)
    plt.yticks(rotation=0)

    for index in expansion:
        ax.add_patch(Rectangle((index[0], index[1]), 1, 1, fill=False, edgecolor='red', lw=2))
    ax.set_aspect("equal")
    # get figure to save to file
    if savefig_path:
        ht_figure = ax.get_figure()
        fig_name = savefig_path+"/expansion_"+xlabel.name+"_"+ylabel.name
        print(os.path.abspath(fig_name))
        ht_figure.savefig(fig_name, dpi=400)

    plt.clf()
    plt.cla()
    plt.close()
 

def plot_svg(xml, filename):
    root = ET.fromstring(xml)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    wsvg(svg_path, filename=filename+'.svg')


def getImage(path):
    img = plt.imread(path)
    res = cv2.resize(img, dsize=(150, 150), interpolation=cv2.INTER_AREA)
    return OffsetImage(res)


def plot_roads(dir_path, xlabel, ylabel):    
    paths = glob.glob(dir_path + "/"+xlabel+"_"+ylabel+"/*.jpg")
    x=[]
    y=[]
    for a in paths:
        pattern = re.compile('[\d\.]+, [\d\.]+')
        segments = pattern.findall(a)
        for se in segments: 
            se = se.split(', ')
            x.append(int(se[0]))
            y.append(int(se[1]))

    plt.cla()
    
    fig, ax = plt.subplots(figsize=(35,35))
    #ax.scatter(x, y) 

    for x0, y0, path in zip(x, y, paths):
        
        ab = AnnotationBbox(getImage(path), (y0, x0), frameon=False)
        ax.add_artist(ab)
    
    ax.set_xlim(6,24)
    ax.set_ylim(6,24)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(np.arange(6, 25, 1)) 
    plt.yticks(np.arange(6, 25, 1)) 
    

    plt.grid(color='blue', linestyle='-', linewidth=0.09)
    ht_figure = ax.get_figure()
    ht_figure.savefig(dir_path+"/roads_"+xlabel+"_"+ylabel, dpi=400)
    
    plt.clf()    
    plt.close()