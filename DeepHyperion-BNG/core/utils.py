import matplotlib
import os
import sys
# Make sure we do not open windows to show images
# TODO We need to make this call somewhere else...
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
import csv
import json
import glob
import numpy as np

# local imports
from core.curvature import findCircle, findCircleReturnCenterAndRadius, define_circle
from self_driving.edit_distance_polyline import _calc_dist_angle
from itertools import tee
import math
import logging as log
# Resampling and spline interpolation
from scipy.interpolate import splev, splprep
from numpy.ma import arange
from numpy import repeat, linspace
import functools
from itertools import islice
from shapely.geometry import Point, LineString


THE_NORTH = [0,1]
ANGLE_THRESHOLD = 0.005

def segment_count(x):
    nodes = x.m.sample_nodes
    
    count , segments = identify_segment(nodes)
    return count #, segments
    # TODO Note that this is identify_segments with a final 's'
    # segments = identify_segments(nodes)
    # return len(segments), segments

def rel_segment_count(x):
    nodes = x.m.sample_nodes
    
    count, segments = identify_segment(nodes)
    rel = (count/len(nodes))
    rel = rel/0.04093567251461988
    return int(rel*100) #, segments

# counts only turns, split turns
def identify_segment(nodes):
     # result is angle, distance, [x2,y2], [x1,y1]
     result = _calc_dist_angle(nodes)

     segments = []
     SEGMENT_THRESHOLD = 15
     SEGMENT_THRESHOLD2 = 10
     ANGLE_THRESHOLD = 0.005


     # iterate over the nodes to get the turns bigger than the threshold
     # a turn category is assigned to each node
     # l is a left turn
     # r is a right turn
     # s is a straight segment
     # TODO: first node is always a s
     turns = []
     for i in range(0, len(result)):
         # result[i][0] is the angle
         angle_1 = (result[i][0] + 180) % 360 - 180
         if np.abs(angle_1) > ANGLE_THRESHOLD:
             if(angle_1) > 0:
                 turns.append("l")
             else:
                 turns.append("r")
         else:
             turns.append("s")

     # this generator groups the points belonging to the same category
     def grouper(iterable):
         prev = None
         group = []
         for item in iterable:
             if not prev or item == prev:
                 group.append(item)
             else:
                 yield group
                 group = [item]
             prev = item
         if group:
             yield group

     # this generator groups:
     # - groups of points belonging to the same category
     # - groups smaller than 10 elements
     def supergrouper1(iterable):
         prev = None
         group = []
         for item in iterable:
             if not prev:
                 group.extend(item)
             elif len(item) < SEGMENT_THRESHOLD2 and item[0] == "s":
                 item = [prev[-1]] * len(item)
                 group.extend(item)
             elif len(item) < SEGMENT_THRESHOLD and item[0] != "s" and prev[-1] == item[0]:
                 item = [prev[-1]] * len(item)
                 group.extend(item)
             else:
                 yield group
                 group = item
             prev = item
         if group:
             yield group

     # this generator groups:
     # - groups of points belonging to the same category
     # - groups smaller than 10 elements
     def supergrouper2(iterable):
         prev = None
         group = []
         for item in iterable:
             if not prev:
                 group.extend(item)
             elif len(item) < SEGMENT_THRESHOLD:
                 item = [prev[-1]]*len(item)
                 group.extend(item)
             else:
                 yield group
                 group = item
             prev = item
         if group:
             yield group

     groups = grouper(turns)

     supergroups1 = supergrouper1(groups)

     supergroups2 = supergrouper2(supergroups1)

     count = 0
     segment_indexes = []
     segment_count = 0
     for g in supergroups2:
        if g[-1] != "s":
            segment_count += 1
        # TODO
        #count += (len(g) - 1)
        count += (len(g))
        # TODO: count -1?
        segment_indexes.append(count)

     # TODO
     #segment_indexes.append(len(turns) - 1)

     segment_begin = 0
     for idx in segment_indexes:
         segment = []
         #segment_end = idx + 1
         segment_end = idx
         for j in range(segment_begin, segment_end):
             if j == 0:
                 segment.append([result[j][2][0], result[j][0]])
             segment.append([result[j][2][1], result[j][0]])
         segment_begin = segment_end
         segments.append(segment)

     return segment_count, segments


# https://docs.python.org/3/library/itertools.html
# Itertools Recipes
def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def direction_coverage(x, n_bins=25):
    """Measure the coverage of road directions w.r.t. to the North (0,1) using the control points of the given road
    to approximate the road direction. BY default we use 36 bins to have bins of 10 deg each"""
    # Note that we need n_bins+1 because the interval for each bean is defined by 2 points
    coverage_buckets = np.linspace(0.0, 360.0, num=n_bins+1)
    direction_list = []
    for a, b in _pairwise(x.m.sample_nodes):
        # Compute the direction of the segment defined by the two points
        road_direction = [b[0] - a[0], b[1] - a[1]]
        # Compute the angle between THE_NORTH and the road_direction.
        # E.g. see: https://www.quora.com/What-is-the-angle-between-the-vector-A-2i+3j-and-y-axis
        # https://www.kite.com/python/answers/how-to-get-the-angle-between-two-vectors-in-python
        unit_vector_1 = road_direction/np.linalg.norm(road_direction)
        dot_product = np.dot(unit_vector_1, THE_NORTH)
        angle = math.degrees(np.arccos(dot_product))
        direction_list.append(angle)

    # Place observations in bins and get the covered bins without repetition
    covered_elements = set(np.digitize(direction_list, coverage_buckets))
    return int((len(covered_elements) / len(coverage_buckets))*100)

def min_radius(x, w=5):
    mr = np.inf
    mincurv = []
    nodes = x.m.sample_nodes
    for i in range(len(nodes) - w):
        p1 = nodes[i]
        p2 = nodes[i + int((w-1)/2)]
        p3 = nodes[i + (w-1)]
        #radius = findCircle(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
        radius = define_circle(p1, p2, p3)
        if radius < mr:
            mr = radius
            mincurv = [p1, p2, p3]

    if mr  > 90:
        mr = 90

    return int(mr*3.280839895)#, mincurv

def curvature(x, w=5):
    mr = np.inf
    mincurv = []
    nodes = x.m.sample_nodes
    for i in range(len(nodes) - w):
        p1 = nodes[i]
        p2 = nodes[i + int((w-1)/2)]
        p3 = nodes[i + (w-1)]
        #radius = findCircle(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
        radius = define_circle(p1, p2, p3)
        if radius < mr:
            mr = radius
            mincurv = [p1, p2, p3]

    curvature = (1/mr)*100

    return int(curvature)#, mincurv

def sd_steering(x):
    states = x.m.simulation.states
    steering = []
    for state in states:
        steering.append(state.steering)
    sd_steering = np.std(steering)
    return int(sd_steering)

def mean_lateral_position(x):
    states = x.m.simulation.states
    lp = []
    for state in states:
        lp.append(state.oob_distance)
    mean_lp = np.mean(lp) * 100
    return int(mean_lp)

def new_rescale(features, perfs, new_min_1, new_max_1, new_min_2, new_max_2):
    if new_max_1 > 25:
        shape_1 = 25
    else:
        shape_1 = new_max_1 + 1
    
    if new_max_2 > 25:
        shape_2 = 25
    else:
        shape_2 = new_max_2 + 1

    output2 = np.full((shape_2, shape_1), np.inf, dtype=(float))

    # interval1 = (new_max_1 - new_min_1) / shape_1
    # interval2 = (new_max_2 - new_min_2) / shape_2

    original_bins1 = np.linspace(new_min_1, new_max_1, shape_1)
    original_bins2 = np.linspace(new_min_2, new_max_2, shape_2)

    for (i, j), value in np.ndenumerate(perfs):
        # new_j = int(j/interval1)
        # new_i = int(i/interval2)
        if i < new_max_2 and j < new_max_1:
            new_j = np.digitize(j, original_bins1, right=False)
            new_i = np.digitize(i, original_bins2, right=False)
            if value != np.inf:
                if output2[new_i, new_j] == np.inf or value < output2[new_i, new_j]:
                    output2[new_i, new_j] = value
                    #output1[new_i, new_j] = solutions[i, j]
    return output2

def new_resampling(sample_nodes, dist=1.5):
    new_sample_nodes = []
    dists = []
    for i in range(1, len(sample_nodes)):
        x0 = sample_nodes[i-1][0]
        x1 = sample_nodes[i][0]
        y0 = sample_nodes[i - 1][1]
        y1 = sample_nodes[i][1]

        d = math.sqrt(math.pow((x1 - x0), 2) + math.pow((y1 - y0), 2))
        dists.append(d)
        if d >= dist:
            dt = dist
            new_sample_nodes.append([x0, y0, -28.0, 8.0])
            while dt <= d - dist:
                t = dt / d
                xt = ((1 - t) * x0 + t * x1)
                yt = ((1 - t) * y0 + t * y1)
                new_sample_nodes.append([xt, yt, -28.0, 8.0])
                dt = dt + dist
            new_sample_nodes.append([x1, y1, -28.0, 8.0])
        else:
            new_sample_nodes.append([x0, y0, -28.0, 8.0])
            new_sample_nodes.append([x1, y1, -28.0, 8.0])

    points_x = []
    points_y = []
    final_nodes = list()
    # discard the Repetitive points
    for i in range(1, len(new_sample_nodes)):
        if new_sample_nodes[i] != new_sample_nodes[i-1]:
            final_nodes.append(new_sample_nodes[i])
            points_x.append(new_sample_nodes[i][0])
            points_y.append(new_sample_nodes[i][1])
    return final_nodes

def setup_logging(log_to, debug):

    def log_exception(extype, value, trace):
        log.exception('Uncaught exception:', exc_info=(extype, value, trace))

    # Disable annoyng messages from matplot lib.
    # See: https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python
    log.getLogger('matplotlib.font_manager').disabled = True

    term_handler = log.StreamHandler()
    log_handlers = [term_handler]
    start_msg = "Started test generation"

    if log_to is not None:
        file_handler = log.FileHandler(log_to, 'a', 'utf-8')
        log_handlers.append( file_handler )
        start_msg += " ".join(["writing to file: ", str(log_to)])

    log_level = log.DEBUG if debug else log.INFO

    log.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=log_level, handlers=log_handlers)

    sys.excepthook = log_exception

    log.info(start_msg)

