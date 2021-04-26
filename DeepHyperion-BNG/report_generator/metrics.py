import logging as log

import copy, re, math,os
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.linear_model import LinearRegression
from itertools import tee
from shapely.geometry import LineString, Polygon, Point

THE_NORTH = [0,1]

########################################################################
#
#   BeamNG metrics follows: Taken from Tara's DeepHyperion-BNG code
#
########################################################################
def oob_distance(pos, right_polyline) -> float:
    """Returns the difference between the width of a lane and
    the distance between the car and the center of the road."""
    car_point = Point(pos)
    road_width = 8.0
    divisor = 4.0

    distance = right_polyline.distance(car_point)

    difference = road_width / divisor - distance
    return difference


def _define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return np.inf

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return radius


def capped_min_radius(max_radius_value, road_nodes, w=5):
    """

    Args:
        max_radius_value: Decides where to cap the min_radius, i.e., at which level we interpret a gentle road as a
        straight segment
        road_nodes:
        w:

    Returns:
        the min radius or the max value to indicate straight segments

    """
    min_radius_value = min_radius(road_nodes, w)
    return min_radius_value if min_radius_value < max_radius_value and min_radius_value != 0 else max_radius_value

def min_radius(road_nodes, w=5):
    """
    Args:
        road_nodes:
        w: window size (?)

    Returns:
        the minimum value of the curvature radius of the road defined by the given list of road_nodes
    """
    mr = np.inf
    for i in range(len(road_nodes) - w):
        p1 = road_nodes[i]
        p2 = road_nodes[i + int((w-1)/2)]
        p3 = road_nodes[i + (w-1)]

        radius = _define_circle(p1, p2, p3)
        if radius < mr:
            mr = radius

    if mr == np.inf:
        mr = 0

    # TODO  not sure what 3.2 ... means
    return int(mr * 3.280839895)


def _calc_angle_distance(v0, v1):
    at_0 = np.arctan2(v0[1], v0[0])
    at_1 = np.arctan2(v1[1], v1[0])
    return at_1 - at_0


def _calc_dist_angle(points):
    assert len(points) >= 2, f'at least two points are needed'

    def vector(idx):
        return np.subtract(points[idx + 1], points[idx])

    n = len(points) - 1
    result = [None] * (n)

    b = vector(0)
    for i in range(n):
        a = b
        b = vector(i)
        angle = _calc_angle_distance(a, b)
        distance = np.linalg.norm(b)
        result[i] = (angle, distance, [points[i+1], points[i]])
    return result

def _identify_segment(road_nodes):
    # result is angle, distance, [x2,y2], [x1,y1]
    result = _calc_dist_angle(road_nodes)

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
            if (angle_1) > 0:
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
                item = [prev[-1]] * len(item)
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
        # count += (len(g) - 1)
        count += (len(g))
        # TODO: count -1?
        segment_indexes.append(count)

    # TODO
    # segment_indexes.append(len(turns) - 1)

    segment_begin = 0
    for idx in segment_indexes:
        segment = []
        # segment_end = idx + 1
        segment_end = idx
        for j in range(segment_begin, segment_end):
            if j == 0:
                segment.append([result[j][2][0], result[j][0]])
            segment.append([result[j][2][1], result[j][0]])
        segment_begin = segment_end
        segments.append(segment)

    return segment_count, segments


def segment_count(road_nodes):
    count, segments = _identify_segment(road_nodes)
    return count  # , segments



# https://docs.python.org/3/library/itertools.html
# Itertools Recipes
def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def direction_coverage(road_node, n_bins=36):
    """Measure the coverage of road directions w.r.t. to the North (0,1) using the control points of the given road
    to approximate the road direction. BY default we use 36 bins to have bins of 10 deg each"""
    # Note that we need n_bins+1 because the interval for each bean is defined by 2 points
    coverage_buckets = np.linspace(0.0, 360.0, num=n_bins+1)
    direction_list = []
    for a, b in _pairwise(road_node):
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


def sd_steering(simulation_states):
    steering = []
    for state in simulation_states:
        steering.append(state["steering"])
    sd_steering = np.std(steering)
    return int(sd_steering)


def mean_lateral_position(states, right_poly):
    # here we commented out the mean_lateral_position from DH
    # since we do not have the correct oob_distance from AF experiments
    #lp = []
    #for state in states:
    #    lp.append(state["oob_distance"])
    #mean_lp_old = np.mean(lp) * 100

    lp = []
    for state in states:
       dist = oob_distance(state["pos"], right_poly)

       #sim_dist = state["oob_distance"]
       #assert (dist == sim_dist)
       lp.append(dist)
    mean_lp = np.mean(lp) * 100
    #assert (mean_lp == mean_lp_old)

    return int(mean_lp)


def calc_point_edges(p1, p2):
    origin = np.array(p1[0:2])

    a = np.subtract(p2[0:2], origin)
    # print(p1, p2)
    v = (a / np.linalg.norm(a)) * p1[3] / 2

    l = origin + np.array([-v[1], v[0]])
    r = origin + np.array([v[1], -v[0]])
    return tuple(l), tuple(r)


def get_geometry(middle_nodes):
    middle = []
    right = []
    left = []
    n = len(middle) + len(middle_nodes)

    # add middle nodes (road points): adds central spline
    middle += list(middle_nodes)
    left += [None] * len(middle_nodes)
    right += [None] * len(middle_nodes)

    # recalculate nodes: adds points of the lateral lane margins
    for i in range(n - 1):
        l, r = calc_point_edges(middle[i], middle[i + 1])
        left[i] = l
        right[i] = r
    # the last middle point
    right[-1], left[-1] = calc_point_edges(middle[-1], middle[-2])

    road_geometry = list()
    for index in range(len(middle)):
        point = dict()
        point['middle'] = middle[index]
        # Copy the Z value from middle
        point['right'] = list(right[index])
        point['right'].append(middle[index][2])
        # Copy the Z value from middle
        point['left'] = list(left[index])
        point['left'].append(middle[index][2])

        road_geometry.append(point)

    return road_geometry


def is_oob(road_nodes, simulation_states):
    # Create the road geometry from the nodes. At this point nodes have been reversed already if needed.
    road_geometry = get_geometry(road_nodes)

    # Compute right polygon
    # Create the right lane polygon from the geometry
    left_edge_x = np.array([e['middle'][0] for e in road_geometry])
    left_edge_y = np.array([e['middle'][1] for e in road_geometry])
    right_edge_x = np.array([e['right'][0] for e in road_geometry])
    right_edge_y = np.array([e['right'][1] for e in road_geometry])

    # Compute the "short" edge at the end of the road to rule out false positives
    shorty = LineString([(left_edge_x[-1], left_edge_y[-1]), (right_edge_x[-1], right_edge_y[-1])]).buffer(2.0)

    # Note that one must be in reverse order for the polygon to close correctly
    right_edge = LineString(zip(right_edge_x[::-1], right_edge_y[::-1]))
    left_edge = LineString(zip(left_edge_x, left_edge_y))

    l_edge = left_edge.coords
    r_edge = right_edge.coords

    right_lane_polygon = Polygon(list(l_edge) + list(r_edge))

    #middle = [e['middle'] for e in road_geometry]
    #right = [e['right'] for e in road_geometry]
    #road_poly = [(p[0], p[1]) for p in middle]
    #right = [(p[0], p[1]) for p in right]
    #road_poly.extend(right[::-1])
    #right_polygon = Polygon(road_poly)


    first_oob_state = None
    position_of_first_oob_state = None
    for idx, simulation_state in enumerate(simulation_states):
        position = Point(simulation_state["pos"][0], simulation_state["pos"][1])
        if not right_lane_polygon.contains(position):

            # As soon as an observation is outside the lane polygon we mark the OOB, and return that position. All the
            # subsequent states will be removed/discarded
            log.debug("First OOB state found at %d", idx)
            first_oob_state = idx
            position_of_first_oob_state = position

            break

    if first_oob_state is not None:
        if shorty.contains(position_of_first_oob_state):
            log.info("*    False Positive. Short Edge")
            return False, None
        else:
            return True, first_oob_state
    else:
        return False, None
