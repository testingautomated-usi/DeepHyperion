import math
from sklearn.linear_model import LinearRegression
import re
from copy import deepcopy
import xml.etree.ElementTree as ET
import numpy as np
from utils import curv_angle

NAMESPACE = '{http://www.w3.org/2000/svg}'


def bitmap_count(digit, threshold):
    image = deepcopy(digit.purified)
    bw = np.asarray(image)
    #bw = bw / 255.0
    count = 0
    for x in np.nditer(bw):
        if x > threshold:
            count += 1
    return count


def move_distance(digit):
    root = ET.fromstring(digit.xml_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    pattern = re.compile('([\d\.]+),([\d\.]+)\sM\s([\d\.]+),([\d\.]+)')
    segments = pattern.findall(svg_path)
    if len(segments) > 0:
        dists = [] # distances of moves
        for segment in segments:
            x1 = float(segment[0])
            y1 = float(segment[1])
            x2 = float(segment[2])
            y2 = float(segment[3])
            dist = math.sqrt(((x1-x2)**2)+((y1-y2)**2))
            dists.append(dist)
        return int(np.sum(dists))
    else:
        return 0


def orientation_calc(digit, threshold):
    x = []
    y = []
    image = deepcopy(digit.purified)
    bw = np.asarray(image)
    for iz, ix, iy, ig in np.ndindex(bw.shape):
        if bw[iz, ix, iy, ig] > threshold:
            x.append([iy])
            y.append(ix)
    X = np.array(x)
    Y = np.array(y)
    lr = LinearRegression(fit_intercept=True).fit(X, Y)
    normalized_ori = -lr.coef_ 
    new_ori = normalized_ori * 100
    return int(new_ori)

def angle_calc(digit):
     angles = []
     root = ET.fromstring(digit.xml_desc)
     svg_path = root.find(NAMESPACE + 'path').get('d')
     pattern = re.compile('([\d\.]+),([\d\.]+)\sC\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s([\d\.]+),([\d\.]+)\s')
     segments = pattern.findall(svg_path)
     for segment in segments:        
         angle = curv_angle(float(segment[0]), float(segment[1]), float(segment[2]), float(segment[3]), float(segment[4]), float(segment[5]), float(segment[6]), float(segment[7]))
         angles.append(angle)
     avg_ang = np.min(angles)
     return int(avg_ang)