import logging as log

import copy, re, math,os
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.linear_model import LinearRegression
from itertools import tee


NAMESPACE = '{http://www.w3.org/2000/svg}'
BITMAP_THRESHOLD = 0.5

def dark_bitmaps(digit):
    count = 0
    iterator = np.nditer(digit)
    for x in iterator:
        if x > BITMAP_THRESHOLD:
            count += 1
    return count


def move_distance(digit):
    root = ET.fromstring(digit)
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


def orientation_calc(digit):
    x = []
    y = []
    image = copy.deepcopy(digit)
    bw = image
    for _, ix,iy, _ in np.ndindex(bw.shape):
        if bw[_,ix,iy,_] > 0:
            x.append([iy])
            y.append(ix)
    X = np.array(x)
    Y = np.array(y)
    lr = LinearRegression(fit_intercept=True).fit(X, Y)
    normalized_ori = (-lr.coef_ + 2)/4
    # scale to be between 0 and 100
    new_ori = normalized_ori * 100
    return int(new_ori)

