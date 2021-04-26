import numpy as np
import matplotlib.pyplot as plt
import random
import os, json, glob

DST = 'road_images'
SRC = 'deepjanus_roads/*'

def save(img1,idx):
    folder = 'road_image_paper'
    if not os.path.exists(folder):
        os.makedirs(folder)

    fig = plt.figure()
    plt.imshow(plt.imread(img1))
    plt.axis('off')
    imgpath = os.path.join(folder, idx+'.svg')
    fig.savefig(imgpath)
    plt.close(fig)
    print(imgpath)


def plot_on_ax(middle, left, right, ax):
    def _plot_xy(points, color, linewidth):
        tup = list(zip(*points))
        ax.plot(tup[0], tup[1], color=color, linewidth=linewidth)

    ax.set_facecolor('#7D9051')  # green
    _plot_xy(middle, '#FEA952', linewidth=3)  # arancio
    _plot_xy(left, 'white', linewidth=3)
    _plot_xy(right, 'white', linewidth=3)
    ax.axis('equal')

def plot_on_ax2(middle, left, right, ax):
    def _plot_xy(points, color, linewidth):
        tup = list(zip(*points))
        ax.scatter(tup[0], tup[1], color=color, linewidth=linewidth, marker='.')
        #ax.plot(tup[0], tup[1], color=color, linewidth=linewidth)

    ax.set_facecolor('#FFFFFF')  # green
    _plot_xy(middle, 'black', linewidth=0.0000001)  # arancio
    _plot_xy(left, 'black', linewidth=0.0000001)
    _plot_xy(right, 'black', linewidth=0.000001)
    ax.axis('equal')

def plot_on_ax3(control, ax):
    def _plot_xy(points, color, linewidth):
        tup = list(zip(*points))
        ax.scatter(tup[0], tup[1], s=[20*2 for n in range(len(points))], color=color, linewidth=linewidth, marker='o')
        #ax.plot(tup[0], tup[1], color=color, linewidth=linewidth)

    ax.set_facecolor('#FFFFFF')  # green
    _plot_xy(control, 'red', linewidth=10)  # arancio
    ax.axis('equal')


def calc_point_edges(p1, p2):
    origin = np.array(p1[0:2])

    a = np.subtract(p2[0:2], origin)
    #print(p1, p2)
    v = (a / np.linalg.norm(a)) * p1[3] / 2

    l = origin + np.array([-v[1], v[0]])
    r = origin + np.array([v[1], -v[0]])
    return tuple(l), tuple(r)


def get_geometry(middle_nodes):

    middle = []
    right = []
    left = []
    n = len(middle) + len(middle_nodes)

    middle += list(middle_nodes)
    left += [None] * len(middle_nodes)
    right += [None] * len(middle_nodes)
    for i in range(n - 1):
        l, r = calc_point_edges(middle[i], middle[i + 1])
        left[i] = l
        right[i] = r
    # the last middle point
    right[-1], left[-1] = calc_point_edges(middle[-1], middle[-2])
    return middle, left, right


if __name__ == '__main__':

    dst_path = DST
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    src_path = SRC
    roads = [filename for filename in glob.glob(src_path)]
    road = random.choice(roads)

    road_desc = json.loads(open(road).read())
    sample_spine = (road_desc['sample_nodes'])
    control_spine = (road_desc['control_nodes'])

    middle, left, right = get_geometry(sample_spine)

    fig, (ax, ax2) = plt.subplots(ncols=2)
    fig.set_size_inches(15, 10)

    plot_on_ax2(middle, left, right, ax)
    ax2.set_title('', fontsize=20)

    plot_on_ax(middle, left, right, ax2)
    ax2.axis('equal')

    plot_on_ax3(control_spine[1:-1], ax)
    ax.axis('equal')

    plt.tick_params(labelsize=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax2.tick_params(axis="x", labelsize=20)

    # plt.axis('off')
    fig.savefig(os.path.join(DST, 'road.png'))
    plt.close(fig)
