import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging as log
import sys
# For Python 3.6 we use the base keras
import keras
#from tensorflow import keras
import math
import numpy as np
# local imports
from properties import IMG_SIZE, INTERVAL

NAMESPACE = '{http://www.w3.org/2000/svg}'




def curv_angle(x11, y11, x12, y12, x21, y21, x22, y22):
     sameside = False
     if x22 - x11 == 0:
         x = x11
         if (x > x12 and x > x21) or (x < x12 and x < x21):
             sameside = True #both on same side
     else:
         m = (y22 - y11) / (x22 - x11)
         b = y11 - m * x11
         if (y12 > m * x12 + b and y21 > m * x21 + b) or (y12 < m * x12 + b and y21 < m *x21 + b):
             sameside = True #both on same side

     if sameside == True:
         if x12 - x11 == 0:
             A = 90
         else:
             y = (y12 - y11)
             x = (x12 - x11)  
             # A = angle between x-axis and line 1
             A = math.atan2(x,y) * 180 / math.pi
             A = np.abs((A + 180) % 360 - 180)

         if x22 - x21 == 0:
             B = 90
         else:
             y = (y22 - y21)
             x = (x22 - x21)  
             # B = angle between x-axis and line 2
             B = math.atan2(x, y) * 180 / math.pi
             B = np.abs((B + 180) % 360 - 180)
         #Angle between line 1 and line 2 = A - B
         angle = np.abs(A - B)
         return angle
     else:        
         # first angle

         if x12 - x11 == 0:
             A = 90
         else:
             y = (y12 - y11)
             x = (x12 - x11)  
             A = math.atan2(x,y) * 180 / math.pi
             A = np.abs((A + 180) % 360 - 180)

         if x21 - x12 == 0:
             B = 90
         else:
             y = (y21 - y12)
             x = (x21 - x12)         
             B = math.atan2(x,y) * 180 / math.pi
             B = np.abs((B + 180) % 360 - 180)

         #Angle between line 1 and line 2 = A - B
         angle1 = np.abs(A - B)

         # second angle

         if x21 - x12 == 0:
             A = 90
         else:
             y = (y21 - y12)
             x = (x21 - x12)       
             A = math.atan2(x,y) * 180 / math.pi
             A = np.abs((A + 180) % 360 - 180)
         if x22 - x21 == 0:
             B = 90
         else:
             y = (y22 - y21)
             x = (x22 - x21)        
             B = math.atan2(x,y) * 180 / math.pi        
             B = np.abs((B + 180) % 360 - 180)
         #Angle between line 1 and line 2 = A - B
         angle2 = np.abs(A - B)

         return np.min([angle1, angle2])


def compute_sparseness(map, x):
    n = len(map)
    # Sparseness is evaluated only if the archive is not empty
    # Otherwise the sparseness is 1
    if (n == 0) or (n == 1):
        sparseness = 0
    else:
        sparseness = density(map, x)
    return sparseness

def get_neighbors(b):
    neighbors = []
    neighbors.append((b[0], b[1]+1))
    neighbors.append((b[0]+1, b[1]+1))
    neighbors.append((b[0]-1, b[1]+1))
    neighbors.append((b[0]+1, b[1]))
    neighbors.append((b[0]+1, b[1]-1))
    neighbors.append((b[0]-1, b[1]))
    neighbors.append((b[0]-1, b[1]-1))
    neighbors.append((b[0], b[1]-1))

    return neighbors

def density(map, x):
    b = x.features
    density = 0
    neighbors = get_neighbors(b)
    for neighbor in neighbors:
        if neighbor not in map:
            density += 1
    return density



def input_reshape(x):
    # shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        x_reshape = x.reshape(x.shape[0], 1, 28, 28)
    else:
        x_reshape = x.reshape(x.shape[0], 28, 28, 1)
    x_reshape = x_reshape.astype('float32')
    x_reshape /= 255.0

    return x_reshape


def get_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def print_image(filename, image, cmap=''):
    if cmap != '':
        plt.imsave(filename, image.reshape(28, 28), cmap=cmap, format='png')
    else:
        plt.imsave(filename, image.reshape(28, 28), format='png')
    np.save(filename, image)


def rescale_map(features, perfs, new_min_1, new_max_1, new_min_2, new_max_2):
    if new_max_1 > 25:
        shape_1 = 25
    else:
        shape_1 = new_max_1 + 1
    
    if new_max_2 > 25:
        shape_2 = 25
    else:
        shape_2 = new_max_2 + 1

    output2 = np.full((shape_2, shape_1), np.inf, dtype=(float))

    original_bins1 = np.linspace(new_min_1, new_max_1, shape_1)
    original_bins2 = np.linspace(new_min_2, new_max_2, shape_2)

    for (i, j), value in np.ndenumerate(perfs):
        if i < new_max_2 and j < new_max_1:
            new_j = np.digitize(j, original_bins1, right=False)
            new_i = np.digitize(i, original_bins2, right=False)
            if value != np.inf:
                if output2[new_i, new_j] == np.inf or value < output2[new_i, new_j]:
                    output2[new_i, new_j] = value
                    #output1[new_i, new_j] = solutions[i, j]
    return output2

# Useful function that shapes the input in the format accepted by the ML model.
def reshape(v):
    v = (np.expand_dims(v, 0))
    # Shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        v = v.reshape(v.shape[0], 1, IMG_SIZE, IMG_SIZE)
    else:
        v = v.reshape(v.shape[0], IMG_SIZE, IMG_SIZE, 1)
    v = v.astype('float32')
    v = v / 255.0
    return v

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