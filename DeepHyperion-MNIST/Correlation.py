import numpy as np

# For Python 3.6 we use the base keras
import keras

import re
import utils
import vectorization_tools
import rasterization_tools
from digit_input import Digit
from individual import Individual
from csv import reader
from scipy import stats
from PIL import Image


def generate_digit(seed):    
    xml_desc = vectorization_tools.vectorize(seed)
    return Digit(xml_desc, 5)

boldness= []
smoothness = []
continuity = []
orientation = []
imgs = []
# open file in read mode
with open('entity.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    pattern1 = re.compile('([\d\.]+)([\d\.]+)([\d\.]+).png')
    pattern2 = re.compile('boldness\[([-?\d\.]+)\],smoothness\[([-?\d\.]+)\],continuity\[([-?\d\.]+)\],orientation\[([-?\d\.]+)\]')
    for row in csv_reader:        
        segments = pattern1.findall(row[0])
        for segment in segments:
            imgs.append(str(segment[0])+str(segment[1])+str(segment[2]))    
        segments = pattern2.findall(row[0])        
        for segment in segments:
            boldness.append(int(segment[0]))
            smoothness.append(int(segment[1]))
            continuity.append(int(segment[2]))
            orientation.append(int(segment[3]))
            

         
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


bitmaps = []
angles = []    
moves = []
orientations = []
seeds = []

for i in range(len(x_test)):      
    if y_test[i] == 5 :
        seeds.append(x_test[i])

for i in range(len(imgs)):            
    seed = seeds[int(imgs[i])-1]          
    digit1 = generate_digit(seed)
    individual = Individual(digit1, seed)
    individual.seed = seed
    x = individual    
 
    bitmap_count = utils.bitmap_count(x.member, 0.6)
    angle_calc = utils.angle_calc(x.member)
    move_dist = utils.move_distance(x.member)
    orientation_calc = utils.orientation_calc(x.member, 0)

    bitmaps.append(bitmap_count)
    angles.append(angle_calc)
    moves.append(move_dist)
    orientations.append(orientation_calc)

boldness_bitmap = stats.pearsonr(boldness, bitmaps)
smoothness_angle = stats.pearsonr(smoothness, angles)
continuity_move = stats.pearsonr(continuity, moves)
orientation_orientation = stats.pearsonr(orientation, orientations)

print("correlation & p-value:")
print(f"boldness and bitmap: {boldness_bitmap}")
print(f"smoothness and angle: {smoothness_angle}")
print(f"continuity and move: {continuity_move}")
print(f"orientation and orientation: {orientation_orientation}")
        






# inds = []

# for i in range(len(x_test)-1):      
#     if y_test[i] == 5:
#         inds.append(x_test[i])


# ECU_dists = []
# for i in range(len(inds) - 1):
#     for j in range(len(inds) -1):
#         if i < j:
#             vec1 = vectorization_tools.vectorize(inds[1])
#             img1 = rasterization_tools.rasterize_in_memory(vec1)
#             vec2 = vectorization_tools.vectorize(inds[2])
#             img2 = rasterization_tools.rasterize_in_memory(vec2)
#             ecu_dist = utils.get_distance(img1, img2) 
#             ECU_dists.append(ecu_dist)


# avg_ecu_dist = np.mean(ECU_dists)

# print(len(ECU_dists))
# print("avg: ")
# print(avg_ecu_dist)


