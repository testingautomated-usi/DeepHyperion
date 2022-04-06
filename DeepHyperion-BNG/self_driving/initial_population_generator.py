import sys
import os
from pathlib import Path
import time
import numpy as np
import random
import logging as log

path = Path(os.path.abspath(__file__))
# This corresponds to DeepHyperion-BNG
sys.path.append(str(path.parent))
sys.path.append(str(path.parent.parent))

from core.folder_storage import SeedStorage
import glob, json
from random import shuffle, choice
from self_driving.road_bbox import RoadBoundingBox
from self_driving.beamng_individual import BeamNGIndividual
import core.utils as us
from scipy.spatial import distance
from core.feature_dimension import FeatureDimension
from self_driving.simulation_data import SimulationParams, SimulationData, SimulationDataRecord, SimulationInfo
from self_driving.vehicle_state_reader import VehicleState
from self_driving.beamng_member import BeamNGMember
from self_driving.decal_road import DecalRoad

def get_spine(member):
    print("member: ", member)
    with open(member) as json_file:
        spine = json.load(json_file)
        return spine['sample_nodes']

def get_min_distance_from_set(ind, solution):
    distances = list()
    ind_spine = ind[0]

    for road in solution:
        road_spine = road[0]
        distances.append(manhattan_dist(ind_spine, road_spine))
    distances.sort()
    return distances[0]

def manhattan_dist(ind1, ind2):
    dist = 0
    for i in range(0,len(ind1)):
        dist = dist + abs(ind1[i] - ind2[i])
    return dist


def initial_pool_generator(config, problem):
    good_members_found = 0
    attempts = 0
    storage = SeedStorage('initial_pool')
    i = 0
    while good_members_found < config.POOLSIZE:
        path = storage.get_path_by_index(good_members_found + 1)
        # if path.exists():
        #     print('member already exists', path)
        #     good_members_found += 1
        #     continue
        attempts += 1
        log.debug(f'attempts {attempts} good {good_members_found} looking for {path}')

        max_angle = random.randint(10,100)

        member = problem.generate_random_member(max_angle)
        member.evaluate()
        if member.distance_to_boundary <= 0:
            continue
        #member = problem.member_class().from_dict(member.to_dict())
        member.config = config
        member.problem = problem
        #member.clear_evaluation()

        #member.distance_to_boundary = None
        good_members_found += 1
       # path.write_text(json.dumps(member.to_dict()))
        path.write_text(json.dumps({
            "control_nodes": member.control_nodes,
            member.simulation.f_params: member.simulation.params._asdict(),
            member.simulation.f_info: member.simulation.info.__dict__,
            member.simulation.f_road: member.simulation.road.to_dict(),
            member.simulation.f_records: [r._asdict() for r in member.simulation.states]
        }))
    return storage.folder


def initial_population_generator(path, config, problem):
    all_roads = [filename for filename in glob.glob(str(path)+"\*.json", recursive=True)]
    type = config.Feature_Combination
    shuffle(all_roads)

    roads = all_roads

    original_set = list()

    individuals = []
    popsize = config.POPSIZE

    for road in roads:
        with open(road) as json_file:
            data = json.load(json_file)
            sample_nodes = data["road"]["nodes"]
            for node in sample_nodes:
                node[2] = -28.0
            sample_nodes = np.array(sample_nodes)
            records = data["records"]

        bbox_size = (-250.0, 0.0, 250.0, 500.0)
        road_bbox = RoadBoundingBox(bbox_size)
        member = BeamNGMember(data["control_nodes"], [tuple(t) for t in sample_nodes], 20,
                              road_bbox)
        member.config = config
        member.problem = problem
        simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
        sim_name = member.config.simulation_name.replace('$(id)', simulation_id)
        simulation_data = SimulationData(sim_name)
        states = []
        for record in records:
            state = VehicleState(timer=record["timer"]
                                 , pos=record["pos"]
                                 , dir=record["dir"]
                                 , vel=record["vel"]
                                 , steering=record["steering"]
                                 , steering_input=record["steering_input"]
                                 , brake=record["brake"]
                                 , brake_input=record["brake_input"]
                                 , throttle=record["throttle"]
                                 , throttle_input=record["throttle_input"]
                                 , wheelspeed=record["wheelspeed"]
                                 , vel_kmh=record["vel_kmh"])

            sim_data_record = SimulationDataRecord(**state._asdict(),
                                                   is_oob=record["is_oob"],
                                                   oob_counter=record["oob_counter"],
                                                   max_oob_percentage=record["max_oob_percentage"],
                                                   oob_distance=record["oob_distance"])
            states.append(sim_data_record)

        simulation_data.params = SimulationParams(beamng_steps=data["params"]["beamng_steps"],
                                                  delay_msec=int(data["params"]["delay_msec"]))

        simulation_data.road = DecalRoad.from_dict(data["road"])
        simulation_data.info = SimulationInfo()
        simulation_data.info.start_time = data["info"]["start_time"]
        simulation_data.info.end_time = data["info"]["end_time"]
        simulation_data.info.elapsed_time = data["info"]["elapsed_time"]
        simulation_data.info.success = data["info"]["success"]
        simulation_data.info.computer_name = data["info"]["computer_name"]
        simulation_data.info.id = data["info"]["id"]

        simulation_data.states = states
        if len(states) > 0:
            member.distance_to_boundary = simulation_data.min_oob_distance()
            member.simulation = simulation_data
            individual: BeamNGIndividual = BeamNGIndividual(member, config)
        #individual.evaluate()
        b = tuple()
        feature_dimensions = generate_feature_dimension(type)
        for ft in feature_dimensions:
            i = feature_simulator(ft.feature_simulator, individual)
            b = b + (i,)
        individuals.append([b, road, individual])

    starting_point = choice(individuals)
    original_set.append(starting_point)

    i = 0
    best_ind = individuals[0]
    while i < popsize - 1:
        max_dist = 0
        for ind in individuals:
            dist = get_min_distance_from_set(ind, original_set)
            if dist > max_dist:
                max_dist = dist
                best_ind = ind
        original_set.append(best_ind)
        i += 1

    base = config.initial_population_folder
    storage = SeedStorage(base)
    for index, road in enumerate(original_set):
        dst = storage.get_path_by_index(index + 1)
        ind = road[2]
        #copy(road[1], dst)
        with open(road[1]) as ff:
            json_file = json.load(ff)
        with open(dst, 'w') as f:
            f.write(json.dumps({
                "control_nodes": json_file["control_nodes"],
                ind.m.simulation.f_params: ind.m.simulation.params._asdict(),
                ind.m.simulation.f_info: ind.m.simulation.info.__dict__,
                ind.m.simulation.f_road: ind.m.simulation.road.to_dict(),
                ind.m.simulation.f_records: [r._asdict() for r in ind.m.simulation.states]
            }))

def feature_simulator(function, x):
    """
    Calculates the number of control points of x's svg path/number of bitmaps above threshold
    :param x: genotype of candidate solution x
    :return:
    """
    if function == 'min_radius':
        return us.new_min_radius(x)
    if function == 'mean_lateral_position':
        return us.mean_lateral_position(x)
    if function == "dir_coverage":
        return us.direction_coverage(x)
    if function == "segment_count":
        return us.segment_count(x)
    if function == "sd_steering":
        return us.sd_steering(x)
    if function == "curvature":
        return us.curvature(x)


def generate_feature_dimension(combination):
    fts = list()

    if "MinRadius" in combination:
            ft1 = FeatureDimension(name="MinRadius", feature_simulator="min_radius", bins=1)
            fts.append(ft1)

    if "MeanLateralPosition" in combination:
        ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
        fts.append(ft3)

    if "DirectionCoverage" in combination:
        ft2 = FeatureDimension(name="DirectionCoverage", feature_simulator="dir_coverage", bins=1)
        fts.append(ft2)

    if "SegmentCount" in combination:
        ft2 = FeatureDimension(name="SegmentCount", feature_simulator="segment_count", bins=1)
        fts.append(ft2)

    if  "SDSteeringAngle" in combination:
        ft2 = FeatureDimension(name="SDSteeringAngle", feature_simulator="sd_steering", bins=1)
        fts.append(ft2)

    if "Curvature" in combination:
        ft3 = FeatureDimension(name="Curvature", feature_simulator="curvature", bins=1)
        fts.append(ft3)

    return fts


if __name__ == '__main__':
    path = initial_pool_generator()
    #path = r"C:\Users\Aurora\new-DeepJanus\DeepJanus\DeepJanus-BNG\data\member_seeds\initial_pool"
    initial_population_generator(path)
