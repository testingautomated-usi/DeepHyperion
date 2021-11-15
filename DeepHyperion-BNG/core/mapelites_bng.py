import sys
import os
import json
import time
from os.path import join
from pathlib import Path
import numpy as np
from datetime import datetime
import logging as log
import shutil
#sys.path.insert(0, r'C:\DeepHyperion-BNG')
#sys.path.append(os.path.dirname(os.path.dirname(os.path.join(__file__))))
path = Path(os.path.abspath(__file__))
# This corresponds to DeepHyperion-BNG
sys.path.append(str(path.parent))
sys.path.append(str(path.parent.parent))

# local imports
import core.utils as us
from core.mapelites import MapElites
from core.feature_dimension import FeatureDimension
import self_driving.beamng_config as cfg
import self_driving.beamng_problem as BeamNGProblem
from self_driving.beamng_individual import BeamNGIndividual
from core.config import Config
from self_driving.road_bbox import RoadBoundingBox

from self_driving.simulation_data import SimulationParams, SimulationData, SimulationDataRecord, SimulationInfo

from self_driving.vehicle_state_reader import VehicleState
from self_driving.beamng_member import BeamNGMember
from self_driving.decal_road import DecalRoad

class MapElitesBNG(MapElites):

    def __init__(self, *args, **kwargs):
        super(MapElitesBNG, self).__init__(*args, **kwargs)

    def map_x_to_b(self, x):
        """
        Map X solution to feature space dimension
        :return: tuple of indexes
        """
        b = tuple()
        for ft in self.feature_dimensions:
            i = ft.feature_descriptor(self, x)
            if i < ft.min:
                ft.min = i
            b = b + (i,)
        return b

    def performance_measure(self, x):
        """
        Apply the fitness function to x
        """
        # "calculate performance measure"
        pref = x.evaluate()
        return pref

    def mutation(self, x, seed):
        """
        Mutate the solution x
        """
        # "apply mutation"
        road = x.m.clone()
        y: BeamNGIndividual = BeamNGIndividual(road, self.config)
        y.seed = seed
        y.mutate()
        return y

    def generate_random_solution(self):
        """
        To ease the bootstrap of the algorithm, we can generate
        the first solutions in the feature space, so that we start
        filling the bins
        """
        # "Generate random solution"
        seed = self.problem._seed_pool_strategy.get_seed()
        road = seed.clone().mutate()
        road.config = self.config
        individual: BeamNGIndividual = BeamNGIndividual(road, self.config)
        individual.seed = seed
        return individual

    def generate_random_solution_without_sim(self):
        """
        To ease the bootstrap of the algorithm, we can generate
        the first solutions in the feature space, so that we start
        filling the bins
        """
        # "Generate random solution"
        path = self.problem._seed_pool_strategy.get_seed()

        with open(path) as json_file:
            data = json.load(json_file)
            sample_nodes = data["road"]["nodes"]
            for node in sample_nodes:
                node[2] = -28.0
            sample_nodes = np.array(sample_nodes)
            records = data["records"]

        bbox_size = (-250.0, 0.0, 250.0, 500.0)
        road_bbox = RoadBoundingBox(bbox_size)
        member = BeamNGMember(data["control_nodes"], [tuple(t) for t in sample_nodes], len(data["control_nodes"]), road_bbox)
        member.config = self.config
        member.problem = self.problem
        simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
        sim_name = member.config.simulation_name.replace('$(id)', simulation_id)
        simulation_data = SimulationData(sim_name)
        states = []
        for record in records:
            state = VehicleState(timer=record["timer"]
                                 , damage=record["damage"]
                                 , pos=record["pos"]
                                 , dir=record["dir"]
                                 , vel=record["vel"]
                                 , gforces=record["gforces"]
                                 , gforces2=record["gforces2"]
                                 , steering=record["steering"]
                                 , steering_input=record["steering_input"]
                                 , brake=record["brake"]
                                 , brake_input=record["brake_input"]
                                 , throttle=record["throttle"]
                                 , throttle_input=record["throttle_input"]
                                 , throttleFactor=record["throttleFactor"]
                                 , engineThrottle=record["engineThrottle"]
                                 , wheelspeed=record["engineThrottle"]
                                 , vel_kmh=record["engineThrottle"])

            sim_data_record = SimulationDataRecord(**state._asdict(),
                                                   is_oob=record["is_oob"],
                                                   oob_counter=record["oob_counter"],
                                                   max_oob_percentage=record["max_oob_percentage"],
                                                   oob_distance=record["oob_distance"])
            states.append(sim_data_record)

        simulation_data.params = SimulationParams(beamng_steps=data["params"]["beamng_steps"], delay_msec=int(data["params"]["delay_msec"]))

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
            individual: BeamNGIndividual = BeamNGIndividual(member, self.config)
            individual.seed = path
        else:
            print("*********Bug************")
        return individual

    def generate_feature_dimensions(self, combination):
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

    @staticmethod
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

    def print_config(self, dir):
        config = {
            "Pop size": str(self.config.POPSIZE),
            "Pool size": str(self.config.POOLSIZE),
            "Mutation extent": str(self.config.MUTATION_EXTENT),
            "Mutation prob": str(self.config.MUTPB),
            "Keras model file": str(self.config.keras_model_file), 
            "Generator name": str(self.config.generator_name),
            "Seed folder": str(self.config.seed_folder),
            "Initial population folder": str(self.config.initial_population_folder), 
            "Feature combination": str(self.config.Feature_Combination),
            "Run time": str(self.config.RUNTIME),
            "Run id": str(self.config.run_id),
            "Rank prob": str(self.config.SELECTIONPROB),
            "Rank bias": str(self.config.RANK_BIAS),
            "Rank base": str(self.config.RANK_BASE),
            "Selection Operator": str(self.config.SELECTIONOP),
        }

        filedest = join(dir, "config.json")
        with open(filedest, 'w') as f:
            (json.dump(config, f, sort_keys=True, indent=4))


def main():
    from core.folders import folders
    log_dir_name = folders.experiments
    # Ensure the folder exists
    if os.path.exists(folders.log):
        shutil.rmtree(folders.log)
    Path(log_dir_name).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_to = f"{log_dir_name}/logs.txt"
    us.setup_logging(log_to)
    print("Logging results to " + log_to)

    Config.EXECTIME = 0

    config = cfg.BeamNGConfig()
    problem = BeamNGProblem.BeamNGProblem(config)
    
    map_E = MapElitesBNG(config.Feature_Combination, problem, log_dir_name, int(config.run_id), True)
    map_E.run()
    map_E.print_config(log_dir_name)


if __name__ == "__main__":
    main()
