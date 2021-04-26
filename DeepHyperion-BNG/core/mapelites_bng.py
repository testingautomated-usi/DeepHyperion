import sys
import os
import json
import time
from pathlib import Path
import numpy as np
from datetime import datetime
import logging as log
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

        simulation_data.params = SimulationParams(beamng_steps=data["params"]["beamng_steps"], delay_msec=int(data["params"]["delay_msec"]))

        simulation_data.road = DecalRoad.from_dict(data["road"])
        simulation_data.info = SimulationInfo()
        simulation_data.info.start_time = data["info"]["start_time"]
        simulation_data.info.end_time = data["info"]["end_time"]
        simulation_data.info.success = data["info"]["success"]
        simulation_data.info.computer_name = data["info"]["computer_name"]
        simulation_data.info.id = data["info"]["id"]

        simulation_data.states = states
        if len(states) > 0:
            member.distance_to_boundary = simulation_data.min_oob_distance()
            member.simulation = simulation_data
            individual: BeamNGIndividual = BeamNGIndividual(member, self.config)
        else:
            print("*********Bug************")
        return individual

    def generate_feature_dimensions(self, combination):
        fts = list()

        if "MinRadius" in combination and "DirectionCoverage" in combination:
            ft1 = FeatureDimension(name="MinRadius", feature_simulator="min_radius", bins=1)
            fts.append(ft1)
            ft2 = FeatureDimension(name="DirectionCoverage", feature_simulator="dir_coverage", bins=1)
            fts.append(ft2)

        elif "MinRadius" in combination and "MeanLateralPosition" in combination:
            ft1 = FeatureDimension(name="MinRadius", feature_simulator="min_radius", bins=1)
            fts.append(ft1)

            ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
            fts.append(ft3)

        elif "DirectionCoverage" in combination and "MeanLateralPosition" in combination:
            ft2 = FeatureDimension(name="DirectionCoverage", feature_simulator="dir_coverage", bins=1)
            fts.append(ft2)

            ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
            fts.append(ft3)

        elif "MinRadius" in combination and "SegmentCount" in combination:
            ft1 = FeatureDimension(name="MinRadius", feature_simulator="min_radius", bins=1)
            fts.append(ft1)
            ft2 = FeatureDimension(name="SegmentCount", feature_simulator="segment_count", bins=1)
            fts.append(ft2)

        elif "SegmentCount" in combination and "MeanLateralPosition" in combination:
            ft1 = FeatureDimension(name="SegmentCount", feature_simulator="segment_count", bins=1)
            fts.append(ft1)

            ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
            fts.append(ft3)

        elif "SegmentCount" in combination and "DirectionCoverage" in combination:
            ft2 = FeatureDimension(name="DirectionCoverage", feature_simulator="dir_coverage", bins=1)
            fts.append(ft2)

            ft3 = FeatureDimension(name="SegmentCount", feature_simulator="segment_count", bins=1)
            fts.append(ft3)

        elif "SegmentCount" in combination and "SDSteeringAngle" in combination:
            ft2 = FeatureDimension(name="SDSteeringAngle", feature_simulator="sd_steering", bins=1)
            fts.append(ft2)

            ft3 = FeatureDimension(name="SegmentCount", feature_simulator="segment_count", bins=1)
            fts.append(ft3)
        
        elif "DirectionCoverage" in combination and "SDSteeringAngle" in combination:
            ft2 = FeatureDimension(name="SDSteeringAngle", feature_simulator="sd_steering", bins=1)
            fts.append(ft2)

            ft3 = FeatureDimension(name="DirectionCoverage", feature_simulator="dir_coverage", bins=1)
            fts.append(ft3)

        elif "MinRadius" in combination and "SDSteeringAngle" in combination:
            ft2 = FeatureDimension(name="SDSteeringAngle", feature_simulator="sd_steering", bins=1)
            fts.append(ft2)

            ft3 = FeatureDimension(name="MinRadius", feature_simulator="min_radius", bins=1)
            fts.append(ft3)
        
        elif "SDSteeringAngle" in combination and "MeanLateralPosition" in combination:
            ft2 = FeatureDimension(name="SDSteeringAngle", feature_simulator="sd_steering", bins=1)
            fts.append(ft2)

            ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
            fts.append(ft3)

        elif "Curvature" in combination and "SDSteeringAngle" in combination:
            ft2 = FeatureDimension(name="SDSteeringAngle", feature_simulator="sd_steering", bins=1)
            fts.append(ft2)

            ft3 = FeatureDimension(name="Curvature", feature_simulator="curvature", bins=1)
            fts.append(ft3)

        elif "Curvature" in combination and "SegmentCount" in combination:
            ft2 = FeatureDimension(name="SegmentCount", feature_simulator="segment_count", bins=1)
            fts.append(ft2)

            ft3 = FeatureDimension(name="Curvature", feature_simulator="curvature", bins=1)
            fts.append(ft3)

        elif "Curvature" in combination and "MeanLateralPosition" in combination:
            ft2 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
            fts.append(ft2)

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

    @staticmethod
    def print_config(dir):
        config = {
            "MUTATION_EXTENT": str(self.config.MUTATION_EXTENT),
            "MUTPB": str(self.config.MUTPB),
            "keras_model_file": str(self.config.keras_model_file), 
            "generator_name": str(self.config.generator_name),
            "seed_folder": str(self.config.seed_folder),
            "initial_population_folder": str(self.config.initial_population_generator), 
            "Feature_Combination": str(self.configg.Feature_Combination),
            "RUNTIME": str(self.config.RUNTIME),
            "RUN": str(self.config.run_id) 

        }

        filedest = join(dir, "config.json")
        with open(filedest, 'w') as f:
            (json.dump(config, f, sort_keys=True, indent=4))


def main():
    
    from folders import folders
    log_dir_name = folders.experiments
    # Ensure the folder exists
    Path(log_dir_name).mkdir(parents=True, exist_ok=True)

    config = cfg.BeamNGConfig()
    problem = BeamNGProblem.BeamNGProblem(config)
    

    log_to = f"{log_dir_name}/logs.txt"
    debug = f"{log_dir_name}/debug.txt"

    # Setup logging
    us.setup_logging(log_to, debug)
    print("Logging results to " + log_to)

    Config.EXECTIME = 0
    map_E = MapElitesBNG(config.Feature_Combination, problem, log_dir_name, int(config.run_id), True)
    map_E.run()
    log.info(f"invalid mutation: {Config.INVALID}")


if __name__ == "__main__":
    main()
