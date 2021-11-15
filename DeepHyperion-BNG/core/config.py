from pathlib import Path
import os

class Config:
    GEN_RANDOM = 'GEN_RANDOM'
    GEN_RANDOM_SEEDED = 'GEN_RANDOM_SEEDED'
    GEN_SEQUENTIAL_SEEDED = 'GEN_SEQUENTIAL_SEEDED'
    GEN_DIVERSITY = 'GEN_DIVERSITY'

    SEG_LENGTH = 25
    NUM_SPLINE_NODES =10
    INITIAL_NODE = (0.0, 0.0, -28.0, 8.0)
    ROAD_BBOX_SIZE = (-1000, 0, 1000, 1500)
    EXECTIME = 0
    INVALID = 0

    def __init__(self):
        # try:
        self.BNG_HOME = os.environ['BNG_HOME']
        # except Error:
        #     self.BNG_HOME = f"{str(Path.home())}/Downloads/BeamNG.research.v1.7.0.1"

        print("Setting BNG_HOME to ", self.BNG_HOME)

        # try:
        # self.BNG_USER = os.environ['BNG_USER']
        # except Error:
        self.BNG_USER = f"{str(Path.home())}/Documents/BeamNG.research"

        print("Setting BNG_USER to ", self.BNG_USER)

        self.experiment_name = 'exp'
        self.fitness_weights = (-1.0,)

        self.POPSIZE = 2
        self.POOLSIZE = 3
        self.NUM_GENERATIONS = 2000000000

        self.ARCHIVE_THRESHOLD = 35.0

        self.RESEED_UPPER_BOUND = int(self.POPSIZE * 0.1)

        self.MUTATION_EXTENT = 6.0
        self.MUTPB = 0.7
        self.SELECTIONPROB = 0.5
        self.SELECTIONOP = "ranked"
        self.RANK_BIAS = 1.5 # 1 to 2
        self.RANK_BASE = "contribution_score"
        self.simulation_save = True
        self.simulation_name = 'beamng_nvidia_runner/sim_$(id)'
        self.keras_model_file = 'self-driving-car-178-2020.h5'
        # self.generator_name = Config.GEN_SEQUENTIAL_SEEDED
        # self.seed_folder = 'population_HQ1'
        self.generator_name = Config.GEN_DIVERSITY
        self.seed_folder = 'initial_pool'
        self.initial_population_folder = "initial_population"

        self.Feature_Combination = ["SegmentCount", "MeanLateralPosition"]
        # self.Feature_Combination = ["SegmentCount", "Curvature"]
        # self.Feature_Combination = ["SDSteeringAngle", "Curvature"]
        # self.Feature_Combination = ["SDSteeringAngle", "MeanLateralPosition"]
        # self.Feature_Combination = ["Curvature", "MeanLateralPosition"]
        # self.Feature_Combination = ["SegmentCount", "SDSteeringAngle"]
        
        self.RUNTIME =  360 # in seconds
        self.INTERVAL =  3600 # interval for temp reports
        
        self.run_id = 1 #int(os.environ['RUN_ID'])






