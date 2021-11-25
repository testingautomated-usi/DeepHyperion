
import operator
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
import json
import logging as log
import random
import math
import itertools 

# local imports
from self_driving.beamng_individual import Individual
from core.plot_utils import plot_heatmap
from self_driving.beamng_config import BeamNGConfig
from datetime import datetime
from core.seed_pool_access_strategy import SeedPoolAccessStrategy
from core.seed_pool_impl import SeedPoolFolder, SeedPoolRandom
from core.folders import folders
from self_driving.beamng_road_imagery import BeamNGRoadImagery
from self_driving.simulation_data import SimulationDataRecordProperties
from core.config import Config
import core.utils as us
from core.timer import Timer

class MapElites(ABC):

    def __init__(self, _type, problem, log_dir_path, run_id, minimization):
        """
        :param minimization: True if solving a minimization problem. False if solving a maximization problem.
        """
        self.problem = problem
        self.config: BeamNGConfig = problem.config
        if self.config.generator_name == self.config.GEN_RANDOM:
            seed_pool = SeedPoolRandom(self.problem, self.config.POPSIZE)
        else:
            seed_pool = SeedPoolFolder(self.problem, self.config.seed_folder)
        self._seed_pool_strategy = SeedPoolAccessStrategy(seed_pool)
        self.experiment_path = folders.experiments.joinpath(self.config.experiment_name)
        self.elapsed_time = 0
        self.log_dir_path = log_dir_path
        self.minimization = minimization
        # set the choice operator either to do a minimization or a maximization
        if self.minimization:
            self.place_operator = operator.lt
        else:
            self.place_operator = operator.ge

        self.ITERATIONs = self.config.NUM_GENERATIONS
        self.config.EXECTIME = 0
        self.random_solutions = self.config.POPSIZE
        self.feature_dimensions = self.generate_feature_dimensions(_type)

        # get number of bins for each feature dimension
        ft_bins = [ft.bins for ft in self.feature_dimensions]

        # Map of Elites: Initialize data structures to store solutions and fitness values
        self.solutions = dict()
        self.performances = dict()
        self.run_id = run_id
        for ft in self.feature_dimensions:
            log.info(f"Feature: {ft.name}")
        log.info("Configuration completed.")

    def generate_initial_population(self):
        """
        Bootstrap the algorithm by generating `self.bootstrap_individuals` individuals
        randomly sampled from a uniform distribution
        """
        log.info("Generate initial population")
        for i in range(0, self.random_solutions):
            # add solution to elites computing features and performance
            if self.config.generator_name == self.config.GEN_DIVERSITY:
                x = self.generate_random_solution_without_sim()
                x.seed = i
                self.place_in_mapelites_without_sim(x, None, 0)
            else:
                x = self.generate_random_solution_without_sim()
                x.seed = i
                self.place_in_mapelites_without_sim(x, None, 0)
                # x = self.generate_random_solution()
                # self.place_in_mapelites(x, None, 0)

    def run(self):
        """
        Main iteration loop of MAP-Elites
        """
        start_time = datetime.now()
        # start by creating an initial set of random solutions
        self.generate_initial_population()
        # iteration counter
        iter = 0
        # ranked selection probability is set to 0 at start
        filled = len(self.solutions)  
        sel_prob = self.config.SELECTIONPROB

        while Config.EXECTIME <= self.config.RUNTIME:  
            print(Config.EXECTIME)          
            # apply epsilon greedy
            selection_prob = random.uniform(0,1)
            if selection_prob <= sel_prob:
                # get the index of a random individual from the map of elites
                idx = self.ranked_selection(individuals=1)[0]
            else:
                idx = self.random_selection(individuals=1)[0]
            
            self.solutions[idx].m.selected_counter += 1  
            parent = self.solutions[idx]
            log.info(f"Iteration {iter}: Selecting individual {parent.m.name} at {parent.m.features} with rank: {parent.m.rank} and seed: {parent.seed}")  
            
            # mutate the individual
            ind = self.mutation(parent, parent.seed)
            # place the new individual in the map of elites
            self.place_in_mapelites(ind, idx, iter)

            if self.config.SELECTIONOP == "dynamic_ranked":
                # check coverage every 10000 iterations
                if iter % 1000 == 0:
                    last_filled = filled
                    filled = len(self.solutions) 
                    # if coverage doesn't change in last 1000 iterations
                    if filled == last_filled:
                        if sel_prob <= 1.0:
                            # increase ranked selection probability
                            sel_prob += 0.05
                            log.info(f"Ranked Selection probability increased to {sel_prob}")
            iter += 1

        end_time = datetime.now()
        self.elapsed_time = end_time - start_time
        self.extract_results(iter, self.elapsed_time)

        if self.minimization:
            idx = min(self.performances.items(), key=operator.itemgetter(1))[0]
        else:
            idx = max(self.performances.items(), key=operator.itemgetter(1))[0]

        
        best_perf = self.performances[idx]
        best_ind = self.solutions[idx]
        log.info(f"Best overall value: {best_perf}"
              f" produced by individual {best_ind}"
              f" and placed at {self.map_x_to_b(best_ind)}")

    def extract_results(self, i, execution_time):
        # now = datetime.now().strftime("%Y%m%d%H%M%S")
        log_dir_name = f"{self.log_dir_path}" #/{i}"
        log_dir_path = Path(f"{log_dir_name}/archive")
        log_dir_path.mkdir(parents=True, exist_ok=True)

         # filled values                                 
        filled = len(self.solutions)        

        original_seeds = set()
        mis_seeds = set()
        Individual.COUNT_MISS = 0
        for x in enumerate(self.performances.items()): 
            # enumerate function returns a tuple in the form
            # (index, (key, value)) it is a nested tuple
            # for accessing the value we do indexing x[1][1]
            original_seeds.add(self.solutions[x[1][0]].seed)
            if x[1][1] < 0:
                Individual.COUNT_MISS += 1
                mis_seeds.add(self.solutions[x[1][0]].seed)
                misbehavior = True
            else:
                misbehavior = False

            sim_folder = Path(f"{log_dir_path}/sim_{self.solutions[x[1][0]].m.name}_{x[1][0]}")
            sim_folder.mkdir(parents=True, exist_ok=True)

            # destination_sim = f"{log_dir_path}\\simulation_{self.solutions[x[1][0]].m.name}_{x[1][0]}.tsv"
            destination_sim_json = f"{sim_folder}\\simulation.full.json"
            destination_road = f"{sim_folder}\\road.json"


            with open(destination_sim_json, 'w') as f:
                f.write(json.dumps({
                    self.solutions[x[1][0]].m.simulation.f_params: self.solutions[x[1][0]].m.simulation.params._asdict(),
                    self.solutions[x[1][0]].m.simulation.f_info: self.solutions[x[1][0]].m.simulation.info.__dict__,
                    self.solutions[x[1][0]].m.simulation.f_road: self.solutions[x[1][0]].m.simulation.road.to_dict(),
                    self.solutions[x[1][0]].m.simulation.f_records: [r._asdict() for r in self.solutions[x[1][0]].m.simulation.states]
                }))

            # with open(destination_sim, 'w') as f:
            #     sep = '\t'
            #     f.write(sep.join(SimulationDataRecordProperties) + '\n')
            #     gen = (r._asdict() for r in self.solutions[x[1][0]].m.simulation.states)
            #     gen2 = (sep.join([str(d[key]) for key in SimulationDataRecordProperties]) + '\n' for d in gen)
            #     f.writelines(gen2)

            with open(destination_road, 'w') as f:
                road = {
                        'name': str(self.solutions[x[1][0]].m.name),
                        'seed': str(self.solutions[x[1][0]].seed),
                        'misbehaviour': misbehavior,
                        'performance': self.performances[x[1][0]],
                        'timestamp': str(self.solutions[x[1][0]].m.timestamp),
                        'elapsed': str(self.solutions[x[1][0]].m.elapsed),
                        'tool' : "DeepHyperion",
                        'run' : str(self.config.run_id),
                        'features': self.solutions[x[1][0]].m.features,
                        'rank': str(self.solutions[x[1][0]].m.rank),
                        'selected': str(self.solutions[x[1][0]].m.selected_counter),
                        'placed_mutant': str(self.solutions[x[1][0]].m.placed_mutant)

                }
                f.write(json.dumps(road))

            road_imagery = BeamNGRoadImagery.from_sample_nodes(self.solutions[x[1][0]].m.sample_nodes)
            image_path = sim_folder.joinpath(f"img_{self.solutions[x[1][0]].m.name}_{x[1][0]}")
            road_imagery.save(image_path.with_suffix('.jpg'))
            road_imagery.save(image_path.with_suffix('.svg'))

        feature_dict = dict()
        for ft in self.feature_dimensions:
            feature_dict.update({f"{ft.name}_min": ft.min,
            f"{ft.name}_max": ft.bins})

        _performances = {}

        # convert keys to string   
        for key, value in self.performances.items():
            _key = str(key)
            _performances[_key] = str(value)

        run_time = execution_time
        report = {
            "Run time": str(run_time),
            'Covered seeds': len(original_seeds),
            'Filled cells': (filled),
            'Misclassified seeds': len(mis_seeds),
            'Misclassifications': (Individual.COUNT_MISS),
            'Performances': _performances
        }

        report.update(feature_dict)
        
        dst = f"{log_dir_name}/report.json"
        with open(dst, 'w') as f:
            (json.dump(report, f, sort_keys=False, indent=4))


        # Find the order of features in tuples
        b = tuple()
        for ft in self.feature_dimensions:
            i = ft.name
            b = b + (i,)

        for feature1, feature2 in itertools.combinations(self.feature_dimensions, 2):         
            # # Create another folder insider the log one ...
            # log_dir_path = Path(f"{log_dir_name}/{feature1.name}_{feature2.name}")
            # log_dir_path.mkdir(parents=True, exist_ok=True)

            # Find the position of each feature in indexes
            x = list(b).index(feature1.name)
            y = list(b).index(feature2.name)

            # Define a new 2-D dict
            _solutions = {}
            _performances = {}
                
            for key, value in self.performances.items():
                _key = (key[x], key[y])
                if _key in _performances:
                    if _performances[_key] > value:
                        _performances[_key] = value
                        _solutions[_key] = self.solutions[key]
                else:
                    _performances[_key] = value
                    _solutions[_key] = self.solutions[key]

            # filled values                                 
            filled = len(_solutions)        

            original_seeds = set()
            mis_seeds = set()
            Individual.COUNT_MISS = 0
            for x in enumerate(_performances.items()): 
                # enumerate function returns a tuple in the form
                # (index, (key, value)) it is a nested tuple
                # for accessing the value we do indexing x[1][1]
                original_seeds.add(_solutions[x[1][0]].seed)
                if x[1][1] < 0:
                    Individual.COUNT_MISS += 1
                    mis_seeds.add(_solutions[x[1][0]].seed)
        
            feature_dict = {}
            feature_dict.update({f"{feature1.name}_min": feature1.min,
            f"{feature1.name}_max": feature1.bins})

            feature_dict.update({f"{feature2.name}_min": feature2.min,
            f"{feature2.name}_max": feature2.bins})

            str_performances = {}
            # convert keys to string   
            for key, value in _performances.items():
                str_performances[str(key)] = str(value)

            run_time = execution_time
            report = {
                "Run time": str(run_time),
                'Covered seeds': len(original_seeds),
                'Filled cells': (filled),
                'Misclassified seeds': len(mis_seeds),
                'Misclassifications': (Individual.COUNT_MISS),
                'Performances': str_performances
            }

            report.update(feature_dict)
            
            dst = f"{log_dir_name}/report_" + feature1.name + "_" + feature2.name + '.json'
            with open(dst, 'w') as f:
                (json.dump(report, f, sort_keys=False, indent=4))

            self.plot_map_of_elites(_performances, log_dir_name, feature1, feature2)       

    def place_in_mapelites(self, x, parent, iter):
        """
        Puts a solution inside the N-dimensional map of elites space.
        The following criteria is used:

        - Compute the feature descriptor of the solution to find the correct
                cell in the N-dimensional space
        - Compute the performance of the solution
        - Check if the cell is empty or if the previous performance is worse
            - Place new solution in the cell
        :param x: genotype of an individual
        """
        # performance of the x
        perf = self.performance_measure(x)

        # get coordinates in the feature space
        b = self.map_x_to_b(x)
        x.m.features = b

        for i in range(len(b)):
            # if the bin is not already present in the map
            if b[i] >= self.feature_dimensions[i].bins:
                self.feature_dimensions[i].bins = b[i] + 1

        status = None
        # place operator performs either minimization or maximization
        # compares value of x with value of the individual already in the bin
        if b in self.performances:
            if self.place_operator(perf, self.performances[b]):
                log.info(f"Iteration {iter}: Replacing individual {x.m.name} at {b} with perf: {perf} and seed: {x.seed}")
                
                if parent is not None:
                    self.solutions[parent].m.placed_mutant += 1
                self.performances[b] = perf
                self.solutions[b] = x
                status = "Replace"
            else:
                log.info(f"Iteration {iter}: Rejecting individual {x.m.name} at {b} with perf: {perf} in favor of {self.performances[b]}")
                status = "Reject"
        else:
            log.info(f"Iteration {iter}: Placing individual {x.m.name} at {b} with perf: {perf} and seed: {x.seed}")                    
            status = "Place"
            if parent is not None:
                self.solutions[parent].m.placed_mutant += 1
            self.performances[b] = perf
            self.solutions[b] = x

        if self.config.SELECTIONOP != "random":
            self.rank_in_mapelites(b, parent, status)

    def rank_in_mapelites(self, b, parent, status):
        if status == "Place" or status == "Replace":
            if self.config.RANK_BASE == "perf":
                self.solutions[b].m.rank = self.performances[b]
            elif self.config.RANK_BASE == "density":
                self.solutions[b].m.rank = us.compute_sparseness(self.solutions, self.solutions[b])
                neighbors = us.get_neighbors(b)
                # compute rank for all neighbors
                for neighbor in neighbors:
                    if neighbor in self.solutions:      
                        self.solutions[neighbor].m.rank = us.compute_sparseness(self.solutions, self.solutions[neighbor])
            else:
                if parent is not None and self.solutions[parent].m.selected_counter > 0:
                    self.solutions[parent].m.rank = self.solutions[parent].m.placed_mutant / self.solutions[parent].m.selected_counter
                self.solutions[b].m.rank = 1.0
        elif self.config.RANK_BASE == "contribution_score":
            if parent is not None and self.solutions[parent].m.selected_counter > 0:
                self.solutions[parent].m.rank = self.solutions[parent].m.placed_mutant / self.solutions[parent].m.selected_counter


    def random_selection(self, individuals=1):
        """
        Select an elite x from the current map of elites.
        The selection is done by selecting a random bin for each feature
        dimension, until a bin with a value is found.
        :param individuals: The number of individuals to randomly select
        :return: A list of N random elites
        """
        solutions = [value for key, value in self.solutions.items()]
        rand_ind = np.random.randint(0, len(solutions), 1)[0]
        idxs = list()
        idxs.append(solutions[rand_ind].m.features)

        return idxs

    def ranked_selection(self, individuals=1):

        def _get_sparse_index():
            """
            Get a  cell in the N-dimensional feature space which is more sparse
            :return: N-dimensional tuple of integers
            """
            if self.config.RANK_BASE == "perf":
                # select the individuals that are most likely to misclassified
                solutions = [value for key, value in self.solutions.items() if value.m.rank >= 0]
                solutions.sort(key=lambda x: x.m.rank, reverse=False)
            elif self.config.RANK_BASE == "density":
                solutions = [value for key, value in self.solutions.items()]
                solutions.sort(key=lambda x: x.m.rank, reverse=False)
            elif self.config.RANK_BASE == "contribution_score":
                solutions = [value for key, value in self.solutions.items()]
                solutions.sort(key=lambda x: x.m.rank, reverse=True)
            r = random.uniform(0, 1)
            d = self.config.RANK_BIAS - math.sqrt((self.config.RANK_BIAS * self.config.RANK_BIAS) - (4.0 * (self.config.RANK_BIAS - 1.0) * r))
            length = len(solutions)
            d = d / 2.0 / (self.config.RANK_BIAS - 1.0)      
            idx = int(length*d)
            return solutions[idx].m.features
        
        # individuals
        inds = list()
        idxs = list()
        for _ in range(0, individuals):
            idx = _get_sparse_index()
            idxs.append(idx)
            inds.append(self.solutions[idx])
        return idxs

    def place_in_mapelites_without_sim(self, x, parent, iter):
        """
        Puts a solution inside the N-dimensional map of elites space.
        The following criteria is used:
        - Compute the feature descriptor of the solution to find the correct
                cell in the N-dimensional space
        - Compute the performance of the solution
        - Check if the cell is empty or if the previous performance is worse
            - Place new solution in the cell
        :param x: genotype of an individual
        """
        # performance of the x
        border = x.m.distance_to_boundary
        x.oob_ff = border if border > 0 else -0.1
        perf = x.oob_ff
        # get coordinates in the feature space
        b = self.map_x_to_b(x)
        x.m.features = b

        for i in range(len(b)):
            # if the bin is not already present in the map
            if b[i] >= self.feature_dimensions[i].bins:
                self.feature_dimensions[i].bins = b[i] + 1

        status = None
        # place operator performs either minimization or maximization
        # compares value of x with value of the individual already in the bin
        if b in self.performances:
            if self.place_operator(perf, self.performances[b]):
                log.info(f"Iteration {iter}: Replacing individual {x.m.name} at {b} with perf: {perf} and seed: {x.seed}")
                
                if parent is not None:
                    self.solutions[parent].m.placed_mutant += 1
                self.performances[b] = perf
                self.solutions[b] = x
                status = "Replace"
            else:
                log.info(f"Iteration {iter}: Rejecting individual {x.m.name} at {b} with perf: {perf} in favor of {self.performances[b]}")
                status = "Reject"
        else:
            log.info(f"Iteration {iter}: Placing individual {x.m.name} at {b} with perf: {perf} and seed: {x.seed}")                    
            status = "Place"
            if parent is not None:
                self.solutions[parent].m.placed_mutant += 1
            self.performances[b] = perf
            self.solutions[b] = x

        if self.config.SELECTIONOP != "random":
            self.rank_in_mapelites(b, parent, status)

    def plot_map_of_elites(self, perfs, log_dir_name, feature1, feature2):
        """
        Plot a heatmap of elites
        """
        plot_heatmap(perfs,
                     feature1.name,
                     feature2.name,
                     savefig_path=log_dir_name
                     )

    @abstractmethod
    def performance_measure(self, x):
        """
        Function to evaluate solution x and give a performance measure
        :param x: genotype of a solution
        :return: performance measure of that solution
        """
        pass

    @abstractmethod
    def mutation(self, x, reference):
        """
        Function to mutate solution x and give a mutated solution
        :param x: genotype of a solution
        :return: mutated solution
        """
        pass

    @abstractmethod
    def map_x_to_b(self, x):
        """
        Function to map a solution x to feature space dimensions
        :param x: genotype of a solution
        :return: phenotype of the solution (tuple of indices of the N-dimensional space)
        """
        pass

    @abstractmethod
    def generate_random_solution(self):
        """
        Function to generate an initial random solution x
        :return: x, a random solution
        """
        pass

    @abstractmethod
    def generate_random_solution_without_sim(self):
        """
        Function to generate an initial random solution x
        :return: x, a random solution
        """
        pass

    @abstractmethod
    def generate_feature_dimensions(self):
        """
        Generate a list of FeatureDimension objects to define the feature dimension functions
        :return: List of FeatureDimension objects
        """
        pass
