import operator
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path
import json
import math
import logging as log


# local imports
from individual import Individual
from plot_utils import plot_heatmap
from text_input import Text
import utils
from utils import compute_sparseness, get_neighbors
from properties import RANK_BIAS, SELECTIONPROB, SELECTIONOP, RANK_BASE
from timer import Timer
import random
import itertools   



class MapElites(ABC):

    def __init__(self, iterations, bootstrap_individuals, log_dir_path, minimization):
        """
        :param iterations: Number of evolutionary iterations        
        :param bootstrap_individuals: Number of individuals randomly generated to bootstrap the algorithm       
        :param minimization: True if solving a minimization problem. False if solving a maximization problem.
        """
        self.elapsed_time = 0
        self.log_dir_path = log_dir_path
        self.minimization = minimization
        # set the choice operator either to do a minimization or a maximization
        if self.minimization:
            self.place_operator = operator.lt
        else:
            self.place_operator = operator.ge

        self.iterations = iterations
        
        self.random_solutions = bootstrap_individuals
        # self.starting_seeds = generate_diverse_initial_population()
        self.feature_dimensions = self.generate_feature_dimensions()

        # get number of bins for each feature dimension
        # ft_bins = [ft.bins for ft in self.feature_dimensions]

        # Map of Elites: Initialize data structures to store solutions and fitness values
        self.solutions = dict()
        self.performances = dict()

        log.info("Configuration completed.")

    def generate_initial_population(self):
        """
        Bootstrap the algorithm by generating `self.bootstrap_individuals` individuals
        randomly sampled from a uniform distribution
        """
        log.info("Generate initial population")
        for _ in range(0, self.random_solutions):
            x = self.generate_random_solution()
            # add solution to elites computing features and performance
            self.place_in_mapelites(x, None, 0)
    



    def run(self):
        """
        Main iteration loop of MAP-Elites
        """
        # start by creating an initial set of random solutions
        self.generate_initial_population()
        # self.generate_training_population()
        # iteration counter
        iter = 0
        # ranked selection probability is set to 0 at start
        filled = len(self.solutions)  
        sel_prob = SELECTIONPROB
        # self.elapsed_time = Timer.get_elapsed_time()
        # self.extract_results(iter, self.elapsed_time)
        while Timer.has_budget():# iter <= 400: #            
            # apply epsilon greedy
            selection_prob = random.uniform(0,1)
            if selection_prob <= sel_prob:
                # get the index of a random individual from the map of elites
                idx = self.rank_selection(individuals=1)[0]
            else:
                idx = self.random_selection(individuals=1)[0]
            
            self.solutions[idx].member.selected_counter += 1  
            parent = self.solutions[idx]
            log.info(f"Iteration {iter}: Selecting individual {parent.member.id} at {parent.features} with rank: {parent.member.rank} and seed: {parent.member.seed}")  
            
            # mutate the individual
            ind = self.mutation(parent)
            # place the new individual in the map of elites
            self.place_in_mapelites(ind, idx, iter)
            self.elapsed_time = Timer.get_elapsed_time()

            if SELECTIONOP == "dynamic_ranked":
                # check coverage every 10000 iterations
                if iter % 1000 == 0:
                    last_filled = filled
                    filled = len(self.solutions) 
                    # if coverage doesn't change in last 1000 iterations
                    if filled == last_filled:
                        if sel_prob <= 1.0:
                            # increase ranked selection probability
                            sel_prob += 0.05
                            log.info(f"Rank Selection probability increased to {sel_prob}")
            iter += 1

        self.elapsed_time = Timer.get_elapsed_time()
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

    def extract_results(self, iterations, execution_time):
        # self.log_dir_path is "logs/temp_..."
        log_dir_name = f"{self.log_dir_path}"#/log_{iterations}_{execution_time}"
        log_dir_path = Path(f"{log_dir_name}")
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
            original_seeds.add(self.solutions[x[1][0]].member.seed)
            if self.solutions[x[1][0]].member.is_misbehavior():
                Individual.COUNT_MISS += 1
                mis_seeds.add(self.solutions[x[1][0]].member.seed)
            self.solutions[x[1][0]].member.export()
    
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
                original_seeds.add(_solutions[x[1][0]].member.seed)
                if _solutions[x[1][0]].member.is_misbehavior():
                    Individual.COUNT_MISS += 1
                    mis_seeds.add(_solutions[x[1][0]].member.seed)
        
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
        # get coordinates in the feature space
        b = self.map_x_to_b(x)
        x.features = b
        # performance of the x
        perf = self.performance_measure(x)
        for i in range(len(b)):
            # if the bin is not already present in the map
            if b[i] >= self.feature_dimensions[i].bins:
                self.feature_dimensions[i].bins = b[i] + 1

        status = None
        # place operator performs either minimization or maximization
        # compares value of x with value of the individual already in the bin
        if b in self.performances:
            if self.place_operator(perf, self.performances[b]):
                log.info(f"Iteration {iter}: Replacing individual {x.member.id} at {b} with perf: {perf} and seed: {x.member.seed}")
                
                if parent is not None:
                    self.solutions[parent].member.placed_mutant += 1
                self.performances[b] = perf
                self.solutions[b] = x
                status = "Replace"
            else:
                log.info(f"Iteration {iter}: Rejecting individual {x.member.id} at {b} with perf: {perf} in favor of {self.performances[b]}")
                status = "Reject"
        else:

            log.info(f"Iteration {iter}: Placing individual {x.member.id} at {b} with perf: {perf} and seed: {x.member.seed}")                    
            status = "Place"
            if parent is not None:
                self.solutions[parent].member.placed_mutant += 1
            self.performances[b] = perf
            self.solutions[b] = x

        if SELECTIONOP != "random":
            self.rank_in_mapelites(b, parent, status)

    def rank_in_mapelites(self, b, parent, status):
        if status == "Place" or status == "Replace":
            if RANK_BASE == "perf":
                self.solutions[b].member.rank = self.performances[b]
            elif RANK_BASE == "density":
                self.solutions[b].member.rank = compute_sparseness(self.solutions, self.solutions[b])
                neighbors = get_neighbors(b)
                # compute rank for all neighbors
                for neighbor in neighbors:
                    if neighbor in self.solutions:      
                        self.solutions[neighbor].member.rank = compute_sparseness(self.solutions, self.solutions[neighbor])
            else:
                if parent is not None and self.solutions[parent].member.selected_counter > 0:
                    self.solutions[parent].member.rank = self.solutions[parent].member.placed_mutant / self.solutions[parent].member.selected_counter
                self.solutions[b].member.rank = 1.0
        elif RANK_BASE == "contribution_score":
            if parent is not None and self.solutions[parent].member.selected_counter > 0:
                self.solutions[parent].member.rank = self.solutions[parent].member.placed_mutant / self.solutions[parent].member.selected_counter



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
        idxs.append(solutions[rand_ind].features)

        return idxs

    def rank_selection(self, individuals=1):

        def _get_sparse_index():
            """
            Get a  cell in the N-dimensional feature space which is more sparse
            :return: N-dimensional tuple of integers
            """
            if RANK_BASE == "perf":
                # select the individuals that are most likely to misclassified
                solutions = [value for key, value in self.solutions.items() if value.member.rank >= 0]
                solutions.sort(key=lambda x: x.member.rank, reverse=False)
            elif RANK_BASE == "density":
                solutions = [value for key, value in self.solutions.items()]
                solutions.sort(key=lambda x: x.member.rank, reverse=False)
            elif RANK_BASE == "contribution_score":
                solutions = [value for key, value in self.solutions.items()]
                solutions.sort(key=lambda x: x.member.rank, reverse=True)
            r = random.uniform(0, 1)
            d = RANK_BIAS - math.sqrt((RANK_BIAS * RANK_BIAS) - (4.0 * (RANK_BIAS - 1.0) * r))
            length = len(solutions)
            d = d / 2.0 / (RANK_BIAS - 1.0)      
            idx = int(length*d)
            return solutions[idx].features
        
        # individuals
        inds = list()
        idxs = list()
        for _ in range(0, individuals):
            idx = _get_sparse_index()
            idxs.append(idx)
            inds.append(self.solutions[idx])
        return idxs

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
    def mutation(self, x):
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
    def generate_feature_dimensions(self):
        """
        Generate a list of FeatureDimension objects to define the feature dimension functions
        :return: List of FeatureDimension objects
        """
        pass
