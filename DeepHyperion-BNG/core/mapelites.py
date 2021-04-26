
import operator
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
import json
import logging as log

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
        self.elapsed_time = 0
        self.log_dir_path = log_dir_path
        self.minimization = minimization
        # set the choice operator either to do a minimization or a maximization
        if self.minimization:
            self.place_operator = operator.lt
        else:
            self.place_operator = operator.ge

        self.iterations = self.config.NUM_GENERATIONS
        self.config.EXECTIME = 0
        self.random_solutions = self.config.POPSIZE
        self.feature_dimensions = self.generate_feature_dimensions(_type)

        # get number of bins for each feature dimension
        ft_bins = [ft.bins for ft in self.feature_dimensions]

        # Map of Elites: Initialize data structures to store solutions and fitness values
        self.solutions = np.full(
            ft_bins, None,
            dtype=object
        )
        self.performances = np.full(ft_bins, np.inf, dtype=float)
        self.run_id = run_id
        log.info("Configuration completed.")

    def generate_initial_population(self):
        """
        Bootstrap the algorithm by generating `self.bootstrap_individuals` individuals
        randomly sampled from a uniform distribution
        """
        log.info("Generate initial population")
        for _ in range(0, self.random_solutions):
            # add solution to elites computing features and performance
            if self.config.generator_name == self.config.GEN_DIVERSITY:
                x = self.generate_random_solution_without_sim()
                self.place_in_mapelites_without_sim(x)
            else:
                x = self.generate_random_solution()
                self.place_in_mapelites(x)

    def run(self):
        """
        Main iteration loop of MAP-Elites
        """
        start_time = datetime.now()

        # start by creating an initial set of random solutions
        self.generate_initial_population()
        self.elapsed_time = datetime.now() - start_time
        self.extract_results()

        for i in range(0, self.iterations):
            if Config.EXECTIME <= self.config.RUNTIME:
                log.info(f"ITERATION {i}")
                log.info("Select and mutate.")
                # get the index of a random individual from the map of elites
                ind = self.random_selection(individuals=1)[0]
                # mutate the individual
                ind = self.mutation(ind, ind.seed)
                # place the new individual in the map of elites
                self.place_in_mapelites(ind)
            else:
                break

        if self.minimization:
            best = self.performances.argmin()
        else:
            best = self.performances.argmax()
        idx = np.unravel_index(best, self.performances.shape)
        best_perf = self.performances[idx]
        best_ind = self.solutions[idx]
        log.info(f"Best overall value: {best_perf}"
              f" produced by individual {best_ind}"
              f" and placed at {self.map_x_to_b(best_ind)}")

        end_time = datetime.now()
        self.elapsed_time = end_time - start_time

        self.extract_results()

    def extract_results(self):
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        log_dir_name = f"{self.log_dir_path}/log_{now}"
        log_dir_path = Path(f"{log_dir_name}/{self.feature_dimensions[1].name}_{self.feature_dimensions[0].name}")

        log_dir_path.mkdir(parents=True, exist_ok=True)
        solutions = self.solutions
        performances = self.performances
        # filled values
        total = np.size(solutions)

        filled = np.count_nonzero(performances != np.inf)

        original_seeds = set()
        mis_seeds = set()
        for (i, j), value in np.ndenumerate(solutions):
            if performances[i, j] != np.inf:
                original_seeds.add(solutions[i, j].seed)
                if performances[i, j] < 0:
                    mis_seeds.add(solutions[i, j].seed)

        Individual.COUNT_MISS = 0
        for (i, j), value in np.ndenumerate(performances):

            if performances[i, j] != np.inf:
                if performances[i, j] < 0:
                    Individual.COUNT_MISS += 1
                    misbehavior = True
                else:
                    misbehavior = False

                destination_sim = f"{log_dir_path}\\simulation_{solutions[i,j].m.name}_{i,j}.tsv"
                destination_sim_json = f"{log_dir_path}\\simulation_{solutions[i,j].m.name}_{i,j}.json"
                destination_road = f"{log_dir_path}\\road_{solutions[i,j].m.name}_{i,j}.json"

                with open(destination_sim_json, 'w') as f:
                    f.write(json.dumps({
                        solutions[i, j].m.simulation.f_params: solutions[i, j].m.simulation.params._asdict(),
                        solutions[i, j].m.simulation.f_info: solutions[i, j].m.simulation.info.__dict__,
                        solutions[i, j].m.simulation.f_road: solutions[i, j].m.simulation.road.to_dict(),
                        solutions[i, j].m.simulation.f_records: [r._asdict() for r in solutions[i, j].m.simulation.states]
                    }))

                with open(destination_sim, 'w') as f:
                    sep = '\t'
                    f.write(sep.join(SimulationDataRecordProperties) + '\n')
                    gen = (r._asdict() for r in solutions[i,j].m.simulation.states)
                    gen2 = (sep.join([str(d[key]) for key in SimulationDataRecordProperties]) + '\n' for d in gen)
                    f.writelines(gen2)

                with open(destination_road, 'w') as f:
                    road = {
                            "road": solutions[i, j].m.to_dict(),
                            "features": [
                                self.feature_dimensions[1].name,
                                self.feature_dimensions[0].name
                            ],
                            "misbehaviour": misbehavior,
                            "performance": performances[i, j],
                            "tool": "DeepHyperion",
                            "run": self.run_id,
                            'timestamp': str(datetime.now()),
                            'elapsed': str(self.elapsed_time),
                            self.feature_dimensions[1].name: j,
                            self.feature_dimensions[0].name: i

                    }
                    f.write(json.dumps(road))

                road_imagery = BeamNGRoadImagery.from_sample_nodes(solutions[i, j].m.sample_nodes)
                image_path = log_dir_path.joinpath(f"img_{solutions[i,j].m.name}_{i,j}")
                road_imagery.save(image_path.with_suffix('.jpg'))
                road_imagery.save(image_path.with_suffix('.svg'))

        run_time = self.get_elapsed_time()
        miss_density = 0
        if filled > 0:
            miss_density = Individual.COUNT_MISS / filled
        report = {
            "Run time": str(run_time),
            "Simulation time": Config.EXECTIME,
            f"{self.feature_dimensions[1].name}_min": self.feature_dimensions[1].min,
            f"{self.feature_dimensions[1].name}_max": self.feature_dimensions[1].bins,
            f"{self.feature_dimensions[0].name}_min": self.feature_dimensions[0].min,
            f"{self.feature_dimensions[0].name}_max": self.feature_dimensions[0].bins,
            'Covered seeds': len(original_seeds),
            'Filled cells': str(filled),
            'Filled density': str(filled / total),
            'Misbehaviour seeds': len(mis_seeds),
            'Misbehaviour': str(Individual.COUNT_MISS),
            'Misbehaviour density': str(miss_density),
            'Performances': self.performances.tolist()
        }
        dst = f"{log_dir_name}/report_" + self.feature_dimensions[1].name + "_" + self.feature_dimensions[
            0].name + '.json'
        report_string = json.dumps(report)

        file = open(dst, 'w')
        file.write(report_string)
        file.close()

        self.plot_map_of_elites(performances, f"{log_dir_name}")

    def place_in_mapelites(self, x):
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

        reconstruct = False
        for i in range(len(b)):
            if b[i] >= self.feature_dimensions[i].bins:
                reconstruct = True
                self.feature_dimensions[i].bins = b[i] + 1

        if reconstruct:
            self.reconstruct_map()

        # place operator performs either minimization or maximization
        if self.place_operator(perf, self.performances[b]):
            log.info(f"PLACE: Placing individual {x} at {b} with perf: {perf}")
            self.performances[b] = perf
            self.solutions[b] = x
        else:
            log.info(f"PLACE: Individual {x} rejected at {b} with perf: {perf} in favor of {self.performances[b]}")

    def place_in_mapelites_without_sim(self, x):
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

        reconstruct = False
        for i in range(len(b)):
            if b[i] >= self.feature_dimensions[i].bins:
                reconstruct = True
                self.feature_dimensions[i].bins = b[i] + 1

        if reconstruct:
            self.reconstruct_map()

        # place operator performs either minimization or maximization
        if self.place_operator(perf, self.performances[b]):
            log.info(f"PLACE: Placing individual {x} at {b} with perf: {perf}")
            self.performances[b] = perf
            self.solutions[b] = x
        else:
            log.info(f"PLACE: Individual {x} rejected at {b} with perf: {perf} in favor of {self.performances[b]}")

    def random_selection(self, individuals=1):
        """
        Select an elite x from the current map of elites.
        The selection is done by selecting a random bin for each feature
        dimension, until a bin with a value is found.
        :param individuals: The number of individuals to randomly select
        :return: A list of N random elites
        """

        def _get_random_index():
            """
            Get a random cell in the N-dimensional feature space
            :return: N-dimensional tuple of integers
            """
            indexes = tuple()
            for ft in self.feature_dimensions:
                rnd_ind = np.random.randint(0, ft.bins, 1)[0]
                indexes = indexes + (rnd_ind,)
            return indexes

        def _is_not_initialized(index):
            """
            Checks if the selected index points to a None solution (not yet initialized)            
            :return: Boolean
            """
            if self.solutions[index] is None:
                return True
            return False

        # individuals
        inds = list()
        idxs = list()
        for _ in range(0, individuals):
            idx = _get_random_index()
            # we do not want to repeat entries
            while idx in idxs or _is_not_initialized(idx):
                idx = _get_random_index()
            idxs.append(idx)
            inds.append(self.solutions[idx])
        return inds

    def get_elapsed_time(self):
        return self.elapsed_time

    def reconstruct_map(self):
        """
        Extend Map of elites dynamically if needed
        """
        # get number of bins for each feature dimension
        ft_bins = [ft.bins for ft in self.feature_dimensions]

        new_solutions = np.full(
            ft_bins, None,
            dtype=object
        )
        new_performances = np.full(ft_bins, np.inf, dtype=float)

        new_solutions[0:self.solutions.shape[0], 0:self.solutions.shape[1]] = self.solutions
        new_performances[0:self.performances.shape[0], 0:self.performances.shape[1]] = self.performances
        self.solutions = new_solutions
        self.performances = new_performances
        return

    def plot_map_of_elites(self, perfs, path):
        """
        Plot a heatmap of elites
        """
        plot_heatmap(perfs,
                     self.feature_dimensions[1],
                     self.feature_dimensions[0],
                     savefig_path=path
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
