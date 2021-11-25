import random
from os.path import join
from pathlib import Path
# For Python 3.6 we use the base keras
import keras
# local imports
from exploration import Exploration
from folder import Folder
from mapelites import MapElites
from feature_dimension import FeatureDimension
from features import move_distance, bitmap_count, orientation_calc
import vectorization_tools
from digit_input import Digit
from individual import Individual
from properties import NGEN, \
    POPSIZE, EXPECTED_LABEL, INITIALPOP, \
    ORIGINAL_SEEDS, BITMAP_THRESHOLD, FEATURES
import utils
import properties

# Load the dataset.
mnist = keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()

# Fetch the starting seeds from file
with open(ORIGINAL_SEEDS) as f:
    starting_seeds = f.read().split(',')[:-1]
    random.shuffle(starting_seeds)
    starting_seeds = starting_seeds[:POPSIZE]
    assert (len(starting_seeds) == POPSIZE)


def generate_digit(seed):
    seed_image = x_test[int(seed)]
    xml_desc = vectorization_tools.vectorize(seed_image)
    return Digit(xml_desc, EXPECTED_LABEL, seed)


class MapElitesMNIST(MapElites):

    def __init__(self, *args, **kwargs):
        super(MapElitesMNIST, self).__init__(*args, **kwargs)

    def map_x_to_b(self, x):
        """
        Map X solution to feature space dimensions
        :param x: individual
        :return b: tuple of indexes, cell of the map
        """
        b = tuple()
        for ft in self.feature_dimensions:
            i = ft.feature_descriptor(self, x)
            # if i < ft.min:
            #     ft.min = i
            b = b + (i,)
        return b

    def performance_measure(self, x):
        """
        Apply the fitness function to individual x
        :param x: individual
        :return performance: fitness of x
        """
        # "calculate performance measure"    
        performance = x.evaluate()
        # TODO: collect all the inputs generated in this run
        Exploration.add_explored(x.member)
        return performance

    def mutation(self, x):
        """
        Mutate the solution x
        :param x: individual to mutate
        :return x: mutated individual
        """
        # "apply mutation"
        Individual.COUNT += 1
        digit1 = x.member.clone()
        ind = Individual(digit1, x.seed)
        ind.mutate()
        return ind

    def generate_random_solution(self):
        """
        To ease the bootstrap of the algorithm, we can generate
        the first solutions in the feature space, so that we start
        filling the bins
        """
        # "Generate random solution"
        Individual.COUNT += 1
        if INITIALPOP == 'random':
            # Choose randomly a file in the original dataset.
            seed = random.choice(starting_seeds)
            Individual.SEEDS.add(seed)
        elif INITIALPOP == 'seeded':
            # Choose sequentially the inputs from the seed list.
            # NOTE: number of seeds should be no less than the initial population
            assert (len(starting_seeds) == POPSIZE)
            seed = starting_seeds[Individual.COUNT - 1]
            Individual.SEEDS.add(seed)

        digit1 = generate_digit(seed)
        individual = Individual(digit1, seed)
        individual.seed = seed

        return individual

    def generate_feature_dimensions(self):
        fts = list()

        if "Moves" in FEATURES:
            # feature 1: moves in svg path
            ft1 = FeatureDimension(name="Moves", feature_simulator="move_distance", bins=1)
            fts.append(ft1)

        if "Bitmaps" in FEATURES:
            # feature 2: bitmaps
            ft2 = FeatureDimension(name="Bitmaps", feature_simulator="bitmap_count", bins=1)
            fts.append(ft2)

        if "Orientation" in FEATURES:
            # feature 3: orientation
            ft3 = FeatureDimension(name="Orientation", feature_simulator="orientation_calc", bins=1)
            fts.append(ft3)

        return fts

    def feature_simulator(self, function, x):
        """
        Calculates the value of the desired feature
        :param function: name of the method to compute the feature value
        :param x: genotype of candidate solution x
        :return: feature value
        """
        if function == 'bitmap_count':
            return bitmap_count(x.member, BITMAP_THRESHOLD)
        if function == 'move_distance':
            return move_distance(x.member)
        if function == 'orientation_calc':
            return orientation_calc(x.member, 0)

def main():
    # Generate random folder to store result
    from folder import Folder
    log_dir_name = Folder.DST
    # Ensure the folder exists
    Path(log_dir_name).mkdir(parents=True, exist_ok=True)

    log_to = f"{log_dir_name}/logs.txt"
    debug = f"{log_dir_name}/debug.txt"

    # Setup logging
    utils.setup_logging(log_to, debug)
    print("Logging results to " + log_to)

    properties.to_json(Folder.DST)

    map_E = MapElitesMNIST(NGEN, POPSIZE, log_dir_name, True)
    map_E.run()

    Individual.COUNT = 0

    print("Exporting inputs ...")
    for digit in Exploration.all_inputs:
        digit.export(all=True)
    
    print("Done")

if __name__ == "__main__":
    main()
