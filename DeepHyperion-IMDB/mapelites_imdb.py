import random
from pathlib import Path

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# local imports
from exploration import Exploration
from mapelites import MapElites
from feature_dimension import FeatureDimension
from features import count_neg, count_neg_relative, count_pos, count_pos_relative, count_words, count_adjs, count_verbs
from datasets import load_dataset

from text_input import Text
from individual import Individual
from properties import EXPECTED_LABEL, NGEN, INITIAL_POP, ORIGINAL_SEEDS,\
    POPSIZE, FEATURES
import utils
import properties


DATASET_DIR = "data"
test_ds = load_dataset('imdb', cache_dir=f"{DATASET_DIR}/imdb", split='test')
x_test, y_test = test_ds['text'], test_ds['label']


# Fetch the starting seeds from file
with open(ORIGINAL_SEEDS) as f:
    starting_seeds = f.read().split(',')[:-1]
    random.shuffle(starting_seeds)
    starting_seeds = starting_seeds[:POPSIZE]
    assert (len(starting_seeds) == POPSIZE)

class MapElitesIMDB(MapElites):

    def __init__(self, *args, **kwargs):
        super(MapElitesIMDB, self).__init__(*args, **kwargs)

    def map_x_to_b(self, x):
        """
        Map X solution to feature space dimensions
        :param x: individual
        :return b: tuple of indexes, cell of the map
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
        Apply the fitness function to individual x
        :param x: individual
        :return performance: fitness of x
        """
        # "calculate performance measure"    
        x.member.performance = x.evaluate()
        # TODO: collect all the inputs generated in this run
        Exploration.add_explored(x.member)
        return x.member.performance 

    def mutation(self, x):
        """
        Mutate the solution x
        :param x: individual to mutate
        :return x: mutated individual
        """
        # "apply mutation"
        Individual.COUNT += 1
        text1 = x.member.clone()
        ind = Individual(text1)
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
        if INITIAL_POP == 'random':
            # Choose randomly a file in the original dataset.
            seed = random.choice(starting_seeds)
            seed = int(seed)
            Individual.SEEDS.add(int(seed))
        elif INITIAL_POP == 'seeded':
            # Choose sequentially the inputs from the seed list.
            # NOTE: number of seeds should be no less than the initial population
            assert (len(starting_seeds) == POPSIZE)
            seed = starting_seeds[Individual.COUNT - 1]
            seed = int(seed)
            Individual.SEEDS.add(seed)

        x = x_test[seed]
        text1 = Text(x, EXPECTED_LABEL, seed)
        individual = Individual(text1)
        individual.member.seed = seed

        return individual

    def generate_feature_dimensions(self):
        fts = list()

        if "NegCount" in FEATURES:
            # feature 1: moves in svg path
            ft1 = FeatureDimension(name="NegCount", feature_simulator="count_neg", bins=1)
            fts.append(ft1)

        if "PosCount" in FEATURES:
            # feature 2: bitmaps
            ft2 = FeatureDimension(name="PosCount", feature_simulator="count_pos", bins=1)
            fts.append(ft2)

        if "WordCount" in FEATURES:
            # feature 3: orientation
            ft3 = FeatureDimension(name="WordCount", feature_simulator="count_words", bins=1)
            fts.append(ft3)

        if "RelNegCount" in FEATURES:
            # feature 1: moves in svg path
            ft4 = FeatureDimension(name="RelNegCount", feature_simulator="rel_count_neg", bins=1)
            fts.append(ft4)

        if "RelPosCount" in FEATURES:
            # feature 2: bitmaps
            ft5 = FeatureDimension(name="RelPosCount", feature_simulator="rel_count_pos", bins=1)
            fts.append(ft5)

        if "VerbCount" in FEATURES:
            # feature 1: moves in svg path
            ft6 = FeatureDimension(name="VerbCount", feature_simulator="count_verbs", bins=1)
            fts.append(ft6)

        if "AdjCount" in FEATURES:
            # feature 2: bitmaps
            ft7 = FeatureDimension(name="AdjCount", feature_simulator="count_adjs", bins=1)
            fts.append(ft7)

        return fts

    def feature_simulator(self, function, x):
        """
        Calculates the value of the desired feature
        :param function: name of the method to compute the feature value
        :param x: genotype of candidate solution x
        :return: feature value
        """
        if function == 'count_neg':
            return count_neg(x.member.text)
        if function == 'count_pos':
            return count_pos(x.member.text)
        if function == 'count_words':
            return count_words(x.member.text)
        if function == 'rel_count_pos':
            return count_pos_relative(x.member.text)
        if function == 'rel_count_neg':
            return count_neg_relative(x.member.text)
        if function == 'count_verbs':
            return count_verbs(x.member.text)
        if function == 'count_adjs':
            return count_adjs(x.member.text)

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

    map_E = MapElitesIMDB(NGEN, POPSIZE, log_dir_name, True)
    map_E.run()

    Individual.COUNT = 0

    print("Exporting inputs ...")
    for text in Exploration.all_inputs:
        text.export(all=True)
    
    print("Done")

    


if __name__ == "__main__":
    main()
