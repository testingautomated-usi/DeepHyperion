# Make sure that any of this properties can be overridden using env.properties
from lib2to3.pgen2.token import NAME
from multiprocessing.pool import RUN
import os
from os.path import join
import json


# GA Setup
POPSIZE          = 1000
NGEN             = 50000

RUNTIME          = 3600

MODEL            = "models/text_classifier.h5"

FEATURES        = ["PosCount", "NegCount"] #PosCount NegCount VerbCount
NUM_CELLS       = 25
RUN             = 1
NAME            = f"RUN_{RUN}_{POPSIZE}_{FEATURES[0]}-{FEATURES[1]}_{RUNTIME}"

EXPECTED_LABEL  = 1 # 0 or 1
MUTLOWERBOUND    = 0.01
MUTUPPERBOUND    = 0.6

SELECTIONOP    = 'ranked' # random or ranked or dynamic_ranked
SELECTIONPROB    = 0.5
RANK_BIAS    = 1.5 # value between 1 and 2
RANK_BASE    = 'contribution_score' # perf or density or contribution_score

INITIAL_POP = 'seeded'

ORIGINAL_SEEDS = "starting_seeds_pos.txt"


INPUT_MAXLEN = 2000

def to_json(folder):
    config = {
        'popsize': str(POPSIZE),
        'model': str(MODEL),
        'runtime': str(RUNTIME),
        'features': str(FEATURES),
        'mut low': str(MUTLOWERBOUND),
        'mut up': str(MUTUPPERBOUND),
        'ranked prob': str(SELECTIONPROB),
        'rank bias' : str(RANK_BIAS),
        'rank base' : str(RANK_BASE),
        'selection': str(SELECTIONOP),
        'expected label': str(EXPECTED_LABEL)
    }
    filedest = join(folder, "config.json")
    with open(filedest, 'w') as f:
        (json.dump(config, f, sort_keys=True, indent=4))
