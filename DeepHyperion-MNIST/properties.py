# Make sure that any of this properties can be overridden using env.properties
import os
from os.path import join
import json
import uuid

class MissingEnvironmentVariable(Exception):
    pass

# GA Setup
POPSIZE          = int(os.getenv('DH_POPSIZE', '800'))
NGEN             = int(os.getenv('DH_NGEN', '500000'))

RUNTIME          = int(os.getenv('DH_RUNTIME', '36'))
INTERVAL         = int(os.getenv('DH_INTERVAL', '900'))

# Mutation Hyperparameters
# range of the mutation
MUTLOWERBOUND    = float(os.getenv('DH_MUTLOWERBOUND', '0.01'))
MUTUPPERBOUND    = float(os.getenv('DH_MUTUPPERBOUND', '0.6'))

SELECTIONOP    = str(os.getenv('DH_SELECTIONOP', 'random')) # random or ranked or dynamic_ranked
SELECTIONPROB    = float(os.getenv('DH_SELECTIONPROB', '0.0'))
RANK_BIAS    = float(os.getenv('DH_RANK_BIAS', '1.5')) # value between 1 and 2
RANK_BASE    = str(os.getenv('DH_RANK_BASE', 'contribution_score')) # perf or density or contribution_score

# Dataset
EXPECTED_LABEL   = int(os.getenv('DH_EXPECTED_LABEL', '5'))

#------- NOT TUNING ----------

# mutation operator probability
MUTOPPROB        = float(os.getenv('DH_MUTOPPROB', '0.5'))
MUTOFPROB        = float(os.getenv('DH_MUTOFPROB', '0.5'))


IMG_SIZE         = int(os.getenv('DH_IMG_SIZE', '28'))
num_classes      = int(os.getenv('DH_NUM_CLASSES', '10'))

INITIALPOP       = os.getenv('DH_INITIALPOP', 'seeded')

MODEL            = os.getenv('DH_MODEL', 'models/model_mnist.h5')

ORIGINAL_SEEDS   = os.getenv('DH_ORIGINAL_SEEDS', 'bootstraps_five')

BITMAP_THRESHOLD = float(os.getenv('DH_BITMAP_THRESHOLD', '0.5'))

DISTANCE_SEED         = float(os.getenv('DH_DISTANCE_SEED', '5.0'))
DISTANCE         = float(os.getenv('DH_DISTANCE', '2.0'))

FEATURES             = os.getenv('FEATURES', ["Bitmaps", "Moves"])
NUM_CELLS           = int(os.getenv("NUM_CELLS", '25'))
# FEATURES             = os.getenv('FEATURES', ["Orientation","Bitmaps"])
# FEATURES             = os.getenv('FEATURES', ["Orientation","Moves"])

TSHD_TYPE             = os.getenv('TSHD_TYPE', '1') # 1: threshold on vectorized-rasterized seed, use DISTANCE = 2

# # TODO Mayber there's a better way to handle this
try:
    NAME = str(os.environ['NAME'])
except Exception:
    NAME = None

try:
    THE_HASH = str(os.environ['THE_HASH'])
except Exception:
    THE_HASH = str(uuid.uuid4().hex)
    print("Generate random Hash", str(THE_HASH))

try:
    RUN = int(os.environ['RUN_ID'])
except KeyError:
    raise MissingEnvironmentVariable("RUN_ID does not exist. Please specify a value for this ENV variable")
except Exception:
    raise MissingEnvironmentVariable("Some other error?")

try:
    FEATURES = str(os.environ['FEATURES'])
    FEATURES = FEATURES.split(',')
except KeyError:
    raise MissingEnvironmentVariable("FEATURES does not exist. Please specify a value for this ENV variable")
except Exception:
    raise MissingEnvironmentVariable("Some other error?")


def to_json(folder):
    if TSHD_TYPE == '0':
        tshd_val = None
    elif TSHD_TYPE == '1':
        tshd_val = str(DISTANCE)
    elif TSHD_TYPE == '2':
        tshd_val = str(DISTANCE_SEED)

    config = {
        'name': str(NAME),
        'hash': str(THE_HASH),
        'popsize': str(POPSIZE),
        'initial pop': str(INITIALPOP),
        'label': str(EXPECTED_LABEL),
        'mut low': str(MUTLOWERBOUND),
        'mut up': str(MUTUPPERBOUND),
        'model': str(MODEL),
        'runtime': str(RUNTIME),
        'run': str(RUN),
        'features': str(FEATURES),
        'tshd_type': str(TSHD_TYPE),
        'tshd_value': str(tshd_val),
        'ranked prob': str(SELECTIONPROB),
        'rank bias' : str(RANK_BIAS),
        'rank base' : str(RANK_BASE),
        'selection': str(SELECTIONOP),

    }
    filedest = join(folder, "config.json")
    with open(filedest, 'w') as f:
        (json.dump(config, f, sort_keys=True, indent=4))
