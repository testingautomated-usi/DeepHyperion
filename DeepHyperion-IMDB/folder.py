from timer import Timer
from os.path import exists, join
from os import makedirs
from datetime import datetime
import random
from properties import NAME, FEATURES

class Folder:
    rand = random.randint(0,1000)
    now = datetime.now().strftime("%Y%m%d%H%M%S") + f"_{rand}"

    if NAME:
        DST = f"logs/{FEATURES[0]}_{FEATURES[1]}/"+ NAME +"_" + now
    else:
        DST = f"logs/{FEATURES[0]}_{FEATURES[1]}/"+ now

    DST_ALL = join(DST, "all")

    if not exists(DST_ALL) and NAME != "":
        makedirs(DST_ALL)

    DST_ARC = join(DST, "archive")

    if not exists(DST_ARC) and NAME != "":
        makedirs(DST_ARC)

