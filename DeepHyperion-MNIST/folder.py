from timer import Timer
from os.path import exists, join
from os import makedirs
from datetime import datetime
import random
from properties import NAME

class Folder:
    rand = random.randint(0,1000)
    now = datetime.now().strftime("%Y%m%d%H%M%S") + f"_{rand}"

    if NAME:
        DST = "run_"+ NAME +"_" + now
    else:
        DST = "run_"+ now

    DST_ALL = join(DST, "all")

    if not exists(DST_ALL) and NAME != "":
        makedirs(DST_ALL)

    DST_ARC = join(DST, "archive")

    if not exists(DST_ARC) and NAME != "":
        makedirs(DST_ARC)

