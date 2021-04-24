from timer import Timer
from os.path import exists, join
from os import makedirs
from datetime import datetime

class Folder:
    DST = "logs/run_"+ datetime.now().strftime("%Y%m%d%H%M%S")
    DST_ALL = join(DST, "all")

    if not exists(DST_ALL):
        makedirs(DST_ALL)

    DST_ARC = join(DST, "archive")

    if not exists(DST_ARC):
        makedirs(DST_ARC)
