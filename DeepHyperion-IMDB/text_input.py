
from timer import Timer
from properties import FEATURES, RUN
from folder import Folder
import json
from os.path import join
from tensorflow import keras
import numpy as np
from utils import decode_imdb_reviews

class Text:
    COUNT = 0

    def __init__(self, text, label, seed):
        self.timestamp, self.elapsed_time = Timer.get_timestamps()
        self.id = Text.COUNT
        self.run = RUN
        self.text = text
        self.expected_label = label
        self.predicted_label = None
        self.features = FEATURES
        self.tool = "DeepHyperion"
        self.rank = np.inf
        self.selected_counter = 0
        self.placed_mutant = 0
        self.seed = seed
        self.confidence = None
        self.performance = None
        Text.COUNT += 1


    def to_dict(self):
        return {'id': str(self.id),
                'seed': str(self.seed),
                'misbehaviour': self.is_misbehavior(),
                'timestamp': str(self.timestamp),
                'elapsed': str(self.elapsed_time),
                'tool' : str(self.tool),
                'run' : str(self.run),
                'features': self.features,
                'rank': str(self.rank),
                'selected': str(self.selected_counter),
                'placed_mutant': str(self.placed_mutant),
                'text': self.text,
                'expected_label': str(self.expected_label),
                'predicted_label': str(self.predicted_label),
                'expected_label': str(self.expected_label),
                'confidence': str(self.confidence),
                'peformance': str(self.performance)
        }

    def dump(self, filename):
        data = self.to_dict()
        filedest = filename+".json"
        with open(filedest, 'w') as f:
            (json.dump(data, f, sort_keys=True, indent=4))

    def save_txt(self, filename):
        data = self.text
        filedest = filename + ".txt"
        with open(filedest, 'w') as f:
            f.write(data)

    def is_misbehavior(self):
        if self.expected_label != self.predicted_label:
            return True
        else:
            return False

    def export(self, all=False):
        if all:
            dst = join(Folder.DST_ALL, "mbr"+str(self.id))
        else:
            dst = join(Folder.DST_ARC, "mbr"+str(self.id))
        self.dump(dst)

    def clone(self):
        clone_text = Text(self.text, self.expected_label, self.seed)
        return clone_text