import logging as log

import glob
import os
import matplotlib.pyplot as plt

import numpy as np
import json

from report_generator import metrics


from math import pi, atan2, sqrt, pow
LABEL = 5

class Sample:

    def __init__(self):
        self.id = None
        self.tool = None
        self.misbehaviour = False
        self.run = None
        self.timestamp = None
        self.elapsed = None
        self.features = {}
        self.is_valid = True
        self.valid_according_to = None

    # TODO Maybe make this an abstract method?
    def is_misbehavior(self):
        return self.misbehaviour

    def get_value(self, feature_name):
        if feature_name in self.features.keys():
            return self.features[feature_name]
        else:
            return None

    @staticmethod
    def from_dict(the_dict):
        sample = Sample()
        for k in sample.__dict__.keys():
            setattr(sample, k, None if k not in the_dict.keys() else the_dict[k])
        return sample


class IMDBSample(Sample):

    def __init__(self, basepath):
        super(IMDBSample, self).__init__()
        self.basepath = basepath
        # self.performance = None
        self.seed = None
        self.expected_label = None
        self.predicted_label = None
        self.confidence = None
        self.xml_desc = None
        self.image = None
        self.performance = None

    def to_dict(self):
        """
            This is common for all the MNIST samples
        """

        return {'id': self.id,
                'seed': self.seed,
                'expected_label': self.expected_label,
                'predicted_label': self.predicted_label,
                'is_valid': self.is_valid,
                'valid_according_to': self.valid_according_to,
                'misbehaviour': self.is_misbehavior(),
                'performance': self.performance,
                'elapsed': self.elapsed,
                'timestamp': self.timestamp,
                'negcount': self.get_value("negcount"),
                'poscount': self.get_value("poscount"),
                'verbcount': self.get_value("verbcount"),
                'tool' : self.tool,
                'run' : self.run,
                'features': self.features}

    def dump(self):
        data = self.to_dict()
        filedest = os.path.join(os.path.dirname(self.basepath), "info_"+str(self.id)+".json")
        with open(filedest, 'w') as f:
            (json.dump(data, f, sort_keys=True, indent=4))


class DeepHyperionSample(IMDBSample):

    def __init__(self, basepath):
        super(DeepHyperionSample, self).__init__(basepath)

        json_path = basepath + ".json"


        with open(json_path) as jf:
            json_data = json.load(jf)

        self.text = json_data['text']

        # Set Sample attributes
        self.id = json_data["id"]
        self.tool = json_data["tool"]
        self.misbehaviour = json_data["misbehaviour"]
        self.run = json_data["run"]
        self.timestamp = json_data["timestamp"]
        self.elapsed = json_data["elapsed"]
        self.confidence = json_data['confidence']

        # Compute the metrics, those are not available in the metadata json
        self.features["negcount"] = metrics.count_neg(self.text)
        self.features["poscount"] = metrics.count_pos(self.text)
        self.features["verbcount"] = metrics.count_verbs(self.text)

        # Set Sample attributes
        # self.performance = json_data["performance"]
        self.predicted_label = json_data["predicted_label"]
        self.expected_label = json_data["expected_label"]
        self.seed = json_data["seed"]
        self.performance = json_data["peformance"]

        # Store to file besides the input json
        self.dump()
