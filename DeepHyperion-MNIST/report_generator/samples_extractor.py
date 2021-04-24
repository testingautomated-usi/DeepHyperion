import logging as log

import glob
import os
import matplotlib.pyplot as plt

import numpy as np
import json

from vectorization_tools import vectorize
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


class MNistSample(Sample):

    def __init__(self, basepath):
        super(MNistSample, self).__init__()
        self.basepath = basepath
        self.performance = None
        self.seed = None
        self.expected_label = None
        self.predicted_label = None
        self.xml_desc = None
        self.image = None

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
                'moves': self.get_value("moves"),
                'bitmaps': self.get_value("bitmaps"),
                'orientation': self.get_value("orientation"),
                'tool' : self.tool,
                'run' : self.run,
                'features': self.features}

    def dump(self):
        data = self.to_dict()
        filedest = os.path.join(os.path.dirname(self.basepath), "info_"+str(self.id)+".json")
        with open(filedest, 'w') as f:
            (json.dump(data, f, sort_keys=True, indent=4))


class DLFuzzSample(MNistSample):
    """

    A DLFuzz sample is a sample made of the following json, plus an xml and npy.

    {
    "elapsed": "0:00:37.304367",                (Sample)
    "expected_label": 5,                        (MNistSample)
    "features": [                               (None)
        "Bitmaps",
        "Orientation",
        "Moves"
    ],
    "id": 98,                                   (Sample)
    "misbehaviour": false,                      (Sample)
    "performance": "0.9999646",                 (MNistSample)
    "predicted_label": 5,                       (MNistSample)
    "run": "0",                                 (Sample)
    "seed": " 2003",                            (MNistSample)
    "timestamp": "2020-12-23 18:01:34.255625",  (Sample)
    "tool": "DLFuzz"                            (Sample)
    }
    """
    def __init__(self, basepath):
        super(DLFuzzSample, self).__init__(basepath)

        # Load the metadata (json) file and the files to compute the features (xml, npy)
        npy_path = basepath + ".npy"
        # TODO Does this have to be an instance attribute? Maybe we can remove it after we compute the features?
        self.image = np.load(npy_path)
        json_path = basepath + ".json"
        with open(json_path) as jf:
            json_data = json.load(jf)
        try:
            # TODO Does this have to be an instance attribute? Maybe we can remove it after we compute the features?
            self.xml_desc = json_data["xml_desc"]
            # TODO Does this have to be an instance attribute? Maybe we can remove it after we compute the features?
            self.approximated_model = False
        except KeyError:
            # TODO Does this have to be an instance attribute? Maybe we can remove it after we compute the features?
            self.xml_desc = vectorize(self.image)
            # TODO Does this have to be an instance attribute? Maybe we can remove it after we compute the features?
            self.approximated_model = True

        # Set Sample attributes
        self.id = json_data["id"]
        self.tool = json_data["tool"]
        self.misbehaviour = json_data["misbehaviour"]
        self.run = json_data["run"]
        self.timestamp = json_data["timestamp"]
        self.elapsed = json_data["elapsed"]
        # Compute the metrics, those are not available in the metadata json
        self.features["moves"] = metrics.move_distance(self.xml_desc)
        self.features["bitmaps"] = metrics.dark_bitmaps(self.image)
        self.features["orientation"] = metrics.orientation_calc(self.image)

        # Set MNistSample attributes
        self.performance = json_data["performance"]
        self.predicted_label = json_data["predicted_label"]
        self.expected_label = json_data["expected_label"]
        self.seed = json_data["seed"]

        # Store to file besides the input json
        self.dump()


class DeepJanusSample(MNistSample):
    """

    A DeepJanus sample is a sample made of the following json, plus an xml and svg.
    Compare to other samples, like DLFuzzSample, it has a different folder structure, but content wise
    they are the same

    {
    "elapsed": "0:00:37.304367",                (Sample)
    "expected_label": 5,                        (MNistSample)
    "features": [                               (None)
        "Bitmaps",
        "Orientation",
        "Moves"
    ],
    "id": 98,                                   (Sample)
    "misbehaviour": false,                      (Sample)
    "performance": "0.9999646",                 (MNistSample)
    "predicted_label": 5,                       (MNistSample)
    "run": "0",                                 (Sample)
    "seed": " 2003",                            (MNistSample)
    "timestamp": "2020-12-23 18:01:34.255625",  (Sample)
    "tool": "DeepJanus"                         (Sample)
    }
    """
    def __init__(self, basepath):
        super(DeepJanusSample, self).__init__(basepath)

        # Load the metadata (json) file and the files to compute the features (xml, svg)
        npy_path = basepath + ".npy"
        json_path = basepath + ".json"
        svg_path = basepath + ".svg"

        with open(json_path) as jf:
            json_data = json.load(jf)

        self.image = np.load(npy_path)

        if os.path.exists(svg_path):
            self.approximated_model = False

            with open(svg_path, 'r') as input_file:
                self.xml_desc = input_file.read()
        else:
            self.approximated_model = True
            self.xml_desc = vectorize(self.image)

        # Set Sample attributes
        self.id = json_data["id"]
        self.tool = json_data["tool"]
        self.misbehaviour = json_data["misbehaviour"]
        self.run = json_data["run"]
        self.timestamp = json_data["timestamp"]
        self.elapsed = json_data["elapsed"]

        # Compute the metrics, those are not available in the metadata json
        self.features["moves"] = metrics.move_distance(self.xml_desc)
        self.features["bitmaps"] = metrics.dark_bitmaps(self.image)
        self.features["orientation"] = metrics.orientation_calc(self.image)

        # Set MNistSample attributes
        self.performance = json_data["performance"]
        self.predicted_label = json_data["predicted_label"]
        self.expected_label = json_data["expected_label"]
        self.seed = json_data["seed"]

        # Store to file besides the input json
        self.dump()


class DeepHyperionSample(MNistSample):
    """

    A DeepHyperion sample is a sample similar to DeepJanusSample

    {
    "elapsed": "0:00:37.304367",                (Sample)
    "expected_label": 5,                        (MNistSample)
    "features": [                               (None)
        "Bitmaps",
        "Orientation",
        "Moves"
    ],
    "id": 98,                                   (Sample)
    "misbehaviour": false,                      (Sample)
    "performance": "0.9999646",                 (MNistSample)
    "predicted_label": 5,                       (MNistSample)
    "run": "0",                                 (Sample)
    "seed": " 2003",                            (MNistSample)
    "timestamp": "2020-12-23 18:01:34.255625",  (Sample)
    "tool": "DeepHyperion"                         (Sample)
    }
    """
    def __init__(self, basepath):
        super(DeepHyperionSample, self).__init__(basepath)

        # Load the metadata (json) file and the files to compute the features (xml, svg)
        npy_path = basepath + ".npy"
        json_path = basepath + ".json"
        svg_path = basepath + ".svg"

        with open(json_path) as jf:
            json_data = json.load(jf)

        self.image = np.load(npy_path)

        if os.path.exists(svg_path):
            self.approximated_model = False

            with open(svg_path, 'r') as input_file:
                self.xml_desc = input_file.read()
        else:
            self.approximated_model = True
            self.xml_desc = vectorize(self.image)

        # Set Sample attributes
        self.id = json_data["id"]
        self.tool = json_data["tool"]
        self.misbehaviour = json_data["misbehaviour"]
        self.run = json_data["run"]
        self.timestamp = json_data["timestamp"]
        self.elapsed = json_data["elapsed"]

        # Compute the metrics, those are not available in the metadata json
        self.features["moves"] = metrics.move_distance(self.xml_desc)
        self.features["bitmaps"] = metrics.dark_bitmaps(self.image)
        self.features["orientation"] = metrics.orientation_calc(self.image)

        # Set MNistSample attributes
        self.performance = json_data["performance"]
        self.predicted_label = json_data["predicted_label"]
        self.expected_label = json_data["expected_label"]
        self.seed = json_data["seed"]

        # Store to file besides the input json
        self.dump()
