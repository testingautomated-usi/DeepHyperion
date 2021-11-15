import os
from pathlib import Path
from datetime import datetime
import self_driving.beamng_config as cfg

class Folders:
    def __init__(self, lib_folder: str, config):

        self.lib: Path = Path(lib_folder).resolve()
        self.root: Path = self.lib.joinpath('..').resolve()
        self.data: Path = self.root.joinpath('data').absolute()
        self.log: Path = self.root.joinpath(f'logs/{config.Feature_Combination[0]}_{config.Feature_Combination[1]}/run_{config.run_id}').absolute()
        self.log_ini: Path = self.data.joinpath('log.ini').absolute()
        self.member_seeds: Path = self.data.joinpath('member_seeds').absolute()
        self.experiments: Path = self.log.joinpath(f'outputs').absolute()
        self.simulations: Path = self.log.joinpath(f'simulations').absolute()
        self.trained_models_colab: Path = self.data.joinpath('trained_models_colab').absolute()

config = cfg.BeamNGConfig()
folders: Folders = Folders(os.path.dirname(__file__), config)
