import random

import numpy as np
from deap import creator

from core.config import Config
from core.individual import Individual
from self_driving.beamng_member import BeamNGMember

import logging as log


class BeamNGIndividual(Individual):
    counter = 0

    def __init__(self, m: BeamNGMember, config: Config):
        super().__init__(m)
        self.m: BeamNGMember = self.m
        BeamNGIndividual.counter += 1
        self.name = f'ind{str(BeamNGIndividual.counter)}'
        self.name_ljust = self.name.ljust(6)
        self.config = config
        self.m.parent = self
        self.seed = None #: BeamNGMember

    def evaluate(self):
        self.m.evaluate()

        border = self.m.distance_to_boundary
        self.oob_ff = border if border > 0 else -0.1

        log.debug(f'evaluated {self}')

        return self.oob_ff

    def clone(self) -> 'BeamNGIndividual':
        res: BeamNGIndividual = creator.Individual(self.m.clone(), self.config)
        res.seed = self.seed
        log.debug(f'cloned to {res} from {self}')
        return res

    def to_dict(self):
        return {'name': self.name,
                'm': self.m.to_dict(),
                'seed': self.seed.to_dict()}

    @classmethod
    def from_dict(self, d):
        m = BeamNGMember.from_dict(d['m'])
        ind = BeamNGIndividual(m, None)
        ind.name = d['name']
        return ind

    def __str__(self):
        return f'{self.name_ljust}  m[{self.m}] seed[{self.seed}]'

    def mutate(self):
        road_to_mutate = self.m
        road_to_mutate.mutate()
        log.debug(f'mutated {road_to_mutate}')
