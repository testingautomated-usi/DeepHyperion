from typing import Tuple

from numpy import mean

from core.member import Member


class Individual:
    def __init__(self, m: Member):
        self.m: Member = m
        self.oob_ff: float = None
        self.seed: Member = None


    def clone(self) -> 'creator.base':
        raise NotImplemented()

    def evaluate(self):
        raise NotImplemented()

    def mutate(self):
        raise NotImplemented()

    def distance(self, i2: 'Individual'):
        i1 = self
        dist = i1.m.distance(i2.m)
        return dist