import random

from core.config import Config
from core.member import Member
from core.seed_pool import SeedPool


class SeedPoolAccessStrategy:
    def __init__(self, pool: SeedPool):
        self.pool = pool
        self.counter = -1

    def get_member_randomly(self) -> Member:
        return random.choice(self.pool)

    def get_member_circular(self) -> Member:
        self.counter += 1
        self.counter = self.counter % len(self.pool)
        return self.pool[self.counter]

    def get_member_diverse(self) -> Member:
        self.counter += 1
        self.counter = self.counter % len(self.pool)
        return self.pool[self.counter]

    def get_seed(self):
        generator_name = self.pool.problem.config.generator_name
        if generator_name == Config.GEN_RANDOM:
            seed = self.pool.problem.generate_random_member()
        elif generator_name == Config.GEN_RANDOM_SEEDED:
            seed = self.get_member_randomly()
        elif generator_name == Config.GEN_SEQUENTIAL_SEEDED:
            seed = self.get_member_circular()
        elif generator_name == Config.GEN_DIVERSITY:
            seed = self.get_member_diverse()
        else:
            raise NotImplemented(generator_name)
        return seed