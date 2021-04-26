from core import ea
from core.archive_impl import SmartArchive
from self_driving.beamng_config import BeamNGConfig
import matplotlib.pyplot as plt

from self_driving.beamng_problem import BeamNGProblem

config = BeamNGConfig()

problem = BeamNGProblem(config)

if __name__ == '__main__':
    ea.main(problem)
    print('done')

    plt.ioff()
    plt.show()
