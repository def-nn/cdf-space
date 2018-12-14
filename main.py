import numpy as np

from cdf_space.kernels import EpanechnikovKernel, BaseKernel
from cdf_space.domain import euclidean_distance


if __name__ == '__main__':
    k = EpanechnikovKernel()
    a = np.array(
        [
            [0, 0, 0],
            [100, 0, -100],
            [20, 0, -100],
            [0, 5, -8],
        ]
    )
    a[:,:] = [1, 2, 3]
    print(a)

