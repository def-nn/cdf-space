import time, random
import numpy as np

from cdf_space.kernels import EpanechnikovKernel, GaussianKernel
from cdf_space.domain import Domain
from cdf_space.space import FeatureSpace


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

    width = 5
    height = 11

    data = np.empty((width * height, 2), dtype=np.int32)
    data_1 = np.empty((width * height, 3), dtype=np.int32)
    print(data.shape)
    print(data_1.shape)

    for i in range(height):
        for j in range(width):
            data[i * width + j][0] = i
            data[i * width + j][1] = j

            data_1[i * width + j][0] = 1
            data_1[i * width + j][1] = 1
            data_1[i * width + j][2] = 1

            # print(data[i * width + j], end='  ')
            # print(data_1[i * width + j], end='  ')
        # print()

    e_kernel = EpanechnikovKernel()
    g_kernel = GaussianKernel()
    domains = [Domain(data_1, g_kernel, label='random', h=20), Domain(data, e_kernel)]
    space = FeatureSpace(domains=domains[1:])

    print()

    pd = space.compute_probability_distribution()

    # for i in range(height):
    #     for j in range(width):
    #         print(str(pd[i * width + j])[:10], end='   ')

    max = np.max(pd)

    print('hey', pd.shape, np.sum(pd))

    # for each in pd:
    #     print(each / max)
