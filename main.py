import time
import numpy as np

from cdf_space.kernels import EpanechnikovKernel, BaseKernel, GaussianKernel
from cdf_space.domain import Domain


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

    width = 15
    height = 11
    data = np.empty((width * height, 2), dtype=np.int32)

    # print(data.shape)

    for i in range(height):
        for j in range(width):
            data[i * width + j][0] = i
            data[i * width + j][1] = j
            # print(data[i * width + j], end='  ')
        # print()

    kernel = EpanechnikovKernel()
    domain = Domain(data, kernel)

    exit(1)
    _time = time.time()

    pdf = domain.generate_domain_probability_distribution()

    exec_time = time.time() - _time

    _time = time.time()

    print('foo')

    print(exec_time)

    pdf_optimized = domain.generate_domain_probability_distribution(optimized=True)

    exec_time_by_row = time.time() - _time

    # print(pdf == pdf_row)
    print(exec_time_by_row)

    # for i in range(width * height):
    #     print(data[i], abs(pdf[i] - pdf_row[i]) < 10 ** -10, pdf[i], pdf_row[i])

    print()
    #
    # for i in range(height):
    #     for j in range(width):
    #         print(str(pdf[i * width + j])[:12], end='   ')
    #     print()
    #     for j in range(width):
    #         print("{:^12}".format(str(data[i * width + j])), end='   ')
    #     print()
