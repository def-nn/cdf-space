import time, random
import cv2
import numpy as np

from cdf_space.kernels import EpanechnikovKernel, GaussianKernel
from cdf_space.domain import Domain
from cdf_space.space import FeatureSpaceAnalyzer


if __name__ == '__main__':
    image = cv2.imread("images/num3.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(type(image))
    print(image.shape)

    width, height = image.shape[0], image.shape[1]
    data_size = width * height

    data_s = np.empty((data_size, 2), dtype=np.int32)
    data_r = np.empty((data_size, 1), dtype=np.int32)

    data_joint = np.empty((data_size, 3), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            data_s[i * width + j][0] = i
            data_s[i * width + j][1] = j

            data_r[i * width + j][0] = image[i][j]

            # data_joint[i * width + j][0] = i
            # data_joint[i * width + j][1] = j
            # data_joint[i * width + j][2] = image[i][j]

    print('hey')

    kernel = EpanechnikovKernel()

    domain_s = Domain(data=data_s, kernel=kernel, label='spatial', h=8)
    domain_r = Domain(data=data_r, kernel=kernel, label='color', h=4)
    #
    analyzer = FeatureSpaceAnalyzer(domains=[domain_s, domain_r])
    # analyzer = FeatureSpaceAnalyzer(data=(data_joint,), domains_meta=({'cols_idx': 0, 'label': 'joint', 'h': 4},))
    print('hey')
    print(analyzer.domains)
    print('hey')

    _time = time.time()

    analyzer.find_convergence_points(0.01, 100)

    print("exec time: {}".format(time.time() - _time))

    # pd_max = np.max(pd)
    #
    # pd_image = pd.reshape(image.shape) / pd_max
    # cv2.imshow("image", pd_image)
    # cv2.waitKey()
    # cv2.imwrite('images/pd_num3.png', pd_image)

    # pd_image = np.empty(image.shape)
    #
    # for i in range(height):
    #     for j in range(width):
    #         pd_image[i][j] = (pd[i * width + j] / pd_max)
    #
    # cv2.imshow("image", pd_image)
    # cv2.waitKey()
    # cv2.imwrite('images/pd_num3.jpg', pd_image)

    print(pd_max, np.sum(pd))

    # k = EpanechnikovKernel()
    # a = np.array(
    #     [
    #         [0, 0, 0],
    #         [100, 0, -100],
    #         [20, 0, -100],
    #         [0, 5, -8],
    #     ]
    # )
    #
    # width = 5
    # height = 11
    #
    # data = np.empty((width * height, 2), dtype=np.int32)
    # data_1 = np.empty((width * height, 3), dtype=np.int32)
    # print(data.shape)
    # print(data_1.shape)
    #
    # for i in range(height):
    #     for j in range(width):
    #         data[i * width + j][0] = i
    #         data[i * width + j][1] = j
    #
    #         data_1[i * width + j][0] = 1
    #         data_1[i * width + j][1] = 1
    #         data_1[i * width + j][2] = 1
    #
    #         # print(data[i * width + j], end='  ')
    #         # print(data_1[i * width + j], end='  ')
    #     # print()
    #
    # e_kernel = EpanechnikovKernel()
    # g_kernel = GaussianKernel()
    # domains = [Domain(data_1, g_kernel, label='random', h=20), Domain(data, e_kernel)]
    # space = FeatureSpaceAnalyzer(domains=domains[1:])
    #
    # print()
    #
    # pd = space.compute_probability_distribution()
    #
    # # for i in range(height):
    # #     for j in range(width):
    # #         print(str(pd[i * width + j])[:10], end='   ')
    #
    # max = np.max(pd)
    #
    # print('hey', pd.shape, np.sum(pd))

    # for each in pd:
    #     print(each / max)
