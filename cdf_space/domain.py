import numbers, collections
import numpy as np

from .kernels import BaseKernel


def euclidean_distance(p1, p2, h=1, axis=0):
    return np.sqrt(np.sum(np.square((p1 - p2) / h), axis=axis))


class Domain:

    def __init__(self, data, kernel, h=None, label=None, dist_function=euclidean_distance):
        self.__data = data
        self.__get_distance = dist_function
        self.__kernel = kernel

        self.__clean()

        self.__data_size = self.__data.shape[0]
        self.__dim = self.__data.data.shape[1]

        self.__h = None
        self.h = h
        self.label = label

    def __clean(self):
        if not isinstance(self.__data, np.ndarray):
            self.__data = np.array(self.__data)

        if len(self.__data.shape) != 2:
            raise ValueError("Argument `data`  must be convertible to 2-dimensional numpy array")

        if self.__data.size == 0:
            raise ValueError("Argument `data` can't be empty")

        if not isinstance(self.__kernel, BaseKernel):
            raise ValueError("kernel should be an instance of cdf_space.kernels.BaseKernel class")

        if not callable(self.__get_distance) or \
                not isinstance(self.__get_distance(self.__data[0], self.__data[0]), numbers.Number):
            raise ValueError(
                "dist_function should be callable object which returns numeric value".format(type(self.__get_distance))
            )

    def __calculate_default_h(self):
        max_dist = -float('inf')

        for x in range(self.__data.shape[0]):
            for x_i in range(x + 1, self.__data.shape[0]):
                _dist = self.__get_distance(self.__data[x], self.__data[x_i])
                max_dist = _dist if _dist > max_dist else max_dist

        return max_dist

    @property
    def label(self):
        return self.__label

    @label.setter
    def label(self, val):
        if val is not None and not isinstance(val, collections.Hashable):
            raise ValueError("label should be a hashable object, not {}".format(type(self.__label)))

        self.__label = val

    @property
    def data_size(self):
        return self.__data_size

    @property
    def h(self):
        return self.__h

    @h.setter
    def h(self, val):
        if val is not None:
            if not isinstance(val, numbers.Number):
                raise ValueError("bandwidth parameter `h` must be a number")
            self.__h = val
        else:
            self.__h = self.__calculate_default_h()

    def __apply_kernel_estimator_optimized(self, prob_dist):
        for i in range(self.__data_size):
            _dist = self.__get_distance(self.__data, self.__data[i], h=self.__h, axis=1)
            prob_dist[i] *= self.__kernel.estimate_density_optimized(_dist)

    def __apply_kernel_estimator(self, prob_dist):
        for i in range(self.__data_size):
            x_density = 0
            for j in range(self.__data_size):
                _dist = self.__get_distance(self.__data[i], self.__data[j], h=self.__h)
                x_density += self.__kernel.estimate_density(_dist)
            prob_dist[i] *= x_density

    def __compute_probability_distribution(self, dst, n, optimized):
        apply_kernel_estimator = self.__apply_kernel_estimator_optimized if optimized else self.__apply_kernel_estimator
        apply_kernel_estimator(dst)

        normalized_constant = self.__kernel.calculate_normalized_constant(self.__dim)
        dst *= normalized_constant / (n * self.__h ** self.__dim)

    def generate_domain_probability_distribution(self, optimized=True, dtype=np.float64):
        prob_dist = np.ones((self.__data_size,), dtype=dtype)
        self.__compute_probability_distribution(prob_dist, self.__data_size, optimized)

        return prob_dist

    def update_space_probability_distribution(self, prob_dist, optimized=True):
        self.__compute_probability_distribution(prob_dist, 1, optimized)
