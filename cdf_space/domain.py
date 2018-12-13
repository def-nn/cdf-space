import numbers
import numpy as np

from .kernels import BaseKernel


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum(np.square(p1 - p2)))


class Domain:

    @property
    def undefined(self):
        raise KeyError("Domain: trying to get undefined property")

    get_distance = undefined

    def __init__(self, data, kernel, dist_function=euclidean_distance):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
            if not isinstance(data[0], np.ndarray):
                raise ValueError("Argument `data` should be convertible to 2-dimensional numpy array")

        if data.size == 0:
            raise ValueError("Argument `data` can't be empty")

        if not isinstance(kernel, BaseKernel):
            raise ValueError("kernel should be an instance of cdf_space.kernels.BaseKernel class")

        if not callable(dist_function) or not isinstance(dist_function(data[0], data[0]), numbers.Number):
            raise ValueError(
                "dist_function should be callable object which returns numeric value".format(type(dist_function))
            )

        self.__data = data
        self.__get_distance = dist_function

        self.__kernel = kernel

    def __calculate_default_h(self):
        return np.max(self.__data) - np.min(self.__data)

    def generate_pdf_matrix(self, h=None, dtype=np.float64):
        data_size = self.__data.shape[0]
        dim = self.__data.data.shape[1]

        pdf_matrix = np.zeros((data_size,), dtype=dtype)

        if not h:
            h = self.__calculate_default_h()
        normalized_constant = self.__kernel.calculate_normalized_constant(dim)

        for i in data_size:
            for j in data_size:
                pdf_matrix[i] += self.__kernel.estimate_density(self.get_distance(data_size[i], data_size[j]) / h)
            pdf_matrix[i] *= normalized_constant / (data_size * h ** dim)

        return pdf_matrix
