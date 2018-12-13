import math

from cdf_space.tools import factorial


class BaseKernel:
    def __init__(self):
        self.__normalization_constants = None

        raise NotImplementedError("""
            You can't create instance of cdf_space.kernels.BaseKernel class.
            Use it as abstract class for your own custom kernel implementation or use one of the provided in the library 
        """)

    def add_precomputed_norm_constant(self, dim):
        if dim not in self.__normalization_constants:
            self.__normalization_constants[dim] = self.__calculate_norm_constant(dim)

    @property
    def c(self, dim):
        if dim not in self.__normalization_constants:
            self.add_precomputed_norm_constant(dim)

        return self.__normalization_constants[dim]

    def estimate_density(self, *args, **kwargs):
        raise NotImplementedError("""
            Method {}.estimate_density hasn't been implemented
        """.format(self.__class__.__name__))

    def __calculate_norm_constant(self, *args, **kwargs):
        raise NotImplementedError("""
            Method {}.__calculate_norm_constant hasn't been implemented
        """.format(self.__class__.__name__))


class EpanechnikovKernel(BaseKernel):

    def __init__(self):
        self.__normalization_constants = {}

    def __calculate_norm_constant(self, dim):
        k = dim // 2

        if k * 2 == dim:
            _c = math.pi ** k / factorial(k)
        else:
            _c = (2 * factorial(k) * (4 * math.pi) ** k) / factorial(2 * k + 1)

        return _c

    def estimate_density(self, dist, dim):
        return (1 - dist) * (dim + 2) / (self.c(dim) * 2) if dist < 1 else 0


class GaussianKernel(BaseKernel):

    def __init__(self):
        self.__normalization_constants = {}

    def __calculate_norm_constant(self, dim):
        return (2 * math.pi) ** (dim / 2)

    def estimate_density(self, dist, dim):
        return math.exp(dist / -2) / self.c(dim)
