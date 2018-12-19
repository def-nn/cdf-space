from collections.abc import Iterable
import numpy as np

from cdf_space.domain import Domain


class FeatureSpace:

    @property
    def undefined(self):
        raise KeyError("CDFSpace: trying to get undefined property")

    def __init__(self, domains=None):
        self.__data_size = None
        self.__domains = None

        self.set_domains(domains)

    def __validate_domain(self, domain):
        if not isinstance(domain, Domain):
            raise ValueError("domain should be instance of class cdf_space.domain.Domain, not {}".format(type(domain)))

        if self.__data_size is not None:
            if self.__data_size != domain.get_data_size():
                raise ValueError("All domains in CDF space should contain equal number of data points")
        else:
            self.__data_size = domain.get_data_size()

    def __clean_data(self):
        if not isinstance(self.__domains, Iterable):
            raise ValueError("Argument `domains` should be iterable object")

        for domain in self.__domains:
            self.add_domain(domain)

    def add_domain(self, domain):
        self.__validate_domain(domain)
        self.__domains.append(domain)

    def set_domains(self, domains):
        self.__domains = domains or []
        self.__clean_data()

    def compute_probability_distribution(self, optimized=False, dtype=np.float64):
        prob_dist = np.ones((self.__data_size,), dtype=dtype)

        for domain in self.__domains:
            domain.update_space_probability_distribution(prob_dist, optimized)

        prob_dist /= self.__data_size

        return prob_dist
