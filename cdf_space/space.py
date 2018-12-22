from collections.abc import Iterable, Mapping, Sequence
import numpy as np

from cdf_space.kernels import EpanechnikovKernel
from cdf_space.domain import Domain


class FeatureSpace:

    @property
    def undefined(self):
        raise KeyError("CDFSpace: trying to get undefined property")

    def __init__(self, data=None, domains_meta=None, domains=None, default_kernel=EpanechnikovKernel):
        if data is not None and domains is not None:
            raise ValueError("You must specify either the `data` attribute or the` domain` attribute")

        self.__data_size = None
        self.__domains = {}

        if data:
            domains = FeatureSpace.generate_domains_from_data(data, domains_meta, default_kernel)

        if domains is not None:
            self.domains = domains

    @property
    def domains(self):
        return self.__domains

    @domains.setter
    def domains(self, domains):
        if not isinstance(domains, Iterable):
            raise ValueError("Argument `domains` should be iterable object")

        self.__domains = {}

        for domain in domains:
            self.add_domain(domain)

    def __validate_domain(self, domain):
        if not isinstance(domain, Domain):
            raise ValueError("domain should be instance of class cdf_space.domain.Domain, not {}".format(type(domain)))

        if self.__data_size is not None:
            if self.__data_size != domain.data_size:
                raise ValueError("All domains in CDF space should contain equal number of data points")
        else:
            self.__data_size = domain.data_size

    def add_domain(self, domain):
        self.__validate_domain(domain)

        domain_key = domain.label or len(self.__domains)
        self.__domains[domain_key] = domain

    @staticmethod
    def generate_domains_from_data(data, domains_meta, kernel):
        if isinstance(data, Sequence):
            _key_iterator = range(len(data))
        elif isinstance(data, Mapping):
            _key_iterator = data.keys()
        elif isinstance(data, Iterable):
            return [Domain(each, kernel) for each in data]
        else:
            raise ValueError(
                "Argument `data` must be a container with data for domains to be created"
            )

        if domains_meta is None:
            domains = [Domain(data[key], kernel=kernel, label=key) for key in _key_iterator]
        else:
            if not isinstance(domains_meta, Iterable):
                raise ValueError(
                    "Argument `domains_meta` must be a container with information about domains to be created"
                )

            data_size = None

            for key in _key_iterator:
                if not isinstance(data[key], np.ndarray):
                    data[key] = np.array(data[key])

                if len(data[key].shape) != 2:
                    raise ValueError("All elements of `data` must be convertible to 2-dimensional numpy array")

                if data_size is None:
                    data_size = data[key].shape[0]
                elif data_size != data[key].shape[0]:
                    raise ValueError("All `data` elements should contain equal number of data points")

            domains = []

            for domain_meta in domains_meta:
                if not isinstance(domain_meta, Mapping) or 'cols_idx' not in domain_meta.keys():
                    raise ValueError(
                        "Each item in `domains_meta` attribute should be a mapping object with required key `cols_idx`"
                    )

                col_idx = domain_meta['cols_idx']

                if not isinstance(col_idx, Iterable):
                    domain_data = data[col_idx]
                elif isinstance(col_idx, Sequence) and len(col_idx):
                    domain_dimension = sum([data[idx].shape[1] for idx in col_idx])

                    domain_data = np.zeros((data_size, domain_dimension), dtype=domain_meta.get('dtype'))

                    i = 0
                    for idx in col_idx:
                        j = i + data[idx].shape[1]
                        domain_data[:, i:j] = data[idx]
                        i = j
                elif isinstance(col_idx, Mapping) and len(col_idx):
                    domain_dimension = 0

                    for idx in col_idx:
                        if not isinstance(col_idx[idx], Sequence):
                            raise ValueError("`domains_meta` attribute is not valid. Please, refer to docs")

                        domain_dimension += len(col_idx[idx])

                    domain_data = np.zeros((data_size, domain_dimension), dtype=domain_meta.get('dtype'))

                    i = 0
                    for idx in col_idx:
                        for col in col_idx[idx]:
                            domain_data[:, i] = data[idx][:, col]
                            i += 1
                else:
                    raise ValueError("""
                        Each element of `domains_meta` should return either key or index of `data` array or mapping or 
                        sequence object using key `cols_idx`
                    """)

                domains.append(Domain(
                    data=domain_data,
                    kernel=domain_meta.get('kernel') or kernel,
                    h=domain_meta.get('h'),
                    label=domain_meta.get('label'),
                    dist_function=domain_meta.get('domain_meta')
                ))

        return domains

    def compute_probability_distribution(self, optimized=True, dtype=np.float64):
        prob_dist = np.ones((self.__data_size,), dtype=dtype)

        for key in self.__domains:
            self.__domains[key].update_space_probability_distribution(prob_dist, optimized)

        prob_dist /= self.__data_size

        return prob_dist
