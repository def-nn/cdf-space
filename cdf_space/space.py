from collections.abc import Iterable, Mapping, Sequence
import numpy as np

from cdf_space.kernels import EpanechnikovKernel
from cdf_space.domain import Domain


def with_initialized_space(decorated):
    def wrapper(*args):
        if not args[0].domains or not args[0].data_size:
            raise ValueError("Feature space is empty")
        return decorated(*args)
    return wrapper


class FeatureSpaceAnalyzer:

    @property
    def undefined(self):
        raise KeyError("CDFSpace: trying to get undefined property")

    def __init__(self, data=None, domains_meta=None, domains=None, default_kernel=None):
        if data is not None and domains is not None:
            raise ValueError("You must specify either the `data` attribute or the` domain` attribute")

        self.__data_size = None
        self.__domains = {}
        self.__multiple_domains = False

        if data is not None:
            default_kernel = default_kernel or EpanechnikovKernel()
            domains = FeatureSpaceAnalyzer.generate_domains_from_data(data, domains_meta, default_kernel)

        if domains is not None:
            self.domains = domains

    @property
    def data_size(self):
        return self.__data_size

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

        self.__multiple_domains = len(self.__domains) > 1

    def add_domains(self, domains):
        for domain in domains:
            self.add_domain(domain)

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
                "Argument `data` must be an iterable container with data for domains to be created"
            )

        if domains_meta is None:
            domains = [Domain(data[key], kernel=kernel, label=key) for key in _key_iterator]
        else:
            if not isinstance(domains_meta, Iterable):
                raise ValueError(
                    "Argument `domains_meta` must be an iterable container with information about domains to be created"
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
                    raise ValueError("All elements in `data` should contain equal number of data points")

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
                    dist_function=domain_meta.get('dist_function')
                ))

        return domains

    @with_initialized_space
    def compute_probability_distribution(self, optimized=True, dtype=np.float64):
        prob_dist = np.ones((self.__data_size,), dtype=dtype)

        for key in self.__domains:
            self.__domains[key].update_space_probability_distribution(prob_dist, optimized)

        prob_dist /= self.__data_size

        return prob_dist

    def __apply_mean_shift(self, i, convergence_limit, max_iter_num):
        y = {key: {'data': self.__domains[key].data[i]} for key in self.__domains}

        for j in range(max_iter_num):
            y_shifted = {}

            L_k = np.ones((self.__data_size,), dtype=np.float64)
            for d_key in self.__domains:
                _dist = self.__domains[d_key].calculate_distance(p_r=y[d_key]['data'])
                y[d_key]['k'] = self.__domains[d_key].kernel.estimate_density_vector(_dist)
                y[d_key]['g'] = self.__domains[d_key].kernel.compute_gradient_fall_rate_vector(_dist)

                L_k *= y[d_key]['k']

                y[d_key]['k'][np.where(y[d_key]['k'] == 0)] = 1

            gradient_converged = []

            for d_key in self.__domains:
                y_shifted[d_key] = {}

                weight_vector = L_k * y[d_key]['g'] / y[d_key]['k']
                weight_vector_sum = np.sum(weight_vector)

                if weight_vector_sum == 0:
                    y_shifted[d_key]['data'] = y[d_key]['data']
                    step_size = 0
                else:
                    y_shifted[d_key]['data'] = weight_vector.dot(self.__domains[d_key].data) / weight_vector_sum
                    step_size = self.__domains[d_key].calculate_distance(p_l=y[d_key]['data'],
                                                                         p_r=y_shifted[d_key]['data'],
                                                                         axis=0)

                gradient_converged.append(step_size <= convergence_limit)

            y = y_shifted
            if all(gradient_converged):
                break

        return y

    @with_initialized_space
    def find_convergence_points(self, convergence_limit, max_iter_num):
        convergence_points = []

        # TODO: fix this mess
        if not self.__multiple_domains:
            print('not multiple domains')
            domain = self.__domains[list(self.__domains.keys())[0]]

            for i in range(self.__data_size):
                y = domain.data[i]

                for j in range(max_iter_num):
                    _dist = domain.calculate_distance(p_r=y)
                    g = domain.kernel.compute_gradient_fall_rate_vector(_dist)
                    g_sum = np.sum(g)

                    if g_sum == 0:
                        y_shifted = y
                        step_size = 0
                    else:
                        y_shifted = g.dot(domain.data) / g_sum
                        step_size = domain.calculate_distance(p_l=y_shifted, p_r=y, axis=0)

                    y = y_shifted
                    if step_size <= convergence_limit:
                        break

                convergence_points.append(y)
        else:
            for i in range(self.__data_size):
                convergence_points.append(self.__apply_mean_shift(i, convergence_limit, max_iter_num))

        return convergence_points
