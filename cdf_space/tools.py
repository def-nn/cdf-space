from functools import reduce


def is_positive_int(x):
    return type(x) == int and x >= 1


def factorial(i):
    if not isinstance(i, int) or i < 0:
        raise ValueError('Argument x should be non-negative integer, not {}'.format(i))

    return reduce(lambda x, y: x * y, range(1, i + 1)) if i > 1 else 1

