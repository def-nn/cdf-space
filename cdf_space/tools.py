from functools import reduce


def factorial(x):
    if type(x) != int or x < 1:
        raise ValueError('Argument x should be positive integer, not {}'.format(x))

    return reduce(lambda x, y: x * y, range(1, x + 1))
