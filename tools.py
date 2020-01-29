import random
import numpy as np


def weighted_choice(choices, weights):
    return random.choices(choices, weights=weights)[0]


def random_choices(choices, chances):
    return list(np.array(choices)[np.random.rand(len(chances)) < chances])


def limited_growth(t, cap, relevance):
    k = 2.2/relevance
    return cap*(1-np.exp(-k*t))