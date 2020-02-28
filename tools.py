import random
import numpy as np


def weighted_choice(choices, weights):
    return random.choices(choices, weights=weights)[0]


def random_choices(choices, chances):
    return list(np.array(choices)[np.random.rand(len(chances)) < chances])


def limited_growth(t, cap, relevance):
    k = 2.2/relevance
    return cap*(1-np.exp(-k*t))


def score_decay(accuracy, training, decay_factor=0.01):
    """
    With increasing training linearly increase the log error [log(10, 1-acc)] on the data
    """
    return 1 - 10**(np.log10(1 - accuracy) + decay_factor * training)
