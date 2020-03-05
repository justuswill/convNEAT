import random
import numpy as np
import gc

import torch


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


def check_cuda_memory():
    """
    Compiles a list of allocated Torch Tensors on the device
    """
    tensor_list = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    tensor_list += [obj]
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                if obj.data.is_cuda:
                    tensor_list += [obj]
        except:
            pass
    return tensor_list
