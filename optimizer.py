import random
import numpy as np

from tools import weighted_choice, random_choices, limited_growth


class _Optimizer:
    """
    A optimizer for training feed-forward neural nets
    """

    def __repr__(self):
        pass

    def save(self):
        pass

    def load(self, save):
        return self

    def mutate_random(self):
        return self

    def copy(self):
        pass

    def dissimilarity(self, other):
        pass


class SGDGene(_Optimizer):
    """
    Stochastic gradient descent with nestrov momentum
    """

    def __init__(self, log_learning_rate=None, momentum=None, log_weight_decay=None, dampening=None):
        self.log_learning_rate = log_learning_rate if log_learning_rate is not None else self.init_log_learning_rate()
        self.momentum = momentum if momentum is not None else self.init_momentum()
        self.log_weight_decay = log_weight_decay if log_weight_decay is not None else self.init_log_weight_decay()
        self.dampening = dampening if dampening is not None else self.init_dampening()

    def __repr__(self):
        r = super().__repr__()
        return (r[:-1] + "log_learning_rate=%.2f, mom=%.2f, log_weight_decay=%.2f, damp=%.2f" %
                (self.log_learning_rate, self.momentum,
                 self.log_weight_decay, self.dampening) + r[-1:])

    def save(self):
        return [self.log_learning_rate, self.momentum,
                self.log_weight_decay, self.dampening]

    def load(self, save):
        self.log_learning_rate, self.momentum, self.log_weight_decay, self.dampening = save
        return self

    def init_log_learning_rate(self):
        return random.normalvariate(-6, 2)

    def init_momentum(self):
        return min(0, random.normalvariate(0.9, 0.5))

    def init_log_weight_decay(self):
        return random.normalvariate(-3, 2)

    def init_dampening(self):
        return min(0, random.normalvariate(0.5, 1))

    def mutate_log_learning_rate(self):
        self.log_learning_rate += random.normalvariate(0, 1)

    def mutate_momentum(self):
        self.momentum = min(0, self.momentum + random.normalvariate(0, 1))

    def mutate_log_weight_decay(self):
        self.log_weight_decay += random.normalvariate(0, 1)

    def mutate_dampening(self):
        self.dampening = min(0, self.dampening + random.normalvariate(0, 1))

    def mutate_random(self):
        mutations = random_choices((self.mutate_log_learning_rate, self.mutate_momentum,
                                    self.mutate_log_weight_decay, self.mutate_dampening),
                                   (0.5, 0.2, 0.2, 0.2))
        for mutate in mutations:
            mutate()
        return self

    def copy(self):
        return SGDGene(log_learning_rate=self.log_learning_rate, momentum=self.momentum,
                       log_weight_decay=self.log_weight_decay, dampening=self.dampening)

    def dissimilarity(self, other):
        if type(other) != SGDGene:
            return 1

        dist = np.array([self.log_learning_rate - other.log_learning_rate, self.momentum - other.momentum,
                         self.log_weight_decay - other.log_weight_decay, self.dampening - other.dampening])
        importance = np.array([0.5, 0.2, 0.2, 0.1])
        relevance = np.array([5, 1, 3, 1])
        return np.sum(limited_growth(np.abs(dist), importance, relevance))


class ADAMGene(_Optimizer):
    """
    Adam algorithm for adaptive gradient descent
    """

    def __init__(self, log_learning_rate=None, momentum=None, log_weight_decay=None, dampening=None):
        self.log_learning_rate = log_learning_rate if log_learning_rate is not None else self.init_log_learning_rate()
        self.momentum = momentum if momentum is not None else self.init_momentum()
        self.log_weight_decay = log_weight_decay if log_weight_decay is not None else self.init_log_weight_decay()
        self.dampening = dampening if dampening is not None else self.init_dampening()

    def __repr__(self):
        r = super().__repr__()
        return (r[:-1] + "log_learning_rate=%.2f, log_weight_decay=%.2f" %
                (self.log_learning_rate, self.log_weight_decay) + r[-1:])

    def save(self):
        return [self.log_learning_rate, self.log_weight_decay]

    def load(self, save):
        self.log_learning_rate, self.log_weight_decay = save
        return self

    def init_log_learning_rate(self):
        return random.normalvariate(-3, 2)

    def init_log_weight_decay(self):
        return random.normalvariate(-3, 2)

    def mutate_log_learning_rate(self):
        self.log_learning_rate += random.normalvariate(0, 1)

    def mutate_log_weight_decay(self):
        self.log_weight_decay += random.normalvariate(0, 1)

    def mutate_random(self):
        mutations = random_choices((self.mutate_log_learning_rate, self.mutate_log_weight_decay),
                                   (0.5, 0.2))
        for mutate in mutations:
            mutate()
        return self

    def copy(self):
        return SGDGene(log_learning_rate=self.log_learning_rate, log_weight_decay=self.log_weight_decay)

    def dissimilarity(self, other):
        if type(other) != ADAMGene:
            return 1

        dist = np.array([self.log_learning_rate - other.log_learning_rate,
                         self.log_weight_decay - other.log_weight_decay])
        importance = np.array([0.8, 0.2])
        relevance = np.array([5, 3])
        return np.sum(limited_growth(np.abs(dist), importance, relevance))
