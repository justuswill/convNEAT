import warnings

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import (
    pairwise_distances,
    pairwise_distances_argmin,
)
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import ConvergenceWarning


class KMedoids(BaseEstimator, ClusterMixin, TransformerMixin):
    """
    k-Medoids clustering.

    Based on the implementation from scikit-learn by

    Timo Erkkilä <timo.erkkila@gmail.com>
    Antti Lehmussola <antti.lehmussola@gmail.com>
    Kornel Kiełczewski <kornel.mail@gmail.com>
    Zane Dufour <zane.dufour@gmail.com>
    License: BSD 3 clause

    Adapted to use a costume cluster init and scoring

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of medoids to
        generate.

    metric : string, or callable, optional, default: 'euclidean'
        What distance metric to use. See :func:metrics.pairwise_distances

    max_iter : int, optional, default : 300
        Specify the maximum number of iterations when fitting.

    random_state : int, RandomState instance or None, optional
        Specify random state for the random number generator. Used to
        initialise medoids when init='random'.

    Attributes
    ----------
    cluster_centers_ : array, shape = (n_clusters, n_features)
            or None if metric == 'precomputed'
        Cluster centers, i.e. medoids (elements from the original dataset)

    medoid_indices_ : array, shape = (n_clusters,)
        The indices of the medoid rows in X

    labels_ : array, shape = (n_samples,)
        Labels of each point

    score_ : float
        Sum of squared distances to cluster centers

    References
    ----------
    Kaufman, L. and Rousseeuw, P.J., Statistical Data Analysis Based on
    the L1–Norm and Related Methods, edited by Y. Dodge, North-Holland,
    405–416. 1987

    Notes
    -----
    Since all pairwise distances are calculated and stored in memory for
    the duration of fit, the space complexity is O(n_samples ** 2).

    """

    def __init__(self, n_clusters=8, metric="euclidean", max_iter=300, random_state=None,):
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.random_state = random_state

    def _check_nonnegative_int(self, value, desc):
        """Validates if value is a valid integer > 0"""

        if (
                value is None
                or value <= 0
                or not isinstance(value, (int, np.integer))
        ):
            raise ValueError(
                "%s should be a nonnegative integer. "
                "%s was given" % (desc, value)
            )

    def _check_init_args(self):
        """Validates the input arguments. """

        # Check n_clusters and max_iter
        self._check_nonnegative_int(self.n_clusters, "n_clusters")
        self._check_nonnegative_int(self.max_iter, "max_iter")

    def fit(self, X, old_centers=None):
        """Fit K-Medoids to the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features), \
                or (n_samples, n_samples) if metric == 'precomputed'
            Dataset to cluster.

        y : Ignored

        Returns
        -------
        self
        """
        random_state_ = check_random_state(self.random_state)

        self._check_init_args()
        X = check_array(X, accept_sparse=["csr", "csc"])
        if self.n_clusters > X.shape[0]:
            raise ValueError(
                "The number of medoids (%d) must be less "
                "than the number of samples %d."
                % (self.n_clusters, X.shape[0])
            )

        D = pairwise_distances(X, metric=self.metric)
        medoid_idxs = self._init_centers(D, self.n_clusters, old_centers=old_centers)
        labels = None

        # Continue the algorithm as long as
        # the medoids keep changing and the maximum number
        # of iterations is not exceeded
        for self.n_iter_ in range(0, self.max_iter):
            old_medoid_idxs = np.copy(medoid_idxs)
            labels = np.argmin(D[medoid_idxs, :], axis=0)

            # Update medoids with the new cluster indices
            self._update_medoid_idxs_in_place(D, labels, medoid_idxs)
            if np.all(old_medoid_idxs == medoid_idxs):
                break
            elif self.n_iter_ == self.max_iter - 1:
                warnings.warn(
                    "Maximum number of iteration reached before "
                    "convergence. Consider increasing max_iter to "
                    "improve the fit.",
                    ConvergenceWarning,
                )

        # Set the resulting instance variables.
        if self.metric == "precomputed":
            self.cluster_centers_ = None
        else:
            self.cluster_centers_ = X[medoid_idxs]

        # Expose labels_ which are the assignments of
        # the training data to clusters
        self.labels_ = labels
        self.medoid_indices_ = medoid_idxs
        self.score_ = self._compute_score(D)

        # Return self to enable method chaining
        return self

    def _update_medoid_idxs_in_place(self, D, labels, medoid_idxs):
        """In-place update of the medoid indices"""

        # Update the medoids for each cluster
        for k in range(self.n_clusters):
            # Extract the distance matrix between the data points
            # inside the cluster k
            cluster_k_idxs = np.where(labels == k)[0]

            if len(cluster_k_idxs) == 0:
                warnings.warn(
                    "Cluster {k} is empty! "
                    "self.labels_[self.medoid_indices_[{k}]] "
                    "may not be labeled with "
                    "its corresponding cluster ({k}).".format(k=k)
                )
                continue

            in_cluster_distances = D[
                cluster_k_idxs, cluster_k_idxs[:, np.newaxis]
            ]

            # Calculate all costs from each point to all others in the cluster
            in_cluster_all_costs = np.sum(in_cluster_distances, axis=1)

            min_cost_idx = np.argmin(in_cluster_all_costs)
            min_cost = in_cluster_all_costs[min_cost_idx]
            curr_cost = in_cluster_all_costs[
                np.argmax(cluster_k_idxs == medoid_idxs[k])
            ]

            # Adopt a new medoid if its distance is smaller then the current
            if min_cost < curr_cost:
                medoid_idxs[k] = cluster_k_idxs[min_cost_idx]

    def _compute_score(self, distances):
        """
        Computes the score of a k-clustering with the Sum of the Squared distances to the cluster centers xi
        sum(i=0, k, sum(j in cluster i, dist(j, xi)^2))

        Input:
        distance matrix for the n points
        labels_ and medoid_indices_ have to be set
        """
        score = 0
        for i in range(self.n_clusters):
            in_cluster_distances = distances[self.labels_ == i, self.medoid_indices_[i]]
            score += np.sum(in_cluster_distances**2)
        return score

    def _init_centers(self, D, n_clusters, old_centers):
        """
        Use old centers as a starting point.
        If less than n_clusters are given
        add centers with biggest squared distance to other clusters centers.
        If more than n_clusters are given
        delete centers from smallest clusters if more than n_clusters.
        """
        if old_centers is None or len(old_centers) == 0:
            raise ValueError("Old centers should be a nonempty list")

        if len(old_centers) == n_clusters:
            return old_centers
        elif len(old_centers) > n_clusters:
            return old_centers[:n_clusters]
        else:
            centers = old_centers.copy()
            while len(centers) < n_clusters:
                # Biggest MSD
                new = np.argmax(np.sum(D[np.ix_([i for i in range(D.shape[1]) if i not in centers], centers)], axis=1)**2)
                # Include offset
                centers += [new + len([c for c in centers if c < new])]
            return centers


