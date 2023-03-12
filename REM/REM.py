"""Gaussian Mixture Model."""

# Modified code from Scikit-Learn
import math
import time
import numpy as np
from matplotlib import transforms
from scipy.spatial.distance import mahalanobis
from numpy.linalg import eig
from scipy import linalg
from queue import PriorityQueue
import pandas as pd
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
import scipy.spatial.distance as distance
from sklearn.neighbors import KernelDensity, KDTree
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
from .overlap import Overlap

from . import GaussianMixture


###############################################################################
# Input parameter checkers used by the REM class
def density_broad_search_star(a_b):
    try:
        return euclidean_distances(a_b[1], a_b[0])
    except Exception as e:
        raise Exception(e)


def _estimate_density_distances(X, bandwidth):
    n_samples, n_features = X.shape

    if bandwidth == "spherical":

        center = X.sum(0) / n_samples

        X_centered = X - center

        covariance_data = np.einsum('ij,ki->jk', X_centered, X_centered.T) / (n_samples - 1)

        bandwidth = 1 / (100 * n_features) * np.trace(covariance_data)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)

        density = kde.score_samples(X)

    elif bandwidth == "diagonal":
        bandwidths = np.array([0.01 * np.std(X[:, i]) for i in range(n_features)])

        var_type = 'c' * n_features

        dens_u = sm.nonparametric.KDEMultivariate(data=X, var_type=var_type, bw=bandwidths)

        density = dens_u.pdf(X)

    elif bandwidth == "normal_reference":
        var_type = 'c' * n_features

        dens_u = sm.nonparametric.KDEMultivariate(data=X, var_type=var_type, bw='normal_reference')

        density = dens_u.pdf(X)

    elif isinstance(bandwidth, int):
        kdt = KDTree(X, metric='euclidean')

        distances, neighbors = kdt.query(X, int(bandwidth))

        density = 1 / distances[:, int(bandwidth) - 1]

    elif isinstance(bandwidth, float):
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)

        density = kde.score_samples(X)

    kdt = KDTree(X, metric='euclidean')

    distances, neighbors = kdt.query(X, np.floor(np.sqrt(n_samples)).astype(int))

    best_distance = np.empty((X.shape[0]))

    radius_diff = density[:, np.newaxis] - density[neighbors]

    rows, cols = np.where(radius_diff < 0)

    rows, unidx = np.unique(rows, return_index=True)

    cols = cols[unidx]

    best_distance[rows] = distances[rows, cols]

    search_idx = list(np.setdiff1d(list(range(X.shape[0])), rows))

    search_density = density[search_idx]

    GT_radius = density > search_density[:, np.newaxis]

    if any(np.sum(GT_radius, axis=1) == 0):
        max_i = [i for i in range(GT_radius.shape[0]) if np.sum(GT_radius[i, :]) == 0]

        if len(max_i) > 1:
            for max_j in max_i[1:len(max_i)]:
                GT_radius[max_j, search_idx[max_i[0]]] = True

        max_i = max_i[0]

        best_distance[search_idx[max_i]] = np.sqrt(((X - X[search_idx[max_i], :]) ** 2).sum(1)).max()

        GT_radius = np.delete(GT_radius, max_i, 0)

        del search_idx[max_i]

    GT_distances = ([X[search_idx[i], np.newaxis], X[GT_radius[i, :], :]] for i in range(len(search_idx)))

    distances_bb = list(map(density_broad_search_star, list(GT_distances)))

    argmin_distance = [np.argmin(l) for l in distances_bb]

    for i in range(GT_radius.shape[0]):
        best_distance[search_idx[i]] = distances_bb[i][argmin_distance[i]]

    return density, best_distance


def _create_decision_plots(density, distance):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(density, distance, s=5, alpha=1, color='black')
    plt.xlabel("Log of Density")
    plt.ylabel("Distance to Neighbor of Higher Density")
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(density)), np.sort(density * distance), s=5, alpha=1, color='black')
    plt.xlabel("Index")
    plt.ylabel("Product of Density and Distance")
    plt.tight_layout()
    plt.show()


def _plot_criterion_score(queue, mixtures, line_style, label):
    criterion_scores = queue.queue
    criterion_scores.sort(key=lambda x: x[1], reverse=True)
    x_values = []
    y_values = []
    for i in criterion_scores:
        y_values.append(i[0])
        x_values.append(mixtures[i[1]].n_components)
    plt.plot(x_values, y_values, line_style, label=label)


def _select_exemplars_from_thresholds(X, density, distance, density_threshold, distance_threshold):
    density_inlier = density > density_threshold
    distance_inlier = distance > distance_threshold
    means_idx = np.where(density_inlier * distance_inlier)[0]
    remainder_idx = np.where(~(density_inlier * distance_inlier))[0]
    means = X[means_idx, :]
    X_iter = X[remainder_idx, :]
    print("%s means selected." % means.shape[0])
    return X_iter, means


def _select_exemplars_fromK(X, density, distance, max_components):
    n_samples, _ = X.shape
    means_idx = np.argsort(- density * distance)[range(max_components)]
    remainder_idx = np.argsort(- density * distance)[range(max_components, n_samples)]
    means = X[means_idx, :]
    X_iter = X[remainder_idx, :]
    print("%s means selected." % means.shape[0])
    return X_iter, means


def _initialize_covariances(X, means, covariance_type):
    n_samples, n_features = X.shape

    n_components = means.shape[0]

    center = X.sum(0) / n_samples

    X_centered = X - center

    covariance_data = np.einsum('ij,ki->jk', X_centered, X_centered.T) / (n_samples - 1)

    variance = 1 / (n_components * n_features) * np.trace(covariance_data)

    if covariance_type == "full":
        covariances = np.stack([np.diag(np.ones(n_features) * variance) for _ in range(n_components)])
    elif covariance_type == "spherical":
        covariances = np.repeat(variance, n_components)
    elif covariance_type == "tied":
        covariances = np.diag(np.ones(n_features) * variance)
    elif covariance_type == "diag":
        covariances = np.ones((n_components, n_features)) * variance
    return covariances


def _get_mixture_score(X, mixture_scores, scoring_function):
    mixture_scores.append(scoring_function(X))


def _get_intervals(n_samples, ltidx, gtidx, ranges):
    raw_intervals = []
    union_intervals = []
    for s in range(n_samples):
        raw_intervals.append([])
        union_intervals.append([])
        for t in ltidx:
            raw_intervals[s].append((-np.inf, ranges[s, t]))
        for t in gtidx:
            raw_intervals[s].append((ranges[s, t], np.inf))
        for begin, end in sorted(raw_intervals[s]):
            if union_intervals[s] and union_intervals[s][-1][1] >= begin - 1:
                union_intervals[s][-1][1] = max(union_intervals[s][-1][1], end)
            else:
                union_intervals[s].append([begin, end])
    return [item for sublist in union_intervals for item in sublist]



def _print_mixing_proportions(weights):
    print("Mixing proportions:")
    format_row = "{:>22}" * (len(weights))
    print(format_row.format(*[i for i in range(len(weights))]))
    print(format_row.format(*weights) + "\n")


def _print_means(means):
    print("Means:")
    format_row = "{:>22}" * (len(means[0]) + 1)
    labels = [""] + ["[,{}]".format(i) for i in range(len(means[0]))]
    print(format_row.format(*labels))
    for i, j in enumerate(means):
        label = "[{},]".format(i)
        print(format_row.format(label, *j))
    print()


def _print_covariances(covariances, covariance_type, n_features):
    expanded_covariances = _expand_covariance_matrix(covariances, covariance_type, n_features)
    print("Variances:")
    format_row = "{:>22}" * (len(expanded_covariances[0][0]) + 1)
    for i, j in enumerate(expanded_covariances):
        label = "[{},,]".format(i)
        print(label)
        labels = [""] + ["[,{}]".format(i) for i in range(len(j[0]))]
        print(format_row.format(*labels))
        for k, l in enumerate(j):
            label = "[{},]".format(k)
            print(format_row.format(label, *l))
    print()


def _print_parameter_info(mixture):
    _print_mixing_proportions(mixture.weights_)
    _print_means(mixture.means_)
    _, n_features = mixture.means_.shape
    _print_covariances(mixture.covariances_, mixture.covariance_type, n_features)


def _expand_covariance_matrix(covariances, covariance_type, n_features):
    if covariance_type == 'spherical':
        return np.array([np.diag(np.ones(n_features) * i) for i in covariances])
    elif covariance_type == 'diag':
        return np.array([np.diag(i) for i in covariances])
    else:
        return covariances


def _draw_ellipse(covariances, means, ax):
    pearson = covariances[0, 1] / np.sqrt(covariances[0, 0] * covariances[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='none', edgecolor='grey')
    scale_x = np.sqrt(covariances[0, 0])
    scale_y = np.sqrt(covariances[1, 1])
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(means[0], means[1])
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


class REM:
    def __init__(
            self,
            *,
            data,
            covariance_type="full",
            criteria="all",
            bandwidth="diagonal",
            tol=1e-3,
            reg_covar=1e-6,
            max_iter=100,
    ):
        self.data = data
        self.fitted = False
        self.t1 = None
        self.t2 = None
        self.covariance_type = covariance_type
        self.criteria = criteria
        self.bandwidth = bandwidth
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self._distance = None
        self._density = None
        self._density_threshold = None
        self._max_components = None
        self._distance_threshold = None

    def _check_parameters(self):
        """Check the Gaussian mixture parameters are well defined."""
        _, n_features = self.data.shape
        if self.covariance_type not in ["spherical", "tied", "diag", "full"]:
            raise ValueError(
                "Invalid value for 'covariance_type': %s "
                "'covariance_type' should be in "
                "['spherical', 'tied', 'diag', 'full']"
                % self.covariance_type
            )

        if self.criteria not in ["all", "aic", "bic", "icl"]:
            raise ValueError(
                "Invalid value for 'criteria': %s "
                "'criteria' should be in "
                "['none', 'aic', 'bic', 'icl']"
                % self.criteria
            )

        if self.bandwidth not in ["diagonal", "spherical", "normal_reference"] and \
                not isinstance(self.bandwidth, int) and not isinstance(self.bandwidth, float):
            raise ValueError(
                "Hello Invalid value for 'bandwidth': %s"
                "'bandwidth' should be 'diagonal', 'spherical', 'normal_reference' or a float."
                % self.bandwidth
            )

    def _print_threshold_parameter_warning(self):
        if self._density_threshold is None:
            message_start = 'Distance threshold provided, but no density threshold provided.'
        else:
            message_start = 'Density threshold provided, but no distance threshold provided.'
        warnings.warn(message_start + ' Clustering will continue with the max number of components set to ' + str(
            self._max_components))

    def _select_exemplars(self):
        if self._density_threshold is not None and self._distance_threshold is not None:
            return _select_exemplars_from_thresholds(self.data, self._density, self._distance, self._density_threshold,
                                                     self._distance_threshold)
        else:
            if self._density_threshold is not None or self._distance_threshold is not None:
                self._print_threshold_parameter_warning()
            return _select_exemplars_fromK(self.data, self._density, self._distance, self._max_components)

    def _get_exemplars(self):
        if self._density is None and self._distance is None:
            self._density, self._distance = _estimate_density_distances(self.data, self.bandwidth)
        return self._select_exemplars()

    def _add_mixture_to_pq(self, new_mixture):
        if self.criteria == 'aic' or self.criteria == 'all':
            self.aics_.put((new_mixture.aic(self.data), self.n_mixtures))
        if self.criteria == 'bic' or self.criteria == 'all':
            self.bics_.put((new_mixture.bic(self.data), self.n_mixtures))
        if self.criteria == 'icl' or self.criteria == 'all':
            self.icls_.put((new_mixture.icl(self.data), self.n_mixtures))

    def _add_mixture(self):
        new_mixture = GaussianMixture.GaussianMixture(n_components=self.n_components_iter, weights=self.weights_iter,
                                                      means=self.means_iter, covariances=self.covariances_iter,
                                                      covariance_type=self.covariance_type, max_iter=self.max_iter,
                                                      tol=self.tol, reg_covar=self.reg_covar).fit(self.X_iter)
        self._add_mixture_to_pq(new_mixture)
        self.mixtures.append(new_mixture)
        self.n_mixtures += 1

    def _initialize_parameters(self):
        """Initialization of the Gaussian mixture exemplars from a decision plot.
        
        Parameters
        ----------
        self.data : array-like of shape (n_samples, n_features)
        
        """
        self.X_iter, self.means_iter = self._get_exemplars()
        self.n_components_iter = self.means_iter.shape[0]
        self.covariances_iter = _initialize_covariances(self.data, self.means_iter, self.covariance_type)
        self.weights_iter = np.ones((self.n_components_iter)) / self.n_components_iter
        self._add_mixture()

    def _prune_exemplar(self):
        n_samples, covariances_logdet_penalty, overlap_max, distances, theta = self._get_pruning_parameters()
        if theta is None:
            return True
        Ob = distances + covariances_logdet_penalty + (theta * overlap_max)
        s_min = Ob.argmin(1)
        resps = np.zeros((n_samples, self.n_components_iter))
        resps[range(resps.shape[0]), s_min] = 1
        self.return_refined(resps)
        return False

    def _get_pruning_parameters(self):
        n_samples, n_features = self.X_iter.shape
        covariances_logdet_penalty, expanded_covariance_iter = self._alter_covariances(n_features, n_samples)
        overlap_max = self._compute_overlap(n_features, expanded_covariance_iter)
        distances = self._get_distances(n_samples, expanded_covariance_iter)
        theta = self._get_theta(distances, covariances_logdet_penalty, overlap_max)
        return n_samples, covariances_logdet_penalty, overlap_max, distances, theta

    def _alter_covariances(self, n_features, n_samples):
        self.covariances_iter += np.ones(self.covariances_iter.shape) * 1e-6
        expanded_covariance_iter = _expand_covariance_matrix(self.covariances_iter, self.covariance_type, n_features)
        covariances_logdet_penalty = np.array(
            [np.log(np.linalg.det(expanded_covariance_iter[i])) for i in range(self.n_components_iter)]) / n_samples
        return covariances_logdet_penalty, expanded_covariance_iter

    def _compute_overlap(self, n_features, cov):
        covariances_jitter = self._update_covariance_for_overlap(n_features, cov)
        return self._get_omega_map(n_features, covariances_jitter)

    def _update_covariance_for_overlap(self, n_features, cov):
        covariances_jitter = np.zeros(cov.shape)
        for i in range(self.n_components_iter):
            val, vec = np.linalg.eig(cov[i])
            val += np.abs(np.random.normal(loc=0, scale=0.01, size=n_features))
            covariances_jitter[i, :, :] = vec.dot(np.diag(val)).dot(np.linalg.inv(vec))
        return covariances_jitter

    def _get_omega_map(self, n_features, covariances_jitter):
        while True:
            n_components, _, _ = covariances_jitter.shape
            omega_map = Overlap(n_features, n_components, self.weights_iter, self.means_iter, covariances_jitter,
                                np.array([1e-06, 1e-06]), 1e06).omega_map
            if np.max(omega_map.max(1)) > 0:
                break
            else:
                covariances_jitter *= 1.1
        return omega_map.max(1)

    def _get_distances(self, n_samples, covariances):
        distances = np.zeros((n_samples, self.n_components_iter))
        for j in range(self.n_components_iter):
            distances[:, j, np.newaxis] = distance.cdist(self.X_iter, self.means_iter[j, :][np.newaxis],
                                                         metric='mahalanobis', VI=covariances[j, :, :])
        return distances

    def _get_theta(self, distances, covariances_logdet_penalty, overlap_max):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._compute_theta(distances, covariances_logdet_penalty, overlap_max)

    def _compute_theta(self, distances, covariances_logdet_penalty, overlap_max):
        n_samples, _ = self.X_iter.shape
        thetas = np.ones(self.n_components_iter) * np.nan
        entry = False
        for i in range(self.n_components_iter):
            p = distances + covariances_logdet_penalty - (distances[:, i] + covariances_logdet_penalty[i])[:,
                                                         np.newaxis]
            ranges = p / (overlap_max[i] - overlap_max)
            noni_idx = list(range(self.n_components_iter))
            noni_idx.pop(i)
            overlap_noni = overlap_max[noni_idx]
            ranges = ranges[:, noni_idx]
            ltidx = np.where(overlap_max[i] < overlap_noni)[0]
            gtidx = np.where(overlap_max[i] > overlap_noni)[0]
            union_intervals = _get_intervals(n_samples, ltidx, gtidx, ranges)
            start, end = None, None
            while union_intervals:
                start_temp, end_temp = union_intervals.pop()
                start = start_temp if start is None else max(start, start_temp)
                end = end_temp if end is None else min(end, end_temp)
            if start is not None and end is not None and end > start > 0:
                entry = True
                thetas[i] = start
        if not entry:
            return None
        theta = thetas[~np.isnan(thetas)].min()
        return theta * 1.0001

    def return_refined(self, resps):
        self._update_weights(resps)
        self._add_exemplar_to_data()
        self._remove_pruned_mean()
        self._remove_pruned_covariance()
        self._remove_pruned_component()

    def _update_weights(self, resps):
        self.weights_iter = resps.sum(0) / resps.shape[0]
        self.weights_iter[self.weights_iter < 0.00001] = 0

    def _add_exemplar_to_data(self):
        rm_means = self.means_iter[self.weights_iter == 0, :]
        if rm_means.ndim == 1:
            self.X_iter = np.append(self.X_iter, rm_means[:, np.newaxis], axis=0)
        else:
            self.X_iter = np.append(self.X_iter, rm_means, axis=0)

    def _remove_pruned_mean(self):
        self.means_iter = self.means_iter[self.weights_iter != 0, :]

    def _remove_pruned_covariance(self):
        if self.covariance_type != 'tied':
            self.covariances_iter = self.covariances_iter[self.weights_iter != 0]

    def _remove_pruned_component(self):
        self.weights_iter = self.weights_iter[self.weights_iter != 0]
        self.n_components_iter = len(self.weights_iter)

    def _update_mixture(self):
        self._add_mixture()
        self.weights_iter = self.mixtures[-1].weights_
        self.covariances_iter = self.mixtures[-1].covariances_

    def _iterative_REM_procedure(self):
        while self.n_components_iter > 1:
            if self._prune_exemplar():
                break
            self._update_mixture()

    def _set_optimal_mixture(self):
        if self.criteria == "aic" or self.criteria == "all":
            self.aic_mixture = self.mixtures[self.aics_.queue[0][1]]
        if self.criteria == "bic" or self.criteria == "all":
            self.bic_mixture = self.mixtures[self.bics_.queue[0][1]]
        if self.criteria == "icl" or self.criteria == "all":
            self.icl_mixture = self.mixtures[self.icls_.queue[0][1]]

    def exemplars_plot(self):
        self._density, self._distance = _estimate_density_distances(self.data, self.bandwidth)
        _create_decision_plots(self._density, self._distance)

    def _print_score_table(self, rating_type, queue):
        print(str.upper(rating_type) + " scores:")
        print("{:<25} {:<15}".format("Number of components", str.upper(rating_type)))
        for i in queue.queue:
            mixture = self.mixtures[i[1]]
            n_components = mixture.n_components
            print("{:<25} {:<15}".format(n_components, i[0]))
        print()

    def _print_classification(self, mixture):
        print("Clustering Table:")
        ys = mixture.predict(self.data)
        unique_ys = np.unique(ys)
        unique_ys.sort()
        format_row = "{:>4}" * (len(unique_ys))
        print(format_row.format(*unique_ys))
        cluster_count = np.array([np.sum(ys == i) for i in unique_ys])
        print(format_row.format(*cluster_count))
        print()

    def _print_top_mixture_info(self, mixture, score_type, queue):
        print("REM " + self.covariance_type + " model with " + str(mixture.n_components) + " components.")
        print()
        print("{:<25} {:<5} {:<15}".format("Log-Likelihood", "n", str.upper(score_type)))
        print("{:<25} {:<5} {:<15}".format(mixture.score(self.data), np.shape(self.data)[0], queue.queue[0][0]))
        print()

    def print_summary(self, score_type, queue, mixture, parameters, classification, scores):
        if scores:
            self._print_score_table(score_type, queue)
        self._print_top_mixture_info(mixture, score_type, queue)
        if classification:
            self._print_classification(mixture)
        if parameters:
            _print_parameter_info(mixture)

    def summary(self, parameters=False, classification=False, criterion_scores=False):
        if not self.fitted:
            raise Exception("Model yet to be fitted")
        if self.criteria == "aic" or self.criteria == "all":
            self.print_summary("aic", self.aics_, self.aic_mixture, parameters=parameters,
                               classification=classification, scores=criterion_scores)
        if self.criteria == "bic" or self.criteria == "all":
            self.print_summary("bic", self.bics_, self.bic_mixture, parameters=parameters,
                               classification=classification, scores=criterion_scores)
        if self.criteria == "icl" or self.criteria == "all":
            self.print_summary("icl", self.icls_, self.icl_mixture, parameters=parameters,
                               classification=classification, scores=criterion_scores)

    def _get_selected_mixture(self, mixture_selection):
        if self.criteria == "aic" or (self.criteria == "all" and str.lower(mixture_selection) == "aic"):
            return self.aic_mixture
        elif self.criteria == "bic" or (self.criteria == "all" and str.lower(mixture_selection) == "bic"):
            return self.bic_mixture
        elif self.criteria == "icl" or (self.criteria == "all" and str.lower(mixture_selection) == "icl"):
            return self.icl_mixture
        elif self.criteria == "all" and mixture_selection == "":
            raise Exception("No model selected to plot")
        else:
            raise Exception("Cannot plot " + mixture_selection + " selected mixture as the selection criteria was set "
                            + "to " + self.criteria)

    def classification_plot(self, mixture_selection='', dimensions=None, axis_labels=None):
        if self.fitted:
            self._class_style_plot(mixture_selection, dimensions, self._draw_classification, axis_labels)
        else:
            raise Exception("Model yet to be fitted")

    def uncertainty_plot(self, mixture_selection='', dimensions=None, axis_labels=None):
        if self.fitted:
            self._class_style_plot(mixture_selection, dimensions, self._draw_uncertainty, axis_labels)
        else:
            raise Exception("Model yet to be fitted")

    def density_plot(self, dimensions=None, axis_labels=None):
        if dimensions is not None and not isinstance(dimensions, list):
            raise Exception("\"dimensions\" must be a list of integers.")
        if dimensions is None:
            dimensions = list(range(self.data.shape[1]))
        if axis_labels is not None and not isinstance(axis_labels, list):
            raise Exception("\"axis_labels\" must be a list of strings.")
        if axis_labels is not None and len(axis_labels) != len(dimensions):
            warnings.warn("Number of axis labels is not equal to the number of dimensions plotted. Provided labels will"
                          " not be used")
            axis_labels = None
        if len(dimensions) == 2:
            self._2d_density_plot(dimensions)
            return
        fig, axs = plt.subplots(len(dimensions), len(dimensions), figsize=(15, 8))
        for axis_i, i in enumerate(dimensions):
            for axis_j, j in enumerate(dimensions):
                try:
                    if i != j:
                        density_data = pd.DataFrame({"x": self.data[:, i], "y": self.data[:, j]})
                        sns.kdeplot(density_data, ax=axs[axis_j, axis_i], x="x", y="y", warn_singular=False)
                    else:
                        self._empty_plot(i, axis_i, axs[axis_i, axis_i], axis_labels)
                except:
                    raise Exception(
                        "An index entered in \"dimensions\" was not valid. Ensure all indexes entered are between 0 "
                        "and the number of features - 1.")
        for ax in fig.get_axes():
            ax.label_outer()
        plt.show()

    def _2d_density_plot(self, dimensions):
        try:
            density_data = pd.DataFrame({"x": self.data[:, dimensions[0]], "y": self.data[:, dimensions[1]]})
            sns.kdeplot(density_data, x="x", y="y", warn_singular=False)
        except:
            raise Exception(
                "An index entered in \"dimensions\" was not valid. Ensure all indexes entered are between 0 "
                "and the number of features - 1.")
        plt.show()

    def _class_style_plot(self, mixture_selection, dimensions, plotting_function, axis_labels):
        if dimensions is not None and not isinstance(dimensions, list):
            raise Exception("\"dimensions\" must be a list of integers.")
        if dimensions is None:
            dimensions = list(range(self.data.shape[1]))
        if axis_labels is not None and not isinstance(axis_labels, list):
            raise Exception("\"axis_labels\" must be a list of strings.")
        if axis_labels is not None and len(axis_labels) != len(dimensions):
            warnings.warn("Number of axis labels is not equal to the number of dimensions plotted. Provided labels will"
                          " not be used")
            axis_labels = None
        _, n_features = self.X_iter.shape
        mixture = self._get_selected_mixture(mixture_selection)
        if len(dimensions) == 2:
            self._2d_class_plot(mixture, plotting_function, dimensions, n_features, axis_labels)
            return
        labels = mixture.predict(self.data)
        fig, axs = plt.subplots(len(dimensions), len(dimensions), figsize=(15, 8))
        expanded_covariances = _expand_covariance_matrix(mixture.covariances_, self.covariance_type, n_features)
        for axis_i, i in enumerate(dimensions):
            for axis_j, j in enumerate(dimensions):
                if i != j:
                    plotting_function(i, j, axs[axis_j, axis_i], labels, mixture)
                    for k in range(mixture.n_components):
                        _draw_ellipse(np.array([[expanded_covariances[k][i][i], expanded_covariances[k][i][j]],
                                                [expanded_covariances[k][j][i], expanded_covariances[k][j][j]]]),
                                      [mixture.means_[k][i], mixture.means_[k][j]],
                                      axs[axis_j, axis_i])
                        axs[axis_j, axis_i].scatter(mixture.means_[k][i], mixture.means_[k][j], c="black", s=10)
                else:
                    self._empty_plot(i, axis_i, axs[axis_i, axis_i], axis_labels)
        for ax in fig.get_axes():
            ax.label_outer()
        plt.show()

    def _2d_class_plot(self, mixture, plotting_function, dimensions, n_features, axis_labels):
        labels = mixture.predict(self.data)
        fig, ax = plt.subplots()
        plotting_function(dimensions[0], dimensions[1], ax, labels, mixture)
        expanded_covariances = _expand_covariance_matrix(mixture.covariances_, self.covariance_type, n_features)
        for i in range(mixture.n_components):
            _draw_ellipse(np.array([[expanded_covariances[i][dimensions[0]][dimensions[0]],
                                     expanded_covariances[i][dimensions[0]][dimensions[1]]],
                                    [expanded_covariances[i][dimensions[1]][dimensions[0]],
                                     expanded_covariances[i][dimensions[1]][dimensions[1]]]]),
                          [mixture.means_[i][dimensions[0]], mixture.means_[i][dimensions[1]]],
                          ax)
            ax.scatter(mixture.means_[i][dimensions[0]], mixture.means_[i][dimensions[1]], c="black", s=10)
        plt.show()

    def _empty_plot(self, index, ax_index, ax, axis_labels):
        label = "C" + str(index) if axis_labels is None else axis_labels[ax_index]
        ax.scatter(self.data[:, index], self.data[:, index], marker="none")
        ax.text(0.5, 0.5, label, horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, fontsize=20)

    def _draw_uncertainty(self, x_index, y_index, ax, labels, mixture):
        uncertainty = mixture.predict_uncertainties(self.data)
        try:
            subplot_data = pd.DataFrame(
                {"x": self.data[:, x_index], "y": self.data[:, y_index], "uncertainty": uncertainty,
                 "labels": labels})
        except:
            raise Exception(
                "An index entered in \"dimensions\" was not valid. Ensure all indexes entered are between 0 "
                "and the number of features - 1.")
        groups = subplot_data.groupby("labels")
        for _, group in groups:
            ax.scatter(group.x, group.y, s=group.uncertainty)

    def _draw_classification(self, x_index, y_index, ax, labels, mixture):
        subplot_data = pd.DataFrame({"x": self.data[:, x_index], "y": self.data[:, y_index], "labels": labels})
        groups = subplot_data.groupby("labels")
        for _, group in groups:
            ax.scatter(group.x, group.y, s=1.5)

    def criterion_plot(self):
        if not self.fitted:
            raise Exception("model yet to be fitted")
        plt.figure(figsize=(15, 6))
        if self.criteria == "aic" or self.criteria == "all":
            _plot_criterion_score(self.aics_, self.mixtures, "-ro", "AIC")
        if self.criteria == "bic" or self.criteria == "all":
            _plot_criterion_score(self.bics_, self.mixtures, "-bo", "BIC")
        if self.criteria == "icl" or self.criteria == "all":
            _plot_criterion_score(self.icls_, self.mixtures, "-go", "ICL")
        plt.xlabel("Number of Components")
        plt.ylabel("Criterion Score")
        plt.legend()
        plt.show()

    def fit(self, max_components=5, density_threshold=None, distance_threshold=None):
        self.t1 = time.perf_counter()
        self.weights_iter = None
        self.covariances_iter = None
        self.mixtures = []
        self.aics_ = PriorityQueue()
        self.bics_ = PriorityQueue()
        self.icls_ = PriorityQueue()
        self.n_mixtures = 0
        self._max_components = max_components
        self._density_threshold = density_threshold
        self._distance_threshold = distance_threshold
        self.fit_predict()
        self.fitted = True
        self.t2 = time.perf_counter()

        return self

    def fit_predict(self):
        self._initialize_parameters()
        self._iterative_REM_procedure()
        self._set_optimal_mixture()
