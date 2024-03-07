import numpy as np
from scipy import stats
from typing import Tuple, Dict


def make_real(loc: float, scale: float, shape: Tuple[int, int], seed: int, *args, **kwargs) -> Tuple[np.ndarray, Dict[int, str]]:
    data = stats.norm.rvs(loc=loc, scale=scale, size=shape, random_state=seed)
    domains = {m: 'continuous' for m in range(shape[1])}
    return data, domains


def make_positive(a: float, scale: float, shape: Tuple[int, int], seed: int, *args, **kwargs) -> Tuple[np.ndarray, Dict[int, str]]:
    data = stats.gamma.rvs(a=a, scale=scale, size=shape, random_state=seed)
    domains = {m: 'positive' for m in range(shape[1])}
    return data, domains


def make_lognormal(loc: float, scale: float, shape: Tuple[int, int], seed: int, *args, **kwargs) -> Tuple[np.ndarray, Dict[int, str]]:
    data = stats.lognorm.rvs(loc=loc, s=scale, size=shape, random_state=seed)
    domains = {m: 'positive' for m in range(shape[1])}
    return data, domains


def make_interval(a: float, b: float, theta: float, shape: Tuple[int, int], seed: int, *args, **kwargs) -> Tuple[np.ndarray, Dict[int, str]]:
    data = theta * stats.beta.rvs(a=a, b=b, size=shape)
    domains = {m: 'continuous' for m in range(shape[1])}
    return data, domains


def make_categorical(n_classes: int, shape: Tuple[int, int], seed: int, *args, **kwargs) -> Tuple[np.ndarray, Dict[int, str]]:
    n_samples, n_features = shape
    class_probs = stats.dirichlet.rvs(
        np.ones(n_classes), size=n_samples, random_state=seed
    )
    class_assignments = []
    for n in range(n_samples):
        class_seed = seed + n if seed is not None else None
        class_assignments.append(
            stats.multinomial.rvs(n=1, p=class_probs[n], random_state=class_seed)
        )
    class_assignments = np.array(class_assignments).argmax(axis=1)
    centroids = stats.norm.rvs(
        loc=0, scale=10, size=(n_classes, n_features - 1), random_state=seed
    )
    features = stats.norm.rvs(
        loc=centroids[class_assignments], scale=1, random_state=seed
    )
    data = np.column_stack((features, class_assignments))
    domains = {}
    for m in range(n_features - 1):
        domains[m] = 'continuous'
    domains[n_features - 1] = 'discrete'
    return data, domains


def make_ordinal(n_classes: int, shape: Tuple[int, int], seed: int, *args, **kwargs) -> Tuple[np.ndarray, Dict[int, str]]:
    n_samples, n_features = shape
    y = stats.uniform.rvs(
        loc=0, scale=n_classes, size=n_samples, random_state=seed
    )
    thetas = stats.uniform.rvs(
        loc=0, scale=n_classes - 1, size=n_classes - 1, random_state=seed
    )
    thetas = np.sort(thetas)
    thetas = np.insert(thetas, 0, -np.inf)
    thetas = np.append(thetas, np.inf)
    x = np.empty((n_samples,), dtype=int)
    for i in range(1, n_classes + 1):
        x[(thetas[i - 1] <= y) & (y < thetas[i])] = i - 1
    centroids = stats.norm.rvs(
        loc=0, scale=10, size=(n_classes, n_features - 2), random_state=seed
    )
    features = stats.norm.rvs(loc=centroids[x], scale=1, random_state=seed)
    data = np.column_stack((features, y, x))
    domains = {}
    for m in range(n_features - 2):
        domains[m] = 'continuous'
    domains[n_features - 2] = 'positive'
    domains[n_features - 1] = 'discrete'
    return data, domains


def make_count(a: float, shape: Tuple[int, int], seed: int, *args, **kwargs) -> Tuple[np.ndarray, Dict[int, str]]:
    n_samples, n_features = shape

    # sample latent variable
    correlated_feature = stats.gamma.rvs(a=a, loc=a / 4, size=n_samples, random_state=seed)

    # derive x by flooring
    count_variable = np.floor(correlated_feature).astype(int)
    n_classes = count_variable.max().astype(int) + 1

    # class centroids
    centroids = []
    for i in range(n_classes):
        class_seed = seed + i if seed is not None else None
        centroids.append(stats.norm.rvs(
            scale=10, size=n_features - 2, random_state=class_seed
        ))
    centroids = np.array(centroids)

    # features
    features = []
    for i in range(shape[0]):
        seed = seed + i if seed is not None else None
        features.append(stats.norm.rvs(
            loc=centroids[count_variable[i] - 1], size=n_features - 2, random_state=seed
        ))
    features = np.array(features)

    # final dataset
    data = np.column_stack((features, correlated_feature, count_variable))
    domains = {}
    for m in range(n_features - 2):
        domains[m] = 'continuous'
    domains[n_features - 2] = 'positive'
    domains[n_features - 1] = 'discrete'
    return data, domains


def make_poisson(w: float, shape: Tuple[int, int], seed: int, *args, **kwargs) -> Tuple[np.ndarray, Dict[int, str]]:
    n_samples, n_features = shape

    # sampling
    features = stats.norm.rvs(
        loc=0, scale=1, size=(n_samples, n_features - 2), random_state=seed
    )
    correlated_feature = stats.uniform.rvs(
        loc=0, scale=1, size=n_samples, random_state=seed
    )
    count_variable = stats.poisson.rvs(
        np.exp(w * correlated_feature), random_state=seed
    )
    data = np.column_stack((features, correlated_feature, count_variable))
    domains = {}
    for m in range(n_features - 2):
        domains[m] = 'continuous'
    domains[n_features - 2] = 'positive'
    domains[n_features - 1] = 'discrete'
    return data, domains
