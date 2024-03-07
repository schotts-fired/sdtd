import numpy as np
from sdtd.gibbs.general_functions import (
    RandomNumberGenerator,
    f_int_inv, f_pos_inv, f_real_inv, g_inv,
    log_pdf_x_int, log_pdf_x_pos, log_pdf_x_real,
    log_pmf_x_cat, log_pmf_x_ord, log_pmf_x_count,
    extended_log
)
from sdtd.data.util import Domain
import math
from scipy import stats, special
import pandas as pd
from typing import Dict
import os
from tqdm import tqdm
from time import time


class Discovery:
    """A class to represent the discovery algorithm for the SDTD model.

    Args:
        X (np.ndarray): The dataset to be analyzed.
        domains (Dict[int, Domain]): The domains of each feature.
        prior_alpha (np.ndarray): The prior alpha for the Dirichlet distribution. Must be specified for each feature.
        s2_b (float): The prior variance for B.
        s2_y (float): The prior variance for Y.
        s2_z (float): The prior variance for Z.
        s2_theta (float): The prior variance for theta.
        s2_u (float): The prior variance for the u term in the definition of the pseudo-variables x = f(y + u)
        K (int): The number of components for the Z parameters.
        n_iterations (int): The number of iterations to run the algorithm.
        seed (int): The seed for the random number generator.
        logging (bool, optional): Whether to log the intermediate values for all variables. Defaults to False.
    """

    def __init__(self, X: np.ndarray,
                 domains: Dict[int, Domain], prior_alpha: np.ndarray,
                 s2_b: float, s2_y: float, s2_z: float, s2_theta: float, s2_u: float,
                 K: int, n_iterations: int, seed: int, logging: bool = False, **kwargs):
        self._validate_inputs(X, domains, prior_alpha, s2_b, s2_y, s2_z, s2_theta, s2_u, K, n_iterations, seed, **kwargs)

        # Dataset
        self.X = X

        # Hyperparameters
        self.K = K
        self.s2_b = s2_b
        self.s2_y = s2_y
        self.s2_z = s2_z
        self.s2_theta = s2_theta
        self.s2_u = s2_u
        self.n_iterations = n_iterations
        self.s_y = math.sqrt(s2_y)

        # Random Number Generator
        self.rng = RandomNumberGenerator(seed)

        # Dataset attributes
        N, D = self.X.shape
        self.N, self.D = self.X.shape
        self.prior_alpha = prior_alpha
        self.domains = domains
        self.n_classes = {d: int(np.nanmax(self.X[:, d])) for d in range(D) if self.domains[d].is_discrete()}  # only needed for discrete data
        self.X_max = {d: np.nanmax(self.X[:, d]) for d in range(D) if self.domains[d].is_continuous()}
        self.X_min = {d: np.nanmin(self.X[:, d]) for d in range(D) if self.domains[d].is_continuous()}
        self.X_mean = {d: np.nanmean(self.X[:, d]) for d in range(D) if self.domains[d].is_continuous()}

        # Likelihood Hyperparameters
        self.w_real = {d: 2 / (self.X_max[d] - self.X_mean[d]) for d in range(D) if self.domains[d].is_continuous()}
        self.b_real = {d: self.X_mean[d] for d in range(D) if self.domains[d].is_continuous()}
        self.w_pos = {d: 2 / self.X_max[d] for d in range(D) if self.domains[d].is_positive_real()}
        self.w_int = 2
        self.w_count = {d: 2 / self.n_classes[d] for d in range(D) if self.domains[d].is_discrete()}
        epsilon = {d: (self.X_max[d] - self.X_min[d]) / 10000 for d in range(D) if self.domains[d].is_continuous()}
        self.theta_L = {d: self.X_min[d] - epsilon[d] for d in range(D) if self.domains[d].is_continuous()}
        self.theta_H = {d: self.X_max[d] + epsilon[d] for d in range(D) if self.domains[d].is_continuous()}

        # Variables
        self.W = np.zeros((self.D, 3))
        self.S = np.zeros((self.N, self.D), dtype=int)
        self.Z = np.zeros((self.K, self.N))
        self.Yreal = {d: np.zeros((self.N, )) for d in range(self.D) if self.domains[d].is_continuous()}
        self.Ypos = {d: np.zeros((self.N, )) for d in range(self.D) if self.domains[d].is_positive_real()}
        self.Yint = {d: np.zeros((self.N, )) for d in range(self.D) if self.domains[d].is_continuous()}
        self.Ybin = {d: np.zeros((self.N, )) for d in range(self.D) if self.domains[d].is_binary()}
        self.Ycat = {d: np.zeros((self.n_classes[d], self.N)) for d in range(self.D) if self.domains[d].is_discrete()}
        self.Yord = {d: np.zeros((self.N, )) for d in range(self.D) if self.domains[d].is_discrete()}
        self.Ycount = {d: np.zeros((self.N, )) for d in range(self.D) if self.domains[d].is_discrete()}
        self.Breal = {d: np.zeros((self.K, )) for d in range(self.D) if self.domains[d].is_continuous()}
        self.Bpos = {d: np.zeros((self.K, )) for d in range(self.D) if self.domains[d].is_positive_real()}
        self.Bint = {d: np.zeros((self.K, )) for d in range(self.D) if self.domains[d].is_continuous()}
        self.Bbin = {d: np.zeros((self.K, )) for d in range(self.D) if self.domains[d].is_binary()}
        self.Bcat = {d: np.zeros((self.K, self.n_classes[d])) for d in range(self.D) if self.domains[d].is_discrete()}
        self.Bord = {d: np.zeros((self.K, )) for d in range(self.D) if self.domains[d].is_discrete()}
        self.Bcount = {d: np.zeros((self.K, )) for d in range(self.D) if self.domains[d].is_discrete()}
        self.theta = {d: np.zeros((self.n_classes[d] - 1, )) for d in range(self.D) if self.domains[d].is_discrete()}  # 5 classes: -inf t1 t2 t3 t4 inf, i.e. R - 1 thresholds

        # Debugging
        self.logging = logging
        self._variables_saved = False

    # HELPERS

    def _save_state(self, array: np.ndarray, filename: str):
        VAR_DIR = 'variables'
        if not os.path.exists(VAR_DIR):
            os.makedirs(VAR_DIR)

        df = pd.DataFrame(array)
        if 'Y' in filename:
            df = df.T
        df.to_csv(f'{VAR_DIR}/{filename}', index=None, header=None)

    def _save_variables(self):
        if not self._variables_saved:
            assert self.D == 5
            assert self.domains[4].is_discrete()
            self._save_state(self.X, 'X.csv')
            self._save_state(self.W, 'W.csv')
            self._save_state(self.S + 1, 'S.csv')
            self._save_state(self.Z, 'Z.csv')
            for d in range(self.D):
                if self.domains[d].is_continuous():
                    self._save_state(self.Yreal[d], f'Yreal{d}.csv')
                    self._save_state(self.Yint[d], f'Yint{d}.csv')
                    self._save_state(self.Breal[d], f'Breal{d}.csv')
                    self._save_state(self.Bint[d], f'Bint{d}.csv')
                if self.domains[d].is_positive_real():
                    self._save_state(self.Ypos[d], f'Ypos{d}.csv')
                    self._save_state(self.Bpos[d], f'Bpos{d}.csv')
                if self.domains[d].is_binary():
                    self._save_state(self.Ybin[d], f'Ybin{d}.csv')
                    self._save_state(self.Bbin[d], f'Bbin{d}.csv')
                if self.domains[d].is_discrete():
                    self._save_state(self.Yord[d], 'Yord.csv')
                    self._save_state(self.Ycount[d], 'Ycount.csv')
                    self._save_state(self.Bord[d], 'Bord.csv')
                    self._save_state(self.Bcount[d], 'Bcount.csv')
                    for r in range(self.n_classes[d]):
                        self._save_state(self.Ycat[d][r], f'Ycat{r + 1}.csv')
                        self._save_state(self.Bcat[d][:, r], f'Bcat{r + 1}.csv')
                    self._save_state(self.theta[d], 'theta.csv')
            self._variables_saved = True

    def _validate_inputs(self, X: np.ndarray,
                         domains: Dict[int, Domain], prior_alpha: np.ndarray,
                         s2B: float, s2Y: float, s2Z: float, s2theta: float, s2u: float,
                         K: int, n_iterations: int, seed: int, **kwargs):
        # shapes
        assert len(domains) == X.shape[1], "domains must be specified for each variable"
        assert X.shape[1] == prior_alpha.shape[0], "X and W must have the same number of columns"

        # domains
        for d in range(X.shape[1]):
            assert X[:, d].dtype == np.float64, f"X must be float but was {X[:, d].dtype}"
            if domains[d].is_real():
                continue  # no restrictions here
            if domains[d].is_positive_real():
                valid_values = (X[:, d] > 0) | np.isnan(X[:, d])
                assert valid_values.all(), f"Positive variables must be strictly positive or nan, {X[~valid_values, d]}"
            if domains[d].is_discrete():
                valid_encoding = ((X[:, d] >= 0) | np.isnan(X[:, d]))
                assert valid_encoding.all(), f"Discrete variables must be encoded as positive integers or -1 for missing values, {X[~valid_encoding, d]}"
                assert np.nanmax(X[:, d]) < 5000, f"If discrete variables contain too many classes, e.g. {X[:, d].max()}, the algorithm will be slow. Consider treating them as continuous instead."
            elif domains[d].is_binary():
                valid_values = (X[:, d] == 0) | (X[:, d] == 1) | np.isnan(X[:, d])
                assert valid_values.all(), f"Binary variables must be 0, 1 or nan, {X[~valid_values, d]}"

        # Hyperparameters
        assert (prior_alpha >= 0).all(), "W must be non-negative"
        assert s2B > 0, "s2B must be positive"
        assert s2Y > 0, "s2Y must be positive"
        assert s2Z > 0, "s2Z must be positive"
        assert s2theta > 0, "s2theta must be positive"
        assert s2u > 0, "s2u must be positive"
        assert K > 0 and K <= X.shape[1], "Kest must be positive but less than the number of columns of X"
        assert n_iterations > 0, "Nsim must be positive"

    def _log(self, array: np.ndarray, filename: str, *args, **kwargs):
        assert array.ndim <= 2, f'{filename}: {array.shape}'

        LOG_DIR = 'traces'
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)

        # average over observations
        if 'S' in filename:
            _, array = np.unique(array, return_counts=True)
        elif 'Yord' not in filename and 'Ycount' not in filename and array.ndim > 0 and array.shape[0] == 1000:
            array = array.mean(axis=0)
        elif 'Yord' not in filename and 'Ycount' not in filename and array.ndim == 2 and array.shape[1] == 1000:
            array = array.mean(axis=1)

        # reshape to store in rows
        if array.ndim < 2:
            array = array.reshape(1, -1)

        # create csv header
        filename = LOG_DIR + '/' + filename
        if os.path.exists(filename):
            header = ''
        else:
            header = ','.join([f'{i}' for i in range(array.shape[1])])

        # store results
        with open(filename, 'ab') as f:
            np.savetxt(f, array, delimiter=',', header=header, comments='')

    def _log_variables(self):
        self._log(self.Z, f'Z.csv')
        for d in range(self.D):
            if self.domains[d].is_continuous():
                self._log(self.Yreal[d], f'Yreal{d}.csv')
                self._log(self.Yint[d], f'Yint{d}.csv')
                self._log(self.Breal[d], f'Breal{d}.csv')
                self._log(self.Bint[d], f'Bint{d}.csv')
            if self.domains[d].is_positive_real():
                self._log(self.Ypos[d], f'Ypos{d}.csv')
                self._log(self.Bpos[d], f'Bpos{d}.csv')
            if self.domains[d].is_binary():
                self._log(self.Ybin[d], f'Ybin{d}.csv')
                self._log(self.Bbin[d], f'Bbin{d}.csv')
            if self.domains[d].is_discrete():
                for current_class in range(self.n_classes[d]):
                    self._log(self.Ycat[d][current_class], f'Ycat{current_class}{d}.csv')
                    self._log(self.Bcat[d][:, current_class], f'Bcat{current_class}{d}.csv')
                self._log(self.Yord[d], f'Yord{d}.csv')
                self._log(self.Ycount[d], f'Ycount{d}.csv')
                self._log(self.theta[d], f'theta{d}.csv')
                self._log(self.Bord[d], f'Bord{d}.csv')
                self._log(self.Bcount[d], f'Bcount{d}.csv')
            self._log(self.S[:, d] + 1, f'S{d}.csv')
            self._log(self.W[d, :], f'W{d}.csv')

    # INITIALIZATION

    def initialize_w(self, d: int):
        assert not self.domains[d].is_binary(), "W not defined on binary domain"

        # initialization
        self.W[d, :] = self.rng.dirichlet(self.prior_alpha[d, :])

        # consistency checks
        if self.domains[d].is_real():
            assert self.W[d, 2] == 0, f"if domain is {self.domains[d]} weight for positive data must remain zero but was {self.W[d, 2]}"

    def initialize_s(self, d: int):
        assert not self.domains[d].is_binary(), "S not defined on binary variables."
        self.S[:, d] = self.rng.multinomial(self.W[d], size=self.S.shape[0])

    def initialize_b(self, d: int):
        muB = np.zeros((self.K, ))
        SB = self.s2_b * np.eye(self.K)
        if self.domains[d].is_continuous():
            self.Breal[d] = self.rng.multivariate_normal(muB, SB)
            self.Bint[d] = self.rng.multivariate_normal(muB, SB)
            if self.domains[d].is_positive_real():
                self.Bpos[d] = self.rng.multivariate_normal(muB, SB)
        elif self.domains[d].is_binary():
            self.Bbin[d] = self.rng.multivariate_normal(muB, SB)
        elif self.domains[d].is_discrete():
            self.Bcat[d][:, 0] = -1.0  # first coefficient fixed for identifiability
            self.Bcat[d][:, 1:] = self.rng.multivariate_normal(muB, SB, size=self.n_classes[d] - 1).T
            assert (self.Bcat[d][:, 0] == -1).all(), f"B(0) must remain fixed for identifiability but was {self.Bcat[d][:, 0]}"
            self.Bord[d] = self.rng.multivariate_normal(muB, SB)
            self.Bcount[d] = self.rng.multivariate_normal(muB, SB)

    def initialize_theta(self, d: int):
        assert self.domains[d].is_discrete(), f"theta not defined on domain {self.domains[d]}"

        # initialization
        self.theta[d][0] = -self.s_y
        for r in range(1, self.n_classes[d] - 1):  # theta_0 fixed, theta_R = infty
            self.theta[d][r] = self.theta[d][r - 1] + (4 * self.s_y / self.n_classes[d]) * self.rng.random()

        # consistency checks
        assert not np.isinf(self.theta[d]).any(), "thresholds must be finite"
        assert (self.theta[d][:-1] < self.theta[d][1:]).all(), "thresholds must be in increasing order"

    def initialize_y(self, d: int):
        missing = np.isnan(self.X[:, d])
        n_missing = missing.sum()
        if self.domains[d].is_continuous():
            # real
            self.Yreal[d][missing] = self.rng.normal(loc=0, scale=self.s_y, size=n_missing)
            self.Yreal[d][~missing] = f_real_inv(self.X[~missing, d], self.w_real[d], self.b_real[d])

            # interval
            self.Yint[d][missing] = self.rng.normal(loc=0, scale=self.s_y, size=n_missing)
            self.Yint[d][~missing] = f_int_inv(self.X[~missing, d], self.w_int, self.theta_L[d], self.theta_H[d])

            # positive
            if self.domains[d].is_positive_real():
                self.Ypos[d][missing] = self.rng.normal(loc=0, scale=self.s_y, size=n_missing)
                self.Ypos[d][~missing] = f_pos_inv(self.X[~missing, d], self.w_pos[d])
        elif self.domains[d].is_binary():
            # missing values
            self.Ybin[d][missing] = self.rng.normal(loc=0, scale=self.s_y, size=n_missing)

            # observed values
            first_class_observations = (self.X[:, d] == 1)  # missing values are neither class 1 nor class 2, so don't check for that explicitly
            second_class_observations = (self.X[:, d] == 2)
            self.Ybin[d][first_class_observations] = self.rng.truncnorm(0, self.s_y, -np.inf, 0, size=first_class_observations.sum())
            self.Ybin[d][second_class_observations] = self.rng.truncnorm(0, self.s_y, 0, np.inf, size=second_class_observations.sum())

            # consistency checks
            max_y_first_class = self.Ybin[d][first_class_observations].max(initial=-np.inf)
            min_y_second_class = self.Ybin[d][second_class_observations].min(initial=np.inf)
            assert max_y_first_class < min_y_second_class, f"Ybin must be in increasing order, but {max_y_first_class} >= {min_y_second_class}"
        elif self.domains[d].is_discrete():
            # missing values
            self.Ycat[d][:, missing] = self.rng.normal(loc=0, scale=self.s_y, size=(self.n_classes[d], n_missing))
            self.Yord[d][missing] = self.rng.normal(loc=0, scale=self.s_y, size=n_missing)
            self.Ycount[d][missing] = self.rng.normal(loc=0, scale=self.s_y, size=n_missing)

            # initialize categorical pseudo-observations
            observed_class = (self.X[:, d] - 1)
            non_missing_idxs = (~missing).nonzero()[0]
            self.Ycat[d][observed_class[non_missing_idxs].astype(int), non_missing_idxs] = self.rng.truncnorm(
                0, self.s_y, 0, np.inf, size=self.N - n_missing
            )
            for current_class in range(self.n_classes[d]):
                idxs_where_current_class_not_observed = (~missing & (observed_class != current_class)).nonzero()[0]
                self.Ycat[d][current_class, idxs_where_current_class_not_observed] = self.rng.truncnorm(
                    0, self.s_y,
                    -np.inf,
                    self.Ycat[d][observed_class[idxs_where_current_class_not_observed].astype(int), idxs_where_current_class_not_observed],
                    size=len(idxs_where_current_class_not_observed)
                )

            # consistency checks categorical
            y_for_observed_class = self.Ycat[d][observed_class[non_missing_idxs].astype(int), non_missing_idxs]
            max_y_of_all_classes = self.Ycat[d][:, non_missing_idxs].max(axis=0)
            assert (y_for_observed_class >= max_y_of_all_classes).all(), f"y for observed class must be argmax but {y_for_observed_class} < {max_y_of_all_classes}"

            # initialize ordinal pseudo-observations
            first_class_observations = (self.X[:, d] == 1)  # no missing values can be 1
            last_class_observations = (self.X[:, d] == self.n_classes[d])  # no missing values can be Rd
            other_class_observations = ~missing & ~first_class_observations & ~last_class_observations
            self.Yord[d][first_class_observations] = self.rng.truncnorm(
                0, self.s_y, -np.inf, self.theta[d][0], size=first_class_observations.sum()
            )
            self.Yord[d][last_class_observations] = self.rng.truncnorm(
                0, self.s_y, self.theta[d][self.n_classes[d] - 2], np.inf, size=last_class_observations.sum()
            )
            previous_class = self.X[other_class_observations, d].astype(int) - 2
            current_class = self.X[other_class_observations, d].astype(int) - 1
            self.Yord[d][other_class_observations] = self.rng.truncnorm(
                0, self.s_y, self.theta[d][previous_class], self.theta[d][current_class]  # don't need size, because theta[between] is a vector of correct size, so it gets broadcasted
            )

            # consistency checks ordinal
            for current_class in range(1, self.n_classes[d]):
                max_y_for_current_class = self.Yord[d][(self.X[:, d] == current_class)].max(initial=-np.inf)
                min_y_for_next_class = self.Yord[d][(self.X[:, d] == current_class + 1)].min(initial=np.inf)
                assert max_y_for_current_class < min_y_for_next_class, f"Yord must be in increasing order, but {max_y_for_current_class} >= {min_y_for_next_class}"

            # initialize count pseudo-observations
            self.Ycount[d][~missing] = g_inv(self.X[~missing, d], self.w_count[d]) + self.rng.normal(loc=0, scale=self.s_y, size=self.N - n_missing)

            # Ycount not necessarily ordered due to the added noise

    # FIT

    def update_z(self):
        PZ = np.zeros((self.N, self.K, self.K))
        muZ = np.zeros((self.N, self.K))
        for d in range(self.D):
            missing = np.isnan(self.X[:, d])
            if self.domains[d].is_continuous():
                # real
                PZ += np.outer(self.Breal[d], self.Breal[d])
                muZ += self.Breal[d] * self.Yreal[d][:, np.newaxis]

                # interval
                PZ += np.outer(self.Bint[d], self.Bint[d])
                muZ += self.Bint[d] * self.Yint[d][:, np.newaxis]

                # positive
                if self.domains[d].is_positive_real():
                    PZ += np.outer(self.Bpos[d], self.Bpos[d])
                    muZ += self.Bpos[d] * self.Ypos[d][:, np.newaxis]
            elif self.domains[d].is_binary():
                # binary
                PZ += np.outer(self.Bbin[d], self.Bbin[d])
                muZ += self.Bbin[d] * self.Ybin[d][:, np.newaxis]
            elif self.domains[d].is_discrete():
                # categorical
                observed_class = self.X[:, d] - 1
                non_missing_idxs = (~missing).nonzero()[0]
                PZ[non_missing_idxs] += np.einsum(
                    'kn,ln->nkl',
                    self.Bcat[d][:, observed_class[non_missing_idxs].astype(int)],
                    self.Bcat[d][:, observed_class[non_missing_idxs].astype(int)]
                )
                muZ_cat = (self.Bcat[d][:, observed_class[non_missing_idxs].astype(int)] * self.Ycat[d][observed_class[non_missing_idxs].astype(int), non_missing_idxs]).T
                muZ_cat[observed_class[non_missing_idxs] == 0] = 0
                muZ[non_missing_idxs] += muZ_cat

                # ordinal
                PZ += np.outer(self.Bord[d], self.Bord[d])
                muZ += self.Bord[d] * self.Yord[d][:, np.newaxis]

                # count
                PZ += np.outer(self.Bcount[d], self.Bcount[d])
                muZ += self.Bcount[d] * self.Ycount[d][:, np.newaxis]
        PZ = (1 / self.s2_y) * PZ + (1 / self.s2_z) * np.eye(self.K)
        SZ = np.linalg.inv(PZ)
        muZ = (1 / self.s2_y) * np.einsum('ijk,ik->ij', SZ, muZ)
        self.Z = self.rng.batched_multivariate_normal(muZ, SZ).T

    def update_y(self, d: int):
        missing = np.isnan(self.X[:, d])
        if self.domains[d].is_continuous():
            # means
            muYreal = self.Z.T @ self.Breal[d]
            muYint = self.Z.T @ self.Bint[d]

            # assignments
            assigned_real = (self.S[:, d] == 0)
            assigned_int = (self.S[:, d] == 1)

            # masks
            real_observations = assigned_real & ~missing
            int_observations = assigned_int & ~missing

            # priors
            self.Yreal[d][~real_observations] = self.rng.normal(muYreal[~real_observations], self.s_y)
            self.Yint[d][~int_observations] = self.rng.normal(muYint[~int_observations], self.s_y)

            # posterior parameters
            s2_hat = 1 / (1 / self.s2_y + 1 / self.s2_u)
            mu_hat_real = s2_hat * (f_real_inv(self.X[real_observations, d], self.w_real[d], self.b_real[d]) / self.s2_u + muYreal[real_observations] / self.s2_y)
            mu_hat_int = s2_hat * (f_int_inv(self.X[int_observations, d], self.w_int, self.theta_L[d], self.theta_H[d]) / self.s2_u + muYint[int_observations] / self.s2_y)

            # sampling
            self.Yreal[d][real_observations] = self.rng.normal(mu_hat_real, math.sqrt(s2_hat))
            self.Yint[d][int_observations] = self.rng.normal(mu_hat_int, math.sqrt(s2_hat))

            if self.domains[d].is_positive_real():
                # means
                muYpos = self.Z.T @ self.Bpos[d]

                # assignments
                assigned_pos = (self.S[:, d] == 2)

                # masks
                pos_observations = assigned_pos & ~missing

                # prior
                self.Ypos[d][~pos_observations] = self.rng.normal(muYpos[~pos_observations], self.s_y)

                # posterior parameters
                s2_hat = 1 / (1 / self.s2_y + 1 / self.s2_u)
                mu_hat = s2_hat * (f_pos_inv(self.X[pos_observations, d], self.w_pos[d]) / self.s2_u + muYpos[pos_observations] / self.s2_y)

                # sampling
                self.Ypos[d][pos_observations] = self.rng.normal(mu_hat, math.sqrt(s2_hat))
        elif self.domains[d].is_binary():
            # mean
            muYbin = self.Z.T @ self.Bbin[d]

            # prior
            self.Ybin[d][missing] = self.rng.normal(muYbin[missing], self.s_y)

            # posterior
            bin_observations_first_class = (self.X[:, d] == 1)  # missing values are neither class 1 nor class 2, so don't check for that explicitly
            bin_observations_second_class = (self.X[:, d] == 2)
            self.Ybin[d][bin_observations_first_class] = self.rng.truncnorm(muYbin[bin_observations_first_class], self.s_y, -np.inf, 0)
            self.Ybin[d][bin_observations_second_class] = self.rng.truncnorm(muYbin[bin_observations_second_class], self.s_y, 0, np.inf)
        elif self.domains[d].is_discrete():
            # means
            muYcat = (self.Z.T @ self.Bcat[d]).T
            muYord = self.Z.T @ self.Bord[d]
            muYcount = self.Z.T @ self.Bcount[d]

            # assignments
            assigned_cat = (self.S[:, d] == 0)
            assigned_ord = (self.S[:, d] == 1)
            assigned_count = (self.S[:, d] == 2)

            # masks
            cat_observations = assigned_cat & ~missing
            ord_observations = assigned_ord & ~missing
            count_observations = assigned_count & ~missing

            # priors
            self.Ycat[d][:, ~cat_observations] = self.rng.normal(muYcat[:, ~cat_observations], self.s_y)
            self.Yord[d][~ord_observations] = self.rng.normal(muYord[~ord_observations], self.s_y)
            self.Ycount[d][~count_observations] = self.rng.normal(muYcount[~count_observations], self.s_y)

            # posteriors categorical
            # loop over number of classes R instead of number of observations N, because most likely R << N,
            # and larger loop benefits more from broadcasting
            observed_class = self.X[:, d] - 1
            idxs_assigned_cat = cat_observations.nonzero()[0]
            self.Ycat[d][observed_class[idxs_assigned_cat].astype(int), idxs_assigned_cat] = self.rng.truncnorm(
                muYcat[observed_class[idxs_assigned_cat].astype(int), idxs_assigned_cat], self.s_y, 0, np.inf
            )
            for current_class in range(self.n_classes[d]):
                idxs_where_current_class_not_observed = (cat_observations & (observed_class != current_class)).nonzero()[0]
                self.Ycat[d][current_class, idxs_where_current_class_not_observed] = self.rng.truncnorm(
                    muYcat[current_class, idxs_where_current_class_not_observed],
                    self.s_y,
                    -np.inf,
                    self.Ycat[d][observed_class[idxs_where_current_class_not_observed].astype(int), idxs_where_current_class_not_observed]
                )

            # consistency checks categorical
            y_for_observed_class = self.Ycat[d][observed_class[idxs_assigned_cat].astype(int), idxs_assigned_cat]
            max_y_of_all_classes = self.Ycat[d][:, idxs_assigned_cat].max(axis=0)
            assert (y_for_observed_class >= max_y_of_all_classes).all(), f"y for observed class must be argmax but {y_for_observed_class} < {max_y_of_all_classes}"

            # posteriors ordinal
            ord_observations_class_1 = ord_observations & (self.X[:, d] == 1)
            ord_observations_class_R = ord_observations & (self.X[:, d] == self.n_classes[d])
            ord_observations_between = ord_observations & ~ord_observations_class_1 & ~ord_observations_class_R

            self.Yord[d][ord_observations_class_1] = self.rng.truncnorm(
                muYord[ord_observations_class_1], self.s_y, -np.inf, self.theta[d][0]
            )
            self.Yord[d][ord_observations_class_R] = self.rng.truncnorm(
                muYord[ord_observations_class_R], self.s_y, self.theta[d][self.n_classes[d] - 2], np.inf
            )
            self.Yord[d][ord_observations_between] = self.rng.truncnorm(
                muYord[ord_observations_between],
                self.s_y,
                self.theta[d][self.X[ord_observations_between, d].astype(int) - 2],
                self.theta[d][self.X[ord_observations_between, d].astype(int) - 1]
            )

            # consistency checks ordinal
            assert (np.nanmin(self.X[:, d]) > 0).all(), f"assumes label encoding starts at 1, but {np.nanmin(self.X[:, d])}"
            for current_class in range(1, self.n_classes[d]):
                ord_observations_current_class = ord_observations & (self.X[:, d] == current_class)
                ord_observations_next_class = ord_observations & (self.X[:, d] == current_class + 1)
                max_y_current_class = self.Yord[d][ord_observations_current_class].max(initial=-np.inf)
                min_y_next_class = self.Yord[d][ord_observations_next_class].min(initial=np.inf)
                assert max_y_current_class < min_y_next_class, f"Yord must be in increasing order, but {max_y_current_class} >= {min_y_next_class}"

            # posteriors count
            self.Ycount[d][count_observations] = self.rng.truncnorm(
                muYcount[count_observations],
                self.s_y,
                g_inv(self.X[count_observations, d], self.w_count[d]),
                g_inv(self.X[count_observations, d] + 1, self.w_count[d])
            )

            # consistency checks count
            assert (np.nanmin(self.X[:, d]) > 0).all(), f"assumes label encoding starts at 1, but {np.nanmin(self.X[:, d])}"
            for current_class in range(1, self.n_classes[d]):
                count_observations_current_class = count_observations & (self.X[:, d] == current_class)
                count_observations_next_class = count_observations & (self.X[:, d] == current_class + 1)
                max_y_current_class = self.Ycount[d][count_observations_current_class].max(initial=-np.inf)
                min_y_next_class = self.Ycount[d][count_observations_next_class].min(initial=np.inf)
                assert max_y_current_class < min_y_next_class, f"Ycount must be in increasing order, but {max_y_current_class} >= {min_y_next_class}"

    def update_theta(self, d: int):
        assert self.domains[d].is_discrete(), "theta is only defined for discrete variables"

        # masks
        ord_observations = (~np.isnan(self.X[:, d]) & (self.S[:, d] == 1))

        # x=1 is never considered here, because it is sampled in the region (-inf, theta_1] and theta_1 is fixed
        # theta_1 is stored at theta[0], so if the current class is 2 and we want to update theta_2 stored at theta[1],
        # we need to store max(theta[0], max(y|x=2)) and min(theta[2], min(y|x=3)) at theta[1]
        for current_class in range(2, self.n_classes[d]):
            # theta min
            y_max_current_class = self.Yord[d][ord_observations & (self.X[:, d] == current_class)].max(initial=-np.inf)
            theta_previous_class = self.theta[d][current_class - 2]
            theta_min = max(theta_previous_class, y_max_current_class)

            # theta max, avoid storing last threshold here and set it to infinity explicitly
            if current_class < self.n_classes[d] - 1:
                y_min_next_class = self.Yord[d][ord_observations & (self.X[:, d] == current_class + 1)].min(initial=np.inf)
                theta_next_class = self.theta[d][current_class]
                theta_max = min(theta_next_class, y_min_next_class)
            else:
                theta_max = np.inf

            # sampling
            assert theta_min <= theta_max, f"at d={d}, xhi={theta_max} <= xlo={theta_min}"
            self.theta[d][current_class - 1] = self.rng.truncnorm(0, self.s2_theta, theta_min, theta_max)

        # consistency checks
        assert self.theta[d][0] == -self.s_y, f"first theta must remain fixed to -sY for identifiability but was {self.theta[d][0]}"
        for current_class in range(1, self.n_classes[d] - 1):
            theta_previous_class = self.theta[d][current_class - 1]
            theta_current_class = self.theta[d][current_class]
            assert theta_previous_class <= theta_current_class, f"theta[{current_class - 1}]={theta_previous_class} < theta[{current_class}]={theta_current_class}"

    def update_b(self, d: int):
        PB = (1 / self.s2_y) * (self.Z @ self.Z.T) + (1 / self.s2_b) * np.eye(self.K)
        SB = np.linalg.inv(PB)
        if self.domains[d].is_continuous():
            # real
            muB = (SB @ self.Z @ self.Yreal[d]) / self.s2_y
            self.Breal[d] = self.rng.multivariate_normal(muB, SB)

            # interval
            muB = (SB @ self.Z @ self.Yint[d]) / self.s2_y
            self.Bint[d] = self.rng.multivariate_normal(muB, SB)

            # positive
            if self.domains[d].is_positive_real():
                muB = (SB @ self.Z @ self.Ypos[d]) / self.s2_y
                self.Bpos[d] = self.rng.multivariate_normal(muB, SB)
        elif self.domains[d].is_binary():
            # binary
            muB = (SB @ self.Z @ self.Ybin[d]) / self.s2_y
            self.Bbin[d] = self.rng.multivariate_normal(muB, SB)
        elif self.domains[d].is_discrete():
            # categorical
            muB = (SB @ self.Z @ self.Ycat[d][1:].T).T / self.s2_y
            self.Bcat[d][:, 1:] = self.rng.batched_multivariate_normal(muB, SB).T

            # consistency checks categorical
            assert (self.Bcat[d][:, 0] == -1).all(), "B(0) must remain fixed for identifiability"

            # ordinal
            muB = (SB @ self.Z @ self.Yord[d]) / self.s2_y
            self.Bord[d] = self.rng.multivariate_normal(muB, SB)

            # count
            muB = (SB @ self.Z @ self.Ycount[d]) / self.s2_y
            self.Bcount[d] = self.rng.multivariate_normal(muB, SB)

    def update_s(self, d: int):
        # missing values
        missing = np.isnan(self.X[:, d])

        component_log_likelihoods = np.zeros(((~missing).sum(), 3))
        if self.domains[d].is_continuous():
            # real
            component_log_likelihoods[:, 0] = log_pdf_x_real(
                self.X[~missing, d],
                self.w_real[d], self.b_real[d],
                self.Z[:, ~missing].T @ self.Breal[d],
                self.s2_y, self.s2_u
            )

            # interval
            component_log_likelihoods[:, 1] = log_pdf_x_int(
                self.X[~missing, d],
                self.w_int, self.theta_L[d], self.theta_H[d],
                self.Z[:, ~missing].T @ self.Bint[d],
                self.s2_y, self.s2_u
            )

            # positive
            if self.domains[d].is_positive_real():
                component_log_likelihoods[:, 2] = log_pdf_x_pos(
                    self.X[~missing, d],
                    self.w_pos[d],
                    self.Z[:, ~missing].T @ self.Bpos[d],
                    self.s2_y, self.s2_u
                )

            # logging
            if self.logging:
                self._log(component_log_likelihoods[:, 0], f'log_prob_x_real{d}.csv')
                self._log(component_log_likelihoods[:, 1], f'log_prob_x_int{d}.csv')
                if self.domains[d].is_positive_real():
                    self._log(component_log_likelihoods[:, 2], f'log_prob_x_pos{d}.csv')
        elif self.domains[d].is_discrete():
            # categorical
            component_log_likelihoods[:, 0] = log_pmf_x_cat(
                self.X[~missing, d],
                self.Z[:, ~missing],
                self.Bcat[d],
                self.rng.normal(loc=0, scale=self.s_y, size=((~missing).sum(), 100))
            )

            # ordinal
            component_log_likelihoods[:, 1] = log_pmf_x_ord(
                self.X[~missing, d],
                self.Z[:, ~missing].T @ self.Bord[d],
                self.theta[d],
                self.s_y
            )

            # count
            component_log_likelihoods[:, 2] = log_pmf_x_count(
                self.X[~missing, d],
                self.Z[:, ~missing].T @ self.Bcount[d],
                self.w_count[d],
                self.s_y
            )

            # consistency checks
            assert (component_log_likelihoods <= 0).all(), f"log probability mass must be non-positive but is {component_log_likelihoods}"

            if self.logging:
                # logging
                self._log(component_log_likelihoods[:, 0], f'log_prob_x_cat{d}.csv')
                self._log(component_log_likelihoods[:, 1], f'log_prob_x_ord{d}.csv')
                self._log(component_log_likelihoods[:, 2], f'log_prob_x_count{d}.csv')

        # weight for positive will always be zero if positive data type is disabled, thus its log probability has permission to be -inf
        log_weights = extended_log(self.W[d, :])

        # posterior component probabilities
        log_responsibilities = log_weights[np.newaxis, :] + component_log_likelihoods

        # numerical Stability
        responsibilities = np.exp(log_responsibilities - log_responsibilities.max(axis=1, keepdims=True))
        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
        # responsibilities = np.where(np.isnan(responsibilities), self.W[d, :], responsibilities)

        # consistency checks
        assert responsibilities.shape == ((~missing).sum(), 3), f"responsibilities must have shape {((~missing).sum(), 3)} but was {responsibilities.shape}"
        assert responsibilities.min() >= 0, f"responsibilities must be non-negative but was {responsibilities.min()}"
        assert np.isclose(responsibilities.sum(axis=1), 1).all(), f"responsibilities must sum to 1 but was {responsibilities.sum(axis=1)}"
        if self.domains[d].is_real():
            assert (responsibilities[:, 2] == 0).all(), "probability for positive data must remain zero if disabled, but " + str(responsibilities[:, 2])

        # Sample S
        self.S[~missing, d] = self.rng.multinomial(responsibilities)
        self.S[missing, d] = self.rng.multinomial(self.W[d, :], size=missing.sum())

    def update_w(self, d: int):
        # sample W
        paramW = self.prior_alpha[d, :] + np.bincount(self.S[:, d], minlength=len(self.prior_alpha[d, :]))
        self.W[d, :] = self.rng.dirichlet(paramW)

        # consistency check
        if self.domains[d].is_real():
            assert self.W[d, 2] == 0, f"if positive data is disabled weight for positive data must remain zero but was {self.W[d, 2]}"

    def fit(self):
        """Fit the model to the data. For documentation of the algorithm's arguments, see the class docstring."""
        # initialization
        for d in range(self.D):
            if not self.domains[d].is_binary():
                self.initialize_w(d)
                self.initialize_s(d)
            self.initialize_b(d)
            if self.domains[d].is_discrete():
                self.initialize_theta(d)
            self.initialize_y(d)

        # main loop
        for _ in tqdm(range(self.n_iterations)):
            self.update_z()

            if self.logging:
                # do this here, because that way Z is initialized and the other variables are not yet updated
                self._save_variables()
                self._log_variables()

            for d in range(self.D):
                self.update_y(d)
                if self.domains[d].is_discrete():
                    self.update_theta(d)
                self.update_b(d)
                if not self.domains[d].is_binary():
                    self.update_s(d)
                    self.update_w(d)

        # allow something like model = model.fit()
        return self

    def score(self, X: np.ndarray) -> float:
        assert X.shape == self.X.shape, f"shape of test data must match that of train data ({self.X.shape}) but was {X.shape}"
        log_likelihood = np.zeros_like(X)
        for d in range(X.shape[1]):
            masked = np.isnan(self.X[:, d]) & ~np.isnan(X[:, d])
            if self.domains[d].is_continuous():
                # assignments
                assigned_real = (self.S[:, d] == 0)
                assigned_int = (self.S[:, d] == 1)

                # masks
                real_observations = assigned_real & masked
                int_observations = assigned_int & masked

                # real
                log_likelihood[real_observations, d] = log_pdf_x_real(
                    X[real_observations, d],
                    self.w_real[d], self.b_real[d],
                    self.Z[:, real_observations].T @ self.Breal[d],
                    self.s2_y, self.s2_u
                )

                # interval
                log_likelihood[int_observations, d] = log_pdf_x_int(
                    X[int_observations, d],
                    self.w_int, self.theta_L[d], self.theta_H[d],
                    self.Z[:, int_observations].T @ self.Bint[d],
                    self.s2_y, self.s2_u
                )

                if self.domains[d].is_positive_real():
                    # assignments
                    assigned_pos = (self.S[:, d] == 2)

                    # masks
                    pos_observations = assigned_pos & masked

                    # positive
                    log_likelihood[pos_observations, d] = log_pdf_x_pos(
                        X[pos_observations, d],
                        self.w_pos[d],
                        self.Z[:, pos_observations].T @ self.Bpos[d],
                        self.s2_y, self.s2_u
                    )
            elif self.domains[d].is_discrete():
                # assignments
                assigned_cat = (self.S[:, d] == 0)
                assigned_ord = (self.S[:, d] == 1)
                assigned_count = (self.S[:, d] == 2)

                # masks
                cat_observations = assigned_cat & masked
                ord_observations = assigned_ord & masked
                count_observations = assigned_count & masked

                # categorical
                log_likelihood[cat_observations, d] = log_pmf_x_cat(
                    X[cat_observations, d],
                    self.Z[:, cat_observations],
                    self.Bcat[d],
                    self.rng.normal(loc=0, scale=self.s_y, size=(cat_observations.sum(), 100))
                )

                # ordinal
                log_likelihood[ord_observations, d] = log_pmf_x_ord(
                    X[ord_observations, d],
                    self.Z[:, ord_observations].T @ self.Bord[d],
                    self.theta[d],
                    self.s_y
                )

                # count
                log_likelihood[count_observations, d] = log_pmf_x_count(
                    X[count_observations, d],
                    self.Z[:, count_observations].T @ self.Bcount[d],
                    self.w_count[d],
                    self.s_y
                )

        return log_likelihood[masked].mean(axis=0).sum()
