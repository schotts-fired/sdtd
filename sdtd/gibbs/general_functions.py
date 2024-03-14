import math
from typing import Union
import numpy as np
from scipy import stats, special
from numba import njit


# random numbers

class RandomNumberGenerator:
    """Wrapper around numpy's random number generator to allow for easy seeding and extensibility.

    Args:
        seed (int): The seed for numpy's random number generator.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.np_rng = np.random.default_rng(seed=seed)

    def random(self, size: int | tuple = None) -> np.ndarray:
        """Draw random numbers from a uniform distribution over [0, 1).

        Args:
            size (int or tuple): The shape of the output.
        Returns:
            np.ndarray: The random numbers.
        """
        return self.np_rng.random(size=size)

    def normal(self, loc: np.ndarray, scale: float, size: int | tuple = None) -> np.ndarray:
        """Draw random numbers from a normal distribution.

        Args:
            loc (np.ndarray): The mean of the normal distribution.
            scale (float): The standard deviation of the normal distribution.
            size (int or tuple): The shape of the output.
        Returns:
            np.ndarray: The random numbers.
        """
        return self.np_rng.normal(loc=loc, scale=scale, size=size)

    def multivariate_normal(self, m: np.ndarray, S: np.matrix, size: int | tuple = None) -> np.ndarray:
        """Draw random numbers from a multivariate normal distribution.

        Args:
            m (np.ndarray): The mean of the multivariate normal distribution.
            S (np.matrix): The covariance matrix of the multivariate normal distribution.
            size (int or tuple): The shape of the output.
        Returns:
            np.ndarray: The random numbers.
        """
        return self.np_rng.multivariate_normal(mean=m, cov=S, size=size)

    def batched_multivariate_normal(self, mean: np.ndarray, cov: np.ndarray, size: int | tuple = ()) -> np.ndarray:
        """Draw random numbers from a multivariate normal distribution. This function allows for batched sampling by passing a batch
        of means and covariance matrices. The output will have the same batch size as the input. The batch shape is determined by
        the first dimensions of the input arrays and must be broadcastable for mean and covariance.

        The implementation is based on the following stackoverflow answer: https://stackoverflow.com/questions/69399035/is-there-a-way-of-batch-sampling-from-numpys-multivariate-normal-distribution-i

        Args:
            mean (np.ndarray): The means of the multivariate normal distributions.
            cov (np.ndarray): The covariance matrices of the multivariate normal distributions.
            size (int or tuple): The shape of the output. This argument is not tested! Use with caution.
        Returns:
            np.ndarray: The random numbers.
        """
        size = (size, ) if isinstance(size, int) else tuple(size)
        shape = size + np.broadcast_shapes(mean.shape, cov.shape[:-1])
        X = self.np_rng.standard_normal((*shape, 1))
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            print((np.linalg.eigvals(cov) < 0).any())
            raise ValueError("Covariance matrix must be positive definite.")
        return (L @ X).reshape(shape) + mean

    def multinomial(self, p: np.ndarray, size: int | tuple = None) -> np.ndarray:
        """Draw random numbers from a multinomial distribution.

        Args:
            p (np.ndarray): The probabilities of the different outcomes.
            size (int or tuple): The shape of the output.
        Returns:
            np.ndarray: The random numbers.
        """
        return self.np_rng.multinomial(n=1, pvals=p, size=size).argmax(axis=-1)

    def truncnorm(self,
                  m: np.ndarray | float,
                  s: np.ndarray | float,
                  a: np.ndarray | float,
                  b: np.ndarray | float,
                  size: int | tuple = None) -> np.ndarray:
        """Draw random numbers from a truncated normal distribution. This implementation allows for batched inputs.
        This is by far not exhaustively tested and there is no guarantee that broadcasting works as expected. Use with caution
        and independently check the output for correctness.

        Args:
            m (np.ndarray or float): The means of the normal distributions.
            s (np.ndarray or float): The standard deviations of the normal distributions.
            a (np.ndarray or float): The lower bounds of the truncation.
            b (np.ndarray or float): The upper bounds of the truncation.
            size (int or tuple): The shape of the output.
        Returns:
            np.ndarray: The random numbers.
        """
        assert np.all(a <= b), f"a={a} > b={b}."
        assert np.all(s > 0), "s must be positive."

        # this does not allow advanced broadcasting but is sufficient for my purposes
        if size is None:
            if isinstance(m, np.ndarray):
                size = m.shape
            if isinstance(s, np.ndarray):
                if size is not None:
                    assert size == s.shape, f"size argument must match shape of scales, but was {size} and {s.shape}."
                size = s.shape
            if isinstance(a, np.ndarray):
                if size is not None:
                    assert size == a.shape, f"size argument must match shape of lower bounds, but was {size} and {a.shape}."
                size = a.shape
            if isinstance(b, np.ndarray):
                if size is not None:
                    assert size == b.shape, f"size argument must match shape of upper bounds, but was {size} and {b.shape}."
                size = b.shape

        low = special.ndtr((a - m) / s)
        high = special.ndtr((b - m) / s)

        r = self.np_rng.random(size=size)
        z = low + (high - low) * r
        assert np.all(0 <= z)  # drand48 in reference samples from [0, 1)
        assert np.all(z <= 1)  # drand48 in reference samples from [0, 1) but may round to 1

        x = s * special.ndtri(z) + m
        assert not np.any(np.isnan(x)), f"truncnorm output contains nan-values: {x}"

        return x

    def dirichlet(self, alpha: np.ndarray, size: int | tuple = None) -> np.ndarray:
        """Draw random numbers from a Dirichlet distribution. This implementation allows for zero-valued concentration parameters, which
        are deterministically mapped to the zero vector.

        Args:
            alpha (np.ndarray): The concentration parameters of the Dirichlet distribution.
            size (int or tuple): The shape of the output.
        Returns:
            np.ndarray: The random numbers.
        """
        res = np.empty(len(alpha))
        res[alpha != 0] = self.np_rng.dirichlet(alpha[alpha != 0], size=size)
        res[alpha == 0] = 0.0
        return res


# transformations and their derivatives

def f_real_inv(x: np.ndarray, w: float, b: float) -> np.ndarray:
    """Inverse of the transformation function for real-valued variables. Following the reference implementation, this is
    realized as the inverse of an affine transformation.

    Args:
        x (np.ndarray): The values in observation space.
        w (float): A slope parameter. This is the reciprocal of the slop parameter in x = w * y + b.
        b (float): A shift parameter.
    Returns:
        np.ndarray: Values of the variable scaled, shifted and mapped to the pseudo-observation space.
    """
    return w * (x - b)


def df_real_inv(x: np.ndarray, w: float) -> np.ndarray:
    """Jacobian factor of the inverse of the transformation function for real-valued variables. Required for the change of variables formula.

    Args:
        x (np.ndarray): The values in observation space.
        w (float): A slope parameter. This is the reciprocal of the slop parameter in x = w * y + b.
    Returns:
        np.ndarray: The Jacobian factor.
    """
    return np.full_like(x, w)  # this could just return w and use broadcasting, but I don't want to touch the code at the moment


def f_pos_inv(x: np.ndarray, w: float) -> np.ndarray:
    """Inverse of the transformation function for positive-valued variables. Following the reference implementation, this is
    realized as the inverse of the softplus function.

    Args:
        x (np.ndarray): The values in observation space.
        w (float): A slope parameter. This is the reciprocal of the slope parameter in x = log(exp(w * y) + 1).
    Returns:
        np.ndarray: Values of the variable scaled and mapped to the pseudo-observation space.
    """
    assert (x > 0).all(), f"f_pos_inv is only defined on positive floats, {x[x <= 0]} is outside the domain"
    return np.log(np.exp(w * x) - 1)


def df_pos_inv(x: np.ndarray, w: float) -> np.ndarray:
    """Jacobian factor of the inverse of the transformation function for positive-valued variables. Required for the change of variables formula.

    Args:
        x (np.ndarray): The values in observation space.
        w (float): A slope parameter. This is the reciprocal of the slope parameter in x = log(exp(w * y) + 1).
    Returns:
        np.ndarray: The Jacobian factor.
    """
    assert (x > 0).all(), "df_pos_inv is only defined on positive floats"
    return w / (1 - np.exp(-w * x))


def f_int_inv(x: np.ndarray, w: float, theta_L: float, theta_H: float) -> np.ndarray:
    """Inverse of the transformation function for interval-valued variables. Following the reference implementation, this is
    realized as the inverse of a sigmoid function, scaled and shifted to match the interval of the dataset.

    Args:
        x (np.ndarray): The values in observation space.
        w (float): A slope parameter. This is the reciprocal of the slope parameter in x = (theta_H - theta_L) * (1 / (1 + exp(-w * y))) + theta_L.
        theta_L (float): The lower bound of the interval.
        theta_H (float): The upper bound of the interval.
    Returns:
        np.ndarray: Values of the variable scaled, shifted and mapped to the pseudo-observation space.
    """
    assert (theta_L < x).all() and (x < theta_H).all(), "f_int_inv is only defined on values between theta_L and theta_H"
    return -1 / w * np.log((theta_H - x) / (x - theta_L))


def df_int_inv(x: np.ndarray, w: float, theta_L: float, theta_H: float) -> np.ndarray:
    """Jacobian factor of the inverse of the transformation function for interval-valued variables. Required for the change of variables formula.

    Args:
        x (np.ndarray): The values in observation space.
        w (float): A slope parameter. This is the reciprocal of the slope parameter in x = (theta_H - theta_L) * (1 / (1 + exp(-w * y))) + theta_L.
        theta_L (float): The lower bound of the interval.
        theta_H (float): The upper bound of the interval.
    Returns:
        np.ndarray: The Jacobian factor.
    """
    assert (theta_L < x).all() and (x < theta_H).all(), f"df_int_inv is only defined on values between theta_L={theta_L} and theta_H={theta_H}, but received x={x[x < theta_L]} and x={x[x > theta_H]}"
    return 1 / w * (theta_H - theta_L) / ((theta_H - x) * (x - theta_L))


def g_inv(x: np.ndarray, w: float) -> np.ndarray:
    """Inverse of the transformation function for count-valued variables. Following the reference implementation, this is
    realized as the inverse of the softplus function.

    Args:
        x (np.ndarray): The values in observation space.
        w (float): A slope parameter. This is the reciprocal of the slope parameter in x = log(exp(w * y) + 1).
    Returns:
        np.ndarray: Values of the variable scaled and mapped to the pseudo-observation space.
    """
    assert (x > 0).all(), f"g_inv is only defined on positive floats, {x} is outside the domain"
    return np.log(np.exp(w * x) - 1)


# log pdfs

def log_pdf_x_real(x: np.ndarray, w: float, mean_x: float, mean_y: np.ndarray, s2y: float, s2u: float) -> np.ndarray:
    """Log pdf for real-valued variables. This is the pdf derived from the pseudo-observations by the change of variables formula.

    Args:
        x (np.ndarray): The values in observation space.
        w (float): A slope parameter. This is the reciprocal of the slope parameter in x = w * y + b.
        mean_x (float): The mean of the variable in observation space.
        mean_y (np.ndarray): The mean of the variable in pseudo-observation space.
        s2y (float): The variance of the variable in pseudo-observation space.
        s2u (float): The variance of the u term in the definition of the pseudo-observations: x = f(y + u).
    Returns:
        np.ndarray: The log pdf of the variable under the assumption of it belonging to the real numbers.
    """
    log_density_y = np.log(stats.norm.pdf(x=f_real_inv(x, w, mean_x), loc=mean_y, scale=math.sqrt(s2y + s2u)))
    log_jacobian = np.log(abs(df_real_inv(x, w)))
    log_pdf_x = log_density_y + log_jacobian
    return log_pdf_x


def log_pdf_x_pos(x: np.ndarray, w: float, mean_y: np.ndarray, s2y: float, s2u: float) -> np.ndarray:
    """Log pdf for positive real-valued variables. This is the pdf derived from the pseudo-observations by the change of variables formula.

    Args:
        x (np.ndarray): The values in observation space. Must be strictly positive.
        w (float): A slope parameter. This is the reciprocal of the slope parameter in x = log(exp(w * y) + 1).
        mean_x (float): The mean of the variable in observation space.
        mean_y (np.ndarray): The mean of the variable in pseudo-observation space.
        s2y (float): The variance of the variable in pseudo-observation space.
        s2u (float): The variance of the u term in the definition of the pseudo-observations: x = f(y + u).
    Returns:
        np.ndarray: The log pdf of the variable under the assumption of it belonging to the positive real numbers.
    """
    assert (x > 0).all(), "x must lie in the support of the distribution"
    log_density_y = np.log(stats.norm.pdf(x=f_pos_inv(x, w), loc=mean_y, scale=math.sqrt(s2y + s2u)))
    log_jacobian = np.log(abs(df_pos_inv(x, w)))
    log_pdf_x = log_density_y + log_jacobian
    return log_pdf_x


def log_pdf_x_int(x: np.ndarray, w: float, theta_L: float, theta_H: float, mean_y: np.ndarray, s2y: float, s2u: float) -> np.ndarray:
    """Log pdf for real-valued variables that are restricted to an interval. This is the pdf derived from the pseudo-observations by the change of variables formula.

    Args:
        x (np.ndarray): The values in observation space. Must lie between theta_L and theta_H.
        w (float): A slope parameter. This is the reciprocal of the slope parameter in x = (theta_H - theta_L) * (1 / (1 + exp(-w * y))) + theta_L.
        mean_x (float): The mean of the variable in observation space.
        mean_y (np.ndarray): The mean of the variable in pseudo-observation space.
        s2y (float): The variance of the variable in pseudo-observation space.
        s2u (float): The variance of the u term in the definition of the pseudo-observations: x = f(y + u).
    Returns:
        np.ndarray: The log pdf of the variable under the assumption of it belonging to a certain interval of the real line.
    """
    assert (theta_L < x).all() and (x < theta_H).all(), "x must lie in the support of the distribution"
    log_density_y = np.log(stats.norm.pdf(x=f_int_inv(x, w, theta_L, theta_H), loc=mean_y, scale=math.sqrt(s2y + s2u)))
    log_jacobian = np.log(abs(df_int_inv(x, w, theta_L, theta_H)))
    log_pdf_x = log_density_y + log_jacobian
    return log_pdf_x


# log pmfs

def log_pmf_x_cat(x: np.ndarray, Z: np.ndarray, b: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Log pmf for categorical variables. This is the pmf derived from the pseudo-observations by taking the probability of the pseudo-observation
    corresponding to the observed class is the argmax and then marginalizing it out. The marginalization is approximated by a monte carlo approximation.

    Args:
        x (np.ndarray): The values in observation space. Must be label encoded starting at 1.
        Z (np.ndarray): The latent variable matrix. Must contain latent variables for all (non-missing) observations in the columns.
        b (np.ndarray): The coefficient vectors for all classes of the observed variable.
        u (np.ndarray): The noise sample used for the monte carlo approximation to the expected value.
    Returns:
        np.ndarray: The log pmf of the variable under the assumption of it being categorical.
    """
    assert not np.isnan(x).any(), "x must not contain missing values"
    assert (x > 0).all(), "x must lie in the support of the distribution"
    assert Z.shape[1] == x.shape[0], "Z must contain latent variables for all (non-missing) observations"
    mean_y = mean_y_girolami_rogers(x, Z.T, b)  # numba wants the transpose
    log_pmf_x = extended_log(np.mean(np.prod(special.ndtr(u[:, np.newaxis, :] + mean_y[:, :, np.newaxis]), axis=1), axis=1))
    return log_pmf_x


def log_pmf_x_ord(x: np.ndarray, mean_y: np.ndarray, theta: np.ndarray, s_y: float) -> np.ndarray:
    """Log pmf for ordinal variables. This is the pmf derived from the pseudo-observations by subtracting cdfs for the thresholds corresponding to the
    different classes.

    Args:
        x (np.ndarray): The values in observation space. Must be label encoded starting at 1.
        mean_y (np.ndarray): The mean of the variable in pseudo-observation.
        theta (np.ndarray): The thresholds specifying the classes.
        s_y (float): The standard deviation of the variable in pseudo-observation space.
    Returns:
        np.ndarray: The log pmf of the variable under the assumption of it being ordinal.
    """
    assert (x > 0).all(), "x must lie in the support of the distribution"
    assert not np.isnan(x).any(), "x must not contain missing values"

    # masks
    first = (x == 1)
    last = (x == (len(theta) + 1))
    between = ~first & ~last

    # log pmf
    log_pmf_x = np.zeros_like(x, dtype=float)
    log_pmf_x[first] = extended_log(special.ndtr((theta[0] - mean_y[first]) / s_y))
    log_pmf_x[last] = extended_log(1 - special.ndtr((theta[-1] - mean_y[last]) / s_y))
    cdf_current_class = special.ndtr((theta[x[between].astype(int) - 1] - mean_y[between]) / s_y)
    cdf_previous_class = special.ndtr((theta[x[between].astype(int) - 2] - mean_y[between]) / s_y)
    log_pmf_x[between] = extended_log(cdf_current_class - cdf_previous_class)

    return log_pmf_x


def log_pmf_x_count(x: np.ndarray, mean_y: np.ndarray, w: float, s_y: float) -> np.ndarray:
    """Log pmf for count variables. This is the pmf derived from the pseudo-observations by subtracting cdfs for the natural numbers.

    Args:
        x (np.ndarray): The values in observation space. Must be label encoded starting at 1.
        mean_y (np.ndarray): The mean of the variable in pseudo-observation.
        w (float): The slope parameter of the transformation that maps the pseudo-observations to positive integers before flooring.
        s_y (float): The standard deviation of the variable in pseudo-observation space.
    Returns:
        np.ndarray: The log pmf of the variable under the assumption of it representing counts.
    """
    assert not np.isnan(x).any(), "x must not contain missing values"
    cdf_n = special.ndtr((g_inv(x, w) - mean_y) / s_y)
    cdf_n_plus_one = special.ndtr((g_inv(x + 1, w) - mean_y) / s_y)
    log_pmf_x = extended_log(cdf_n_plus_one - cdf_n)
    return log_pmf_x


# helpers
@njit
def mean_y_girolami_rogers(x: np.ndarray, Z: np.ndarray, b: np.ndarray):
    """Compute the inner term of the expected value for the pmf of the categorical data type. This requires a loop over the observations and
    is therefore implemented in numba for speed.

    Args:
        x (np.ndarray): The values in observation space. Must be label encoded starting at 1.
        Z (np.ndarray): The latent variable matrix. Must contain latent variables for all (non-missing) observations in the rows, becuase numba
        wants contiguous arrays.
        b (np.ndarray): The coefficient vectors for all classes of the observed variable.
    Returns:
        np.ndarray: The inner term of the expected value for the pmf of the categorical data type (zt(b(r) - b(r'))).
    """
    # dimensions
    N = x.shape[0]
    K, n_classes = b.shape

    # precompute differences of coefficients for each class
    b_diff = np.empty((n_classes, K, n_classes - 1))
    class_idxs = np.arange(n_classes)
    for cls in range(n_classes):
        b_diff[cls] = b[:, cls, np.newaxis] - b[:, class_idxs != cls]

    # main loop
    mean_y = np.zeros((N, n_classes - 1))
    for n in range(N):
        mean_y[n] = Z[n, :] @ b_diff[int(x[n] - 1)]

    return mean_y


def extended_log(x: np.ndarray) -> np.ndarray:
    """Logarithm that handles zero values by returning -inf. This avoids the warning raised by numpy and
    is useful for log pdfs and log pmfs that are explicitly allowed to be zero.

    Args:
        x (np.ndarray): The values to take the logarithm of.
    Returns:
        np.ndarray: The logarithm of the input values.
    """
    assert x.ndim == 1, f"masking assumes x.ndim == 1 but shape was {x.shape}"
    assert np.all(x >= 0), f"x < 0: {x[x < 0]}"
    result = np.zeros_like(x)
    result[x > 0] = np.log(x[x > 0])
    result[x == 0] = -np.inf
    return result
