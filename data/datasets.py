import numpy as np
import seaborn as sns
import pandas as pd
from typing import Tuple, Dict
import torch
import os
from torch.utils.data import Dataset, TensorDataset
from sdtd.data.util import Domain, download_dataset

sns.set(style="whitegrid", context="paper")


class SDTDDataset:

    SPLIT_SEED = 42

    def __init__(self,
                 domains: Dict[int, Domain],
                 label_encoding_starts_at_one: bool = False,
                 na_value: float = np.nan,
                 *args, **kwargs) -> None:
        self.domains = domains
        self.label_encoding_starts_at_one = label_encoding_starts_at_one
        self.na_value = na_value

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, data) -> None:
        # store dataset
        self._data = data

        # store dataset attributes
        self.n_classes = {m: int(np.nanmax(data[:, m], axis=0)) + 1 for m in range(data.shape[1]) if self.domains[m].is_discrete()}  # +1, because here always 0-based label encoding
        self.n_samples, self.n_features = self._data.shape

        # adjust encoding
        if self.label_encoding_starts_at_one:
            for m in range(self._data.shape[1]):
                missing = np.isnan(self._data[:, m])
                self._data[missing, m] = self.na_value
                if self.domains[m].is_discrete() or self.domains[m].is_binary():
                    self._data[~missing, m] += 1
                if m in self.n_classes:
                    self.n_classes[m] += 1

        # split into train, val and test set
        indices = range(self.n_samples)
        train_size = int(0.8 * self.n_samples)
        val_size = int(0.1 * self.n_samples)
        test_size = self.n_samples - train_size - val_size
        rng = np.random.default_rng(self.SPLIT_SEED)
        self._train_indices = rng.choice(indices, size=train_size, replace=False)
        self._val_indices = rng.choice(list(set(indices) - set(self._train_indices)), size=val_size, replace=False)
        self._test_indices = np.array(list(set(indices) - set(self._train_indices) - set(self._val_indices)))

    def to_numpy_dataset(self, split: str = None) -> np.ndarray:
        if split == 'train':
            return self.data[self._train_indices].astype(float)
        elif split == 'val':
            return self.data[self._val_indices].astype(float)
        elif split == 'test':
            return self.data[self._test_indices].astype(float)
        elif split is None:
            return self.data.astype(float)
        else:
            raise ValueError(f"Split {split} not supported.")

    def to_tensor_dataset(self, split: str = None) -> Dataset:
        numpy_dataset = self.to_numpy_dataset(split)
        tensors = [torch.from_numpy(numpy_dataset[:, m]).float() for m in range(self.n_features)]
        return TensorDataset(*tensors)

    @property
    def columns(self) -> list:
        return self._columns

    @columns.setter
    def columns(self, columns: list) -> None:
        self._columns = columns

    def to_df(self, split: str = None) -> pd.DataFrame:
        return pd.DataFrame(self.to_numpy_dataset(split), columns=self.columns)

    def __getitem__(self, idx) -> np.ndarray:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)


class SyntheticDataset(SDTDDataset):

    def __init__(self, domains: Dict[int, Domain], shape: Tuple[int, int], seed: int, *args, **kwargs) -> None:
        super().__init__(domains=domains, *args, **kwargs)

        self.shape = shape
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.columns = [f'$x_{m}$' for m in range(self.shape[1])]

    def plot(self, *args, **kwargs) -> None:
        df = self.to_df()
        if df.shape[1] == 1:
            g = sns.histplot(df, *args, **kwargs)
            g.set_ylabel('')
            g.legend_.remove()
        else:
            sns.pairplot(df, *args, **kwargs)


# CONTINUOUS

class RealDataset(SyntheticDataset):
    """A dataset of real-valued features.

    Attributes:
        loc (float): The mean of the normal distribution
        scale (float): The standard deviation of the normal distribution. Must be non-negative.
        shape (Tuple[int, int]): The number of samples and features in the dataset.
        seed (int): Random seed.
        data (np.ndarray): The dataset
        domains (Dict[int, Domain]): The domains for each feature.
    """

    def __init__(self, loc: float, scale: float, shape: Tuple[int, int], seed: int, *args, **kwargs) -> None:
        assert scale > 0, "Scale must be positive"

        super().__init__(
            domains={m: Domain('real') for m in range(shape[1])},
            shape=shape,
            seed=seed,
            *args, **kwargs
        )

        self.loc = loc
        self.scale = scale

        self.data = self._rng.normal(loc=self.loc, scale=self.scale, size=self.shape)


class PositiveDataset(SyntheticDataset):
    """Generate a dataset of positive-valued features from a Gamma distribution.

    Attributes:
        a (float): The shape of the gamma distribution. Must be non-negative.
        scale (float): The scale of the gamma distribution. Must be non-negative.
        shape (Tuple[int, int]): The number of samples and features in the dataset.
        seed (int): Random seed.
        data (np.ndarray): The dataset.
        domains (Dict[int, Domain]): The domains for each feature.
    """

    def __init__(self, a: float, scale: float, shape: Tuple[int, int], seed: int, *args, **kwargs) -> None:
        assert a > 0, "Shape must be positive"
        assert scale > 0, "Scale must be positive"

        super().__init__(
            domains={m: Domain('positive') for m in range(shape[1])},
            shape=shape,
            seed=seed,
            *args, **kwargs
        )

        self.a = a
        self.scale = scale

        self.data = self._rng.gamma(shape=self.a, scale=self.scale, size=self.shape)


class LognormalDataset(SyntheticDataset):
    """Generate a dataset of positive-valued features from a log-normal distribution.

    Args:
        loc (float): The mean of the normal distribution.
        scale (float): The scale of the normal distribution. Must be non-negative.
        shape (Tuple[int, int]): The number of samples and features in the dataset.
        seed (int): Random seed.
        data (np.ndarray): The dataset.
        domains (Dict[int, Domain]): The domains for each feature.
    """

    def __init__(self, loc: float, scale: float, shape: Tuple[int, int], seed: int, *args, **kwargs) -> None:
        assert scale > 0, "Scale must be positive"

        super().__init__(
            domains={m: Domain('positive') for m in range(shape[1])},
            shape=shape,
            seed=seed,
            *args, **kwargs
        )

        self.loc = loc
        self.scale = scale

        self.data = self._rng.lognormal(mean=self.loc, sigma=self.scale, size=self.shape)


class IntervalDataset(SyntheticDataset):
    """Generate a dataset of interval-valued features from a Beta distribution.

    Args:
        a (float): The alpha parameter of the Beta distribution. Must be positive.
        b (float): The beta parameter of the Beta distribution. Must be positive.
        theta (float): Scaling parameter.
        shape (Tuple[int, int]): The number of samples and features in the dataset.
        seed (int): Random seed.
        data (np.ndarray): The dataset.
        domains (Dict[int, Domain]): The domains for each feature.
    """

    def __init__(self, a: float, b: float, theta: float, shape: Tuple[int, int], seed: int, *args, **kwargs) -> None:
        assert a > 0, "Alpha must be positive"
        assert b > 0, "Beta must be positive"
        assert theta > 0, "Theta must be positive"

        super().__init__(
            domains={m: Domain('positive') for m in range(shape[1])},
            shape=shape,
            seed=seed,
            *args, **kwargs
        )

        self.a = a
        self.b = b
        self.theta = theta

        self.data = theta * self._rng.beta(a=self.a, b=self.b, size=self.shape)


# DISCRETE

class CategoricalDataset(SyntheticDataset):
    """Generate a dataset of a categorical feature completed with Gaussian features.

    Args:
        n_classes (int): The number of classes. Must be greater than 2.
        shape (Tuple[int, int]): The number of samples and features in the dataset. The last feature is the categorical feature.
        seed (int): Random seed.
        centroid_scale (float, optional): The scale of the class centroids. Must be positive. Defaults to 1.0.
        feature_scale (float, optional): The scale of the Gaussian features. Must be positive. Defaults to 1.0.
        data (np.ndarray): The dataset.
        domains (Dict[int, Domain]): The and domains for each feature.
    """

    def __init__(self, n_classes: int, shape: Tuple[int, int], seed: int, centroid_scale: float = 1.0, feature_scale: float = 1.0, *args, **kwargs) -> None:
        assert n_classes > 2, "Number of classes must be greater than 2, otherwise use a binary feature."
        assert centroid_scale > 0, "Centroid scale must be positive"
        assert feature_scale > 0, "Feature scale must be positive"

        super().__init__(
            domains={
                **{m: Domain('real') for m in range(shape[1] - 1)},
                **{shape[1] - 1: Domain('discrete')}
            },
            shape=shape,
            seed=seed,
            *args, **kwargs
        )

        self.centroid_scale = centroid_scale
        self.feature_scale = feature_scale

        # sample class probabilities
        class_probs = self._rng.dirichlet(alpha=np.ones(n_classes), size=1)

        # sample class assignments
        class_assignments = self._rng.multinomial(n=1, pvals=class_probs, size=shape[0]).argmax(axis=1)

        # sample class means
        centroids = self._rng.normal(loc=0, scale=self.centroid_scale, size=(n_classes, shape[1] - 1))

        # sample features
        features = self._rng.normal(loc=centroids[class_assignments], scale=self.feature_scale)

        # create dataset
        self.data = np.column_stack((features, class_assignments))


class OrdinalDataset(SyntheticDataset):
    """Generate a dataset of an ordinal feature completed with Gaussian features. The ordinal feature is generated from a latent variable, which is also included in the dataset.

    Args:
        n_classes (int): The number of classes. Must be greater than 2.
        shape (Tuple[int, int]): The number of samples and features in the dataset. The last two features are the latent variable and the ordinal feature.
        seed (int): Random seed.
        centroid_scale (float, optional): The scale of the class centroids. Must be positive. Defaults to 1.0.
        feature_scale (float, optional): The scale of the Gaussian features. Must be positive. Defaults to 1.0.
        data (np.ndarray): The dataset.
        domains (Dict[int, Domain]): The domains for each feature.
    """

    def __init__(self, n_classes: int, shape: Tuple[int, int], seed: int, centroid_scale: float = 1.0, feature_scale: float = 1.0, *args, **kwargs) -> None:
        assert n_classes > 2, "Number of classes must be greater than 2, otherwise use a binary feature."
        assert centroid_scale > 0, "Centroid scale must be positive"
        assert feature_scale > 0, "Feature scale must be positive"

        super().__init__(
            domains={
                **{m: Domain('real') for m in range(shape[1] - 2)},
                **{shape[1] - 2: Domain('positive')},
                **{shape[1] - 1: Domain('discrete')}
            },
            shape=shape,
            seed=seed,
            *args, **kwargs
        )

        self.centroid_scale = centroid_scale
        self.feature_scale = feature_scale

        # sample latent variable
        y = self._rng.uniform(low=0, high=n_classes, size=shape[0])

        # sample thresholds
        # thetas = self._rng.uniform(low=0, high=n_classes - 1, size=n_classes - 1)
        thetas = self._rng.uniform(low=0, high=n_classes, size=n_classes - 1)
        thetas = np.sort(thetas)
        thetas = np.insert(thetas, 0, -np.inf)
        thetas = np.append(thetas, np.inf)

        # derive class assignments
        x = np.empty((shape[0]), dtype=int)
        for i in range(n_classes):
            x[(thetas[i] <= y) & (y < thetas[i + 1])] = i

        # sample feature centroids
        centroids = self._rng.normal(loc=0, scale=self.centroid_scale, size=(n_classes, shape[1] - 2))

        # sample features
        features = self._rng.normal(loc=centroids[x], scale=self.feature_scale)

        # combine features and class assignments
        self.data = np.column_stack((features, y, x))


class CountDataset(SyntheticDataset):
    """Generate a dataset of a feature representing counts completed with Gaussian features. The count feature is generated from a latent variable, which is also included in the dataset.

    Args:
        a (float): The shape and 4*scale of the Gamma distribution. Indirectly determines the number of classes by controlling the length of the tail of the Gamma distribution. Must be positive.
        shape (Tuple[int, int]): The number of samples and features in the dataset. The last two features are the latent variable and the count feature.
        seed (int): Random seed.
        centroid_scale (float, optional): The scale of the class centroids. Must be positive. Defaults to 1.0.
        feature_scale (float, optional): The scale of the Gaussian features. Must be positive. Defaults to 1.0.
        data (np.ndarray): The dataset.
        domains (Dict[int, Domain]): The domains for each feature.
    """

    def __init__(self, a: float, shape: Tuple[int, int], seed: int, centroid_scale: float = 1.0, feature_scale: float = 1.0, *args, **kwargs) -> None:
        assert a > 0, "Shape must be positive"

        super().__init__(
            domains={
                **{m: Domain('real') for m in range(shape[1] - 2)},
                **{shape[1] - 2: Domain('positive')},
                **{shape[1] - 1: Domain('discrete')}
            },
            shape=shape,
            seed=seed,
            *args, **kwargs
        )

        self.a = a
        self.centroid_scale = centroid_scale
        self.feature_scale = feature_scale

        # sample latent variable
        correlated_feature = self._rng.gamma(shape=self.a, scale=self.a / 4, size=shape[0])

        # derive x by flooring
        count_variable = np.floor(correlated_feature).astype(int)
        n_classes = count_variable.max().astype(int) + 1

        # class centroids
        centroids = self._rng.normal(loc=0, scale=self.centroid_scale, size=(n_classes, shape[1] - 2))

        # features
        features = self._rng.normal(loc=centroids[count_variable], scale=self.feature_scale, size=(shape[0], shape[1] - 2))

        # final dataset
        self.data = np.column_stack((features, correlated_feature, count_variable))


class PoissonDataset(SyntheticDataset):
    """Generate a dataset of a Poisson feature completed with Gaussian features. The Poisson feature is generated from a uniform latent variable, which is also included in the dataset.

    Args:
        w (float): Coefficient of the latent variable. Indirectly determines the number of classes by controlling the tail of the Poisson variable. Must be positive.
        shape (Tuple[int, int]): The number of samples and features in the dataset. The last two features are the latent variable and the count feature.
        seed (int): Random seed.
        centroid_scale (float, optional): The scale of the class centroids. Must be positive. Defaults to 1.0.
        feature_scale (float, optional): The scale of the Gaussian features. Must be positive. Defaults to 1.0.
        data (np.ndarray): The dataset.
        domains (Dict[int, Domain]): The domains for each feature.
    """

    def __init__(self, w: float, shape: Tuple[int, int], seed: int, centroid_scale: float = 1.0, feature_scale: float = 1.0, *args, **kwargs) -> None:
        assert w > 0, "Coefficient must be positive"

        super().__init__(
            domains={
                **{m: Domain('real') for m in range(shape[1] - 2)},
                **{shape[1] - 2: Domain('positive')},
                **{shape[1] - 1: Domain('discrete')}
            },
            shape=shape,
            seed=seed,
            *args, **kwargs
        )

        self.w = w
        self.centroid_scale = centroid_scale
        self.feature_scale = feature_scale

        # sample latent variable
        correlated_feature = self._rng.uniform(low=0, high=1, size=shape[0])

        # derive count variable
        count_variable = self._rng.poisson(lam=np.exp(self.w * correlated_feature))
        n_classes = count_variable.max().astype(int) + 1

        # class centroids
        centroids = self._rng.normal(loc=0, scale=self.centroid_scale, size=(n_classes, shape[1] - 2))

        # complete with Gaussian features
        features = self._rng.normal(loc=centroids[count_variable], scale=self.feature_scale, size=(shape[0], shape[1] - 2))

        # final dataset
        self.data = np.column_stack((features, correlated_feature, count_variable))


class UCIDataset(SDTDDataset):

    SUPPORTED_SPLITS = ['train', 'val', 'test']

    def __init__(self, root_dir: str, url: str, dtypes: Dict[str, str | type], *args, **kwargs) -> None:
        super().__init__(domains={
            **{m: Domain('positive') for m, (column, dtype) in enumerate(dtypes.items()) if dtype == float},
            **{m: Domain('discrete') for m, (column, dtype) in enumerate(dtypes.items()) if dtype == int or dtype == 'category'},
            **{m: Domain('binary') for m, (column, dtype) in enumerate(dtypes.items()) if dtype == bool},
        }, *args, **kwargs)

        data_path = os.path.join(root_dir, os.path.basename(url))

        # download dataset if necessary
        if not os.path.exists(data_path):
            os.makedirs(root_dir)
            download_dataset(url, data_path)

        # read csv
        df = self._read_csv(data_path, dtypes)

        # encode categorical columns
        for col in df.columns:
            if df[col].dtype.name == 'category':
                df[col] = df[col].cat.codes
                df.loc[df[col] == -1, col] = np.nan
                valid = (df[col] >= 0) | np.isnan(df[col])
                assert valid.all(), f"Column {col} has invalid values: {df[col][~valid].unique()}"

        # convert to numpy
        self.data = df.to_numpy().astype(float)

        # store column names to allow reconstructing the data frame when necessary
        self.columns = df.columns

    def _read_csv(self, data_path: str, dtypes: Dict[str, str | type]) -> None:
        raise NotImplementedError


class GermanDataset(UCIDataset):
    """ Load the German credit dataset with the documented domains.

    Args:
        root_dir (str): The root directory of the dataset. Expects the files 'train.csv', 'val.csv', and 'test.csv' without headers or index.
        data (np.ndarray): The dataset.
        domains (Dict[str, str]): The domains of each feature.
    """

    DTYPES = {
        "Account status": 'category',
        "Duration": int,
        "Credit history": 'category',
        "Purpose": 'category',
        "Credit amount": float,
        "Savings": 'category',
        "Employment since": 'category',
        "Installment rate": int,
        "Personal status": 'category',
        "Other debtors": 'category',
        "Resident since": int,
        "Property": 'category',
        "Age": int,
        "Other installment plans": 'category',
        "Housing": 'category',
        "Existing credits": int,
        "Job": 'category',
        "People liable for": int,
        "Telephone": bool,
        "Foreign": bool,
        "Credit risk": bool,
    }

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

    def __init__(self, root_dir: str, *args, **kwargs) -> None:
        super().__init__(root_dir, self.URL, self.DTYPES, *args, **kwargs)
        assert self.data.shape[1] == 21

    def _read_csv(self, data_path: str, dtypes: Dict[str, str | type]) -> None:
        return pd.read_csv(
            data_path,
            sep=' ',
            header=None,
            index_col=False,
            dtype={col: dtype if dtype != bool else 'category' for col, dtype in dtypes.items()},
            names=list(dtypes.keys()),
            engine='python'
        )


class AdultDataset(UCIDataset):
    """ Load the Adult dataset with the documented domains.

    Args:
        root_dir (str): The root directory of the dataset. Expects the files 'train.csv', 'val.csv', and 'test.csv' without headers or index.
        dataset (np.ndarray): The dataset.
        domains (Dict[str, str]): The domains of each feature.
    """

    DTYPES = {
        'Age': int,
        'Workclass': 'category',
        'Final Weight': float,
        'Education': 'category',
        'Years of Education': int,
        'Marital Status': 'category',
        'Occupation': 'category',
        'Relationship': 'category',
        'Race': 'category',
        'Sex': bool,
        'Capital Gain': float,
        'Capital Loss': float,
        'Hours per Week': int,
        'Native Country': 'category',
        'Income': bool
    }

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    def __init__(self, root_dir: str, *args, **kwargs) -> None:
        super().__init__(root_dir, self.URL, self.DTYPES, *args, **kwargs)
        assert self.data.shape[1] == 15

    def _read_csv(self, data_path: str, dtypes: Dict[str, str | type]) -> None:
        return pd.read_csv(
            data_path,
            sep=', ',
            header=None,
            index_col=False,
            dtype={col: dtype if dtype != bool else 'category' for col, dtype in dtypes.items()},
            names=list(dtypes.keys()),
            engine='python',
            na_values='?'
        )
