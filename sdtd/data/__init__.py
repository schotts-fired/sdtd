from sdtd.data import datasets
import numpy as np
from omegaconf import DictConfig

SYNTHETIC_DATASETS = ['real', 'positive', 'lognormal', 'interval', 'categorical', 'ordinal', 'count', 'poisson']
UCI_DATASETS = ['adult', 'german']


def load_dataset(name,
                 label_encoding_starts_at_one: bool = False,
                 na_value: float = np.nan,
                 **dataset_cfg):
    assert name in (SYNTHETIC_DATASETS + UCI_DATASETS), f"Dataset {name} not supported. Supported datasets: {SYNTHETIC_DATASETS + UCI_DATASETS}"

    if name == 'real':
        return datasets.RealDataset(**dataset_cfg,
                                    na_value=na_value,
                                    label_encoding_starts_at_one=label_encoding_starts_at_one)
    elif name == 'positive':
        return datasets.PositiveDataset(**dataset_cfg,
                                        na_value=na_value,
                                        label_encoding_starts_at_one=label_encoding_starts_at_one)
    elif name == 'lognormal':
        return datasets.LognormalDataset(**dataset_cfg,
                                         na_value=na_value,
                                         label_encoding_starts_at_one=label_encoding_starts_at_one)
    elif name == 'interval':
        return datasets.IntervalDataset(**dataset_cfg,
                                        na_value=na_value,
                                        label_encoding_starts_at_one=label_encoding_starts_at_one)
    elif name == 'categorical':
        return datasets.CategoricalDataset(**dataset_cfg,
                                           na_value=na_value,
                                           label_encoding_starts_at_one=label_encoding_starts_at_one)
    elif name == 'ordinal':
        return datasets.OrdinalDataset(**dataset_cfg,
                                       na_value=na_value,
                                       label_encoding_starts_at_one=label_encoding_starts_at_one)
    elif name == 'count':
        return datasets.CountDataset(**dataset_cfg,
                                     na_value=na_value,
                                     label_encoding_starts_at_one=label_encoding_starts_at_one)
    elif name == 'poisson':
        return datasets.PoissonDataset(**dataset_cfg,
                                       na_value=na_value,
                                       label_encoding_starts_at_one=label_encoding_starts_at_one)
    elif name == 'adult':
        return datasets.AdultDataset(**dataset_cfg,
                                     na_value=na_value,
                                     label_encoding_starts_at_one=label_encoding_starts_at_one)
    elif name == 'german':
        return datasets.GermanDataset(**dataset_cfg,
                                      na_value=na_value,
                                      label_encoding_starts_at_one=label_encoding_starts_at_one)
