from torch.utils.data import Dataset
import pandas as pd
from typing import Any
import torch
from scipy import stats
import numpy as np
from typing import Dict, List
from datasets import synthetic


class SDTDDataset(Dataset):

    def __init__(self, dataset: str, **kwargs) -> None:
        # data
        if dataset == 'real':
            self.data, self.domains = synthetic.make_real(**kwargs)
        elif dataset == 'positive':
            self.data, self.domains = synthetic.make_positive(**kwargs)
        elif dataset == 'lognormal':
            self.data, self.domains = synthetic.make_lognormal(**kwargs)
        elif dataset == 'interval':
            self.data, self.domains = synthetic.make_interval(**kwargs)
        elif dataset == 'categorical':
            self.data, self.domains = synthetic.make_categorical(**kwargs)
        elif dataset == 'ordinal':
            self.data, self.domains = synthetic.make_ordinal(**kwargs)
        elif dataset == 'count':
            self.data, self.domains = synthetic.make_count(**kwargs)
        elif dataset == 'poisson':
            self.data, self.domains = synthetic.make_poisson(**kwargs)

        # let domains be indexed by ints
        self.feature_names = domains.keys()
        self.domains = {}
        for m, v in enumerate(domains.values()):
            self.domains[m] = v

        # metadata
        self.n_features = data.shape[1]
        self.n_classes = {m: len(np.unique(data[:, m])) for m, domain in enumerate(domains.values()) if domain == 'discrete'}

    def __getitem__(self, index: int) -> Any:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
