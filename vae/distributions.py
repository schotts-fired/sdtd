import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class OrderedLogistic:

    def __init__(self, loc: torch.Tensor, thresholds: torch.Tensor):
        self.loc = loc
        self.thresholds = thresholds
        self.distribution = self._init_distribution()
        self.probs = self.distribution.probs

    def _init_distribution(self):
        cum_class_probs = F.sigmoid(self.thresholds - self.loc[:, None])
        class_probs = torch.cat([cum_class_probs, torch.ones_like(self.loc[:, None])], dim=1) - torch.cat([torch.zeros_like(self.loc[:, None]), cum_class_probs], dim=1)
        assert class_probs.sum(dim=1).allclose(torch.ones_like(self.loc[:, None])), class_probs.sum(dim=1)
        class_probs[class_probs < 0] = 0.0  # avoid numerical issues
        return D.Categorical(probs=class_probs)

    def log_prob(self, x: torch.Tensor):
        return self.distribution.log_prob(x)

    def sample(self):
        return self.distribution.sample()

    def __repr__(self):
        return f"OrderedLogistic(loc: {self.loc.shape}, thresholds: {self.thresholds.shape})"

    def __str__(self):
        return f"OrderedLogistic(loc: {self.loc.shape}, thresholds: {self.thresholds.shape})"


class Mixture:

    def __init__(self, mixture_distribution: D.Categorical, component_distributions: list):
        self.mixture_distribution = mixture_distribution
        self.component_distributions = component_distributions

    def log_prob(self, x):
        log_mix_prob = self.mixture_distribution.logits  # they are already normalized, because they are the logs of the dirichlet samples
        log_comp_prob = torch.column_stack([comp.log_prob(x) for comp in self.component_distributions])
        log_likelihood = torch.logsumexp(log_mix_prob + log_comp_prob, dim=1)
        assert log_likelihood.shape == (x.shape[0], ), log_likelihood.shape
        return log_likelihood

    def sample(self):
        component_sample = self.sample_components()
        if self.mixture_distribution.batch_shape == ():  # for generative sampling, need to explicitly tell to get N samples
            mixture_sample = self.mixture_distribution.sample((component_sample.shape[0], ))
        else:  # for reconstruction sampling
            mixture_sample = self.mixture_distribution.sample()
        sample = torch.empty_like(component_sample[:, 0], dtype=torch.float32)
        for i in range(len(self.component_distributions)):
            sample[mixture_sample == i] = component_sample[mixture_sample == i, i].squeeze()
        return sample

    def sample_components(self):
        component_sample = torch.column_stack([component.sample().float() for component in self.component_distributions])
        return component_sample

    def __repr__(self):
        return f"Mixture({self.mixture_distribution}, {self.component_distributions})"

    def __str__(self):
        return f"Mixture({self.mixture_distribution}, {self.component_distributions})"
