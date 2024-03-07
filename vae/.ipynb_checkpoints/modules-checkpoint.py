import torch
import torch.nn.functional as F
import torch.distributions as D
from torch import nn, optim
from typing import List, Dict
from distributions import OrderedLogistic, Mixture


class MLP(nn.Module):

    SUPPORTED_ACTIVATIONS = ["relu", "tanh"]

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 activation: str,
                 n_layers: int):
        super().__init__()
        assert input_dim > 0
        assert output_dim > 0
        assert hidden_dim >= 0
        assert (not n_layers > 1) and (hidden_dim == 0)
        assert activation in MLP.SUPPORTED_ACTIVATIONS
        assert n_layers > 0
        # main modules
        modules = []
        for _ in range(n_layers - 1):
            modules.append(nn.Linear(input_dim, hidden_dim))
            if activation == "relu":
                modules.append(nn.ReLU())
            elif activation == "tanh":
                modules.append(nn.Tanh())
            input_dim = hidden_dim
        modules.append(nn.Linear(input_dim, output_dim))
        self.model = nn.Sequential(*modules)

    def forward(self, X):
        return self.model(X)


class MultiHeadMLP(nn.Module):

    def __init__(self, n_heads: int, **mlp_kwargs):
        super().__init__()
        self.heads = nn.ModuleList([MLP(**mlp_kwargs) for _ in range(n_heads)])

    def forward(self, X):
        return torch.stack([head[m](X) for head in self.heads], dim=1)


class BatchNorm(nn.Module):

    def __init__(self, domains: List[str], n_classes: Dict[int, int]):
        super().__init__()
        self.domains = domains
        self.n_classes = n_classes

        # compute output dimensions
        n_pos = sum([1 if domain == "positive" else 0 for domain in domains])
        output_dim_pos = 2
        n_cont = sum([1 if domain == "continuous" else 0 for domain in domains])
        output_dim_cont = 1
        n_disc = sum([1 if domain == "discrete" else 0 for domain in domains])
        output_dim_disc = 2 * max([n_classes for n_classes in self.n_classes.values()], default=0) + 1
        self.output_dim = n_pos * output_dim_pos + n_cont * output_dim_cont + n_disc * output_dim_disc

    def real_transform(self, X: torch.Tensor):
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        return (X - mu) / sigma, mu, sigma

    def pos_transform(self, X: torch.Tensor):
        mu = X.log().mean(axis=0)
        sigma = X.log().std(axis=0)
        return (X.log() - mu) / sigma, mu, sigma

    def cat_transform(self, X: torch.Tensor, n_classes: torch.float32):
        return F.one_hot(X, num_classes=n_classes).type(torch.float32)

    def ord_transform(self, X: torch.Tensor, n_classes: torch.float32):
        """Input is assumed to be label encoded starting at 0. For three classes,
        the first class is encoded as 000 and the last as 110. Technically, we could make the
        output two dimensional, but this clashes with the categorical one-hot encoding."""
        assert X.max() < n_classes
        assert X.min() >= 0
        assert X.dtype == torch.long
        assert X.ndim == 1
        batch_size = len(X)
        labels = torch.arange(n_classes, device=X.device).view(1, -1).expand(batch_size, -1)
        batch_values = X.view(-1, 1)
        encoding = (labels < batch_values).type(torch.float32)
        assert (encoding[:, -1] == 0).all()
        return encoding

    def count_transform(self, X: torch.Tensor):
        return torch.where(X == 0, torch.zeros_like(X), torch.log(X))

    def forward(self, X: torch.Tensor):
        N, M = X.shape
        stats_real = {}
        stats_pos = {}
        X_list = []
        for m in range(M):
            if self.domains[m] == "positive":
                assert (X[:, m] >= 0).all(), "Positive variables must be strictly positive."
                X_real, *stats_real[m] = self.real_transform(X[:, m])
                X_pos, *stats_pos[m] = self.pos_transform(X[:, m])
                X_list.extend([X_real, X_pos])
            elif self.domains[m] == "continuous":
                X_real, *stats_real[m] = self.real_transform(X[:, m])
                X_list.append(X_real)
            elif self.domains[m] == "discrete":
                X_cat = self.cat_transform(X[:, m].long(), n_classes=self.n_classes[m])
                X_ord = self.ord_transform(X[:, m].long(), n_classes=self.n_classes[m])
                X_count = self.count_transform(X[:, m])
                X_list.extend([X_cat, X_ord, X_count])
        X_tilde = torch.column_stack(X_list)
        assert not torch.isnan(X_tilde).any(), f"NaNs in input {X_tilde}."
        assert not torch.isinf(X_tilde).any(), f"INFs in input {X_tilde}."
        return X_tilde, stats_real, stats_pos


class EncoderW(nn.Module):

    def __init__(self, domains: List[str], alpha: float = 1.0):
        super().__init__()
        self.n_features = len(domains)
        self.prior_alpha = nn.ParameterList()
        self.posterior_alpha = nn.ParameterList()
        self.prior = []
        for m in range(len(domains)):
            if domains[m] == "positive":
                initial_alpha = [alpha] * 2
            elif domains[m] == "continuous":
                initial_alpha = [alpha]
            elif domains[m] == "discrete":
                initial_alpha = [alpha] * 3
            self.prior_alpha.append(nn.Parameter(
                torch.tensor(initial_alpha), requires_grad=False
            ))
            self.prior.append(D.Dirichlet(self.prior_alpha[m]))
            self.posterior_alpha.append(nn.Parameter(
                nn.Parameter(torch.tensor(initial_alpha), requires_grad=True)
            ))

    def forward(self, X):
        posterior = []
        for m in range(self.n_features):
            alpha = F.softplus(self.posterior_alpha[m])
            posterior.append(D.Dirichlet(alpha))
        return posterior


class EncoderZ(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 activation: str,
                 n_layers: int):
        super().__init__()
        self.prior_mean = nn.Parameter(torch.zeros(output_dim), requires_grad=False)
        self.prior_variance = nn.Parameter(torch.ones(output_dim), requires_grad=False)
        self.prior = D.Independent(D.Normal(
            self.prior_mean, self.prior_variance
        ), 1)
        self.posterior_mean = MLP(input_dim, output_dim, hidden_dim, activation, n_layers)
        self.posterior_variance = MLP(input_dim, output_dim, hidden_dim, activation, n_layers)

    def forward(self, X):
        # posterior Z
        loc = self.posterior_mean(X)
        scale = F.softplus(self.posterior_variance(X)).sqrt()
        posterior_Z = D.Independent(D.Normal(loc, scale), 1)
        return posterior_Z


class DecoderY(nn.Module):

    def __init__(self,
                 # dataset attributes
                 n_features: int,

                 # hyperparameters
                 shared: bool,

                 # architecture
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 activation: str,
                 n_layers: int):
        super().__init__()
        self.shared = shared
        self.n_features = n_features
        self.output_dim = output_dim
        if self.shared:
            self.g = MLP(input_dim,
                         output_dim * n_features,
                         hidden_dim,
                         activation,
                         n_layers)
        else:
            self.g = MultiHeadMLP(n_heads=n_features,
                                  input_dim=input_dim,
                                  output_dim=output_dim,
                                  hidden_dim=hidden_dim,
                                  activation=activation,
                                  n_layers=n_layers)

    def forward(self, Z):
        Y = self.g(Z)
        if self.shared:
            Y = Y.reshape((-1, self.n_features, self.output_dim))
        return Y


class DecoderX(nn.Module):

    def __init__(self,
                 # dataset attributes
                 domains: List[str],
                 n_classes: Dict[int, int],

                 # hyperparameters
                 trainable_variance: bool,

                 # architecture
                 input_dim: int,
                 hidden_dim: int,
                 activation: str,
                 n_layers: int):
        super().__init__()
        # dataset attributes
        self.domains = domains
        self.n_features = len(self.domains)

        # hyperparameters
        self.trainable_variance = trainable_variance

        # decoders
        self.mean_real = nn.ModuleDict()
        self.mean_pos = nn.ModuleDict()
        if self.trainable_variance:
            self.variance_real = nn.ModuleDict()
            self.variance_pos = nn.ModuleDict()
        else:
            self.variance_real = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            self.variance_pos = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.logits_cat = nn.ModuleDict()
        self.parameters_ord = nn.ModuleDict()
        self.rate_count = nn.ModuleDict()
        for m in range(len(domains)):
            if domains[m] == 'positive' or domains[m] == 'continuous':
                self.mean_real[str(m)] = MLP(input_dim, 1, hidden_dim, activation, n_layers)
                if self.trainable_variance:
                    self.variance_real[str(m)] = MLP(input_dim, 1, hidden_dim, activation, n_layers)
            if domains[m] == 'positive':
                self.mean_pos[str(m)] = MLP(input_dim, 1, hidden_dim, activation, n_layers)
                if self.trainable_variance:
                    self.variance_pos[str(m)] = MLP(input_dim, 1, hidden_dim, activation, n_layers)
            if domains[m] == 'discrete':
                self.logits_cat[str(m)] = MLP(input_dim, n_classes[m], hidden_dim, activation, n_layers)
                self.parameters_ord[str(m)] = MLP(input_dim, n_classes[m] - 1, hidden_dim, activation, n_layers)
                self.rate_count[str(m)] = MLP(input_dim, 1, hidden_dim, activation, n_layers)

    def decoder_real(self, y, m, mu, sigma):
        loc = self.mean_real[str(m)](y)
        loc = sigma * loc + mu if mu and sigma else loc
        if self.trainable_variance:
            scale = self.variance_real[str(m)](y)
            assert scale.shape == (y.shape[0], 1), scale.shape
            scale = sigma * scale if mu and sigma else scale
        else:
            scale = self.variance_real
            # no denormalization needed
        assert loc.shape == (y.shape[0], 1), loc.shape
        return D.Normal(loc=loc.squeeze(), scale=scale.squeeze())

    def decoder_pos(self, y, m, mu, sigma):
        loc = F.softplus(self.mean_real[str(m)](y))
        loc = sigma * loc + mu if mu and sigma else loc
        if self.trainable_variance:
            scale = self.variance_pos[str(m)](y)
            assert scale.shape == (y.shape[0], 1), scale.shape
            scale = sigma * scale if mu and sigma else scale
        else:
            scale = self.variance_pos
            # no denormalization needed
        assert loc.shape == (y.shape[0], 1), loc.shape
        return D.LogNormal(loc=loc.squeeze(), scale=scale.squeeze())

    def decoder_cat(self, y, m):
        logits = self.logits_cat[str(m)](y)
        return D.Categorical(logits=logits.squeeze())

    def decoder_ord(self, y, m):
        parameters = self.parameters_ord[str(m)](y)
        loc = parameters[:, 0]
        thresholds = F.pad(torch.cumsum(F.softplus(parameters[:, 1:]), dim=1), (1, 0), value=0)
        assert loc.shape == (y.shape[0], ), loc.shape
        return OrderedLogistic(loc=loc.squeeze(), thresholds=thresholds.squeeze())

    def decoder_count(self, y, m):
        rate = torch.exp(self.rate_count[str(m)](y))
        assert rate.shape == (y.shape[0], 1), rate.shape
        return D.Poisson(rate=rate.squeeze())

    def forward(self, Y, W, stats_real, stats_pos):
        mixtures = []
        for m in range(self.n_features):
            y = Y[:, m]
            mixture_distribution = D.Categorical(W[m])
            component_distributions = []
            if self.domains[m] == "continuous" or self.domains[m] == "positive":
                component_distributions.append(self.decoder_real(y, m, *stats_real[m]))
            if self.domains[m] == "positive":
                component_distributions.append(self.decoder_pos(y, m, *stats_pos[m]))
            elif self.domains[m] == "discrete":
                component_distributions.append(self.decoder_cat(y, m))
                component_distributions.append(self.decoder_ord(y, m))
                component_distributions.append(self.decoder_count(y, m))
            mixture = Mixture(mixture_distribution, component_distributions)
            mixtures.append(mixture)
        return mixtures

