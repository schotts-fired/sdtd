import torch
import torch.nn.functional as F
import torch.distributions as D
from torch import nn, optim
from typing import List, Dict
from .distributions import OrderedLogistic, Mixture


class MLP(nn.Module):

    SUPPORTED_ACTIVATIONS = ["relu", "tanh"]

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 activation: str,
                 n_layers: int):
        super().__init__()
        assert not (n_layers > 1 and activation is None), "Cannot have more than one layer with no activation"
        assert input_dim > 0
        assert output_dim > 0
        assert hidden_dim is None or hidden_dim > 0, f"hidden_dim: {hidden_dim}"
        assert not (n_layers > 1 and hidden_dim is None), (n_layers, hidden_dim)
        assert activation is None or activation in MLP.SUPPORTED_ACTIVATIONS
        assert n_layers > 0

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


class Encoder_W(nn.Module):

    def __init__(self, prior_alpha: List[float]):
        super().__init__()
        self.q_w_alpha = nn.Parameter(torch.tensor(prior_alpha), requires_grad=True)

    def forward(self, x):
        q_w = D.Dirichlet(F.softplus(self.q_w_alpha))
        return q_w.expand((x.shape[0],))


class Encoder_Z(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 activation: str,
                 n_layers: int):
        super().__init__()
        self.q_z_loc = MLP(input_dim, output_dim, hidden_dim, activation, n_layers)
        self.q_z_log_scale = MLP(input_dim, output_dim, hidden_dim, activation, n_layers)

    def forward(self, X):
        loc = self.q_z_loc(X)
        scale = self.q_z_log_scale(X).exp()
        return D.Normal(loc, scale)


class Decoder_Y(nn.Module):
    """Intermediate decoder between encoder and per-data type decoders.

    Can be shared or multi-headed and disabled by setting n_layers=0. If it is disabled,
    it will simply return a view of the input tensor, such that indexing can be done
    as if it was enabled.

    Args:
        n_features (int): Number of features in the dataset.
        shared (bool): Whether to use a shared or multi-headed architecture.
        input_dim (int): Input dimension.
        output_dim (int): Output dimension.
        hidden_dim (int): Hidden dimension.
        activation (str): Activation function.
        n_layers (int): Number of layers. Set to 0 to disable this decoder.
    """

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
        assert n_layers is None or n_layers > 0, f"n_layers: {n_layers}"
        self.shared = shared
        self.n_features = n_features
        self.output_dim = output_dim
        self.disabled = n_layers is None
        if not self.disabled:
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
        else:
            self.g = nn.Identity()

    def forward(self, Z):
        Y = self.g(Z)

        if self.disabled:
            return Y.unsqueeze(1).expand(-1, self.n_features, -1)

        if self.shared:
            return Y.reshape((-1, self.n_features, self.output_dim))

        return Y


class Decoder_X_cont(nn.Module):

    def __init__(self,
                 # architecture
                 input_dim: int,
                 hidden_dim: int,
                 activation: str,
                 n_layers: int):
        super().__init__()
        self.heads = nn.ModuleDict()
        self.heads['real'] = Decoder_X_real(input_dim, hidden_dim, activation, n_layers)
        self.heads['pos'] = Decoder_X_pos(input_dim, hidden_dim, activation, n_layers)

    def forward(self,
                x: torch.Tensor,
                w: torch.Tensor = None,
                batch_loc_real: torch.Tensor = None,
                batch_scale_real: torch.Tensor = None,
                batch_loc_pos: torch.Tensor = None,
                batch_scale_pos: torch.Tensor = None) -> D.Distribution:
        mixture_distribution = D.Categorical(w)
        component_distributions = [
            self.heads['real'](x, batch_loc_real, batch_scale_real),
            self.heads['pos'](x, batch_loc_pos, batch_scale_pos)
        ]
        return Mixture(mixture_distribution, component_distributions)


class Decoder_X_disc(nn.Module):

    def __init__(self,
                 # dataset attributes
                 n_classes: int,

                 # architecture
                 global_thresholds: bool,
                 input_dim: int,
                 hidden_dim: int,
                 activation: str,
                 n_layers: int):
        super().__init__()
        self.heads = nn.ModuleDict()
        self.heads['cat'] = Decoder_X_cat(n_classes, input_dim, hidden_dim, activation, n_layers)
        self.heads['ord'] = Decoder_X_ord(n_classes, global_thresholds, input_dim, hidden_dim, activation, n_layers)
        self.heads['count'] = Decoder_X_count(input_dim, hidden_dim, activation, n_layers)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        mixture_distribution = D.Categorical(w)
        component_distributions = [self.heads['cat'](x), self.heads['ord'](x), self.heads['count'](x)]
        return Mixture(mixture_distribution, component_distributions)


class Decoder_X_bin(nn.Module):

    def __init__(self,
                 # architecture
                 input_dim: int,
                 hidden_dim: int,
                 activation: str,
                 n_layers: int):
        super().__init__()
        self.gamma = MLP(input_dim, 1, hidden_dim, activation, n_layers)

    def forward(self, x: torch.Tensor) -> D.Bernoulli:
        logits = self.gamma(x)
        assert logits.shape == (x.shape[0], 1), logits.shape
        return D.Bernoulli(logits=logits)


class Decoder_X_real(nn.Module):

    def __init__(self,
                 # architecture
                 input_dim: int,
                 hidden_dim: int,
                 activation: str,
                 n_layers: int):
        super().__init__()
        self.gamma = MLP(input_dim, 1, hidden_dim, activation, n_layers)
        self.moving_loc = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.moving_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.momentum = 0.99

    def forward(self, x: torch.Tensor,
                batch_loc: torch.Tensor | float = None,
                batch_scale: torch.Tensor | float = None) -> D.Normal:
        # input validation
        assert not (batch_loc is None and batch_scale is not None)
        assert not (batch_loc is not None and batch_scale is None)
        if isinstance(batch_loc, torch.Tensor):
            assert batch_loc.shape == (), batch_loc.shape
        if isinstance(batch_scale, torch.Tensor):
            assert batch_scale.shape == (), batch_scale.shape
        assert not self.training or (batch_loc is not None and batch_scale is not None), "mean and std must be provided during training"

        # track moving average of loc and scale for generation
        if batch_loc is not None and batch_scale is not None:
            self.moving_loc -= (self.moving_loc - batch_loc) * (1 - self.momentum)
            self.moving_scale -= (self.moving_scale - batch_scale) * (1 - self.momentum)
        else:
            batch_loc = self.moving_loc
            batch_scale = self.moving_scale

        # compute parameters
        parameters = self.gamma(x)
        assert parameters.shape == (x.shape[0], 1), parameters.shape

        # extract and denormalize loc
        loc = parameters[:, 0]
        loc = batch_scale * loc + batch_loc
        assert loc.shape == (x.shape[0], ), loc.shape

        # denormalize scale
        scale = batch_scale
        assert (scale == 1.0) or (scale == batch_scale) or (scale.shape == (x.shape[0], )), scale.shape

        return D.Normal(loc=loc, scale=scale)


class Decoder_X_pos(nn.Module):

    def __init__(self,
                 # architecture
                 input_dim: int,
                 hidden_dim: int,
                 activation: str,
                 n_layers: int):
        super().__init__()
        self.gamma = MLP(input_dim, 1, hidden_dim, activation, n_layers)
        self.momentum = 0.99
        self.moving_loc = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.moving_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, x: torch.Tensor,
                batch_loc: torch.Tensor | float = None,
                batch_scale: torch.Tensor | float = None) -> D.LogNormal:
        # input validation
        assert not (batch_loc is None and batch_scale is not None)
        assert not (batch_loc is not None and batch_scale is None)
        if isinstance(batch_loc, torch.Tensor):
            assert batch_loc.shape == (), batch_loc.shape
        if isinstance(batch_scale, torch.Tensor):
            assert batch_scale.shape == (), batch_scale.shape
        assert not self.training or (batch_loc is not None and batch_scale is not None), "mean and std must be provided during training"

        # track moving average of loc and scale for generation
        if batch_loc is not None and batch_scale is not None:
            self.moving_loc -= (self.moving_loc - batch_loc) * (1 - self.momentum)
            self.moving_scale -= (self.moving_scale - batch_scale) * (1 - self.momentum)
        else:
            batch_loc = self.moving_loc
            batch_scale = self.moving_scale

        # compute parameters
        parameters = self.gamma(x)
        assert parameters.shape == (x.shape[0], 1), parameters.shape

        # extract and denormalize loc
        loc = F.softplus(parameters[:, 0])
        loc = batch_scale * loc + batch_loc
        assert loc.shape == (x.shape[0], ), loc.shape

        # extract and denormalize scale
        scale = batch_scale
        assert (scale == 1.0) or (scale == batch_scale) or (scale.shape == (x.shape[0], )), scale.shape

        return D.LogNormal(loc=loc, scale=scale)


class Decoder_X_cat(nn.Module):

    def __init__(self,
                 # dataset attributes
                 n_classes: int,

                 # architecture
                 input_dim: int,
                 hidden_dim: int,
                 activation: str,
                 n_layers: int):
        super().__init__()
        assert n_classes > 2, f"n_classes: {n_classes}"
        self.n_classes = n_classes
        self.gamma = MLP(input_dim, n_classes, hidden_dim, activation, n_layers)

    def forward(self, x: torch.Tensor) -> D.Categorical:
        logits = self.gamma(x)
        assert logits.shape == (x.shape[0], self.n_classes), logits.shape
        return D.Categorical(logits=logits)


class Decoder_X_ord(nn.Module):

    def __init__(self,
                 # dataset attributes
                 n_classes: int,

                 # architecture
                 global_thresholds: bool,

                 # architecture
                 input_dim: int,
                 hidden_dim: int,
                 activation: str,
                 n_layers: int):
        super().__init__()

        assert n_classes > 2, f"n_classes: {n_classes}"

        # hyperparameters
        self.n_classes = n_classes
        self.global_thresholds = global_thresholds
        if self.global_thresholds:
            self.gamma = MLP(input_dim, 1, hidden_dim, activation, n_layers)
            self.thresholds = nn.Parameter(torch.linspace(-1, 1, n_classes - 1), requires_grad=True)
        else:
            self.gamma = MLP(input_dim, n_classes, hidden_dim, activation, n_layers)

    def forward(self, x) -> OrderedLogistic:
        # compute parameters
        parameters = self.gamma(x)

        # extract loc
        loc = parameters[:, 0]

        # extract thresholds
        if self.global_thresholds:
            thresholds = self.thresholds[None, :].expand(x.shape[0], -1)
        else:
            thresholds = parameters[:, 1:]

        # OOORDER
        thresholds = torch.cumsum(F.softplus(thresholds), dim=1)

        assert loc.shape == (x.shape[0], ), loc.shape
        assert thresholds.shape == (x.shape[0], self.n_classes - 1), thresholds.shape
        return OrderedLogistic(loc=loc, thresholds=thresholds)


class Decoder_X_count(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 activation: str,
                 n_layers: int):
        super().__init__()
        self.gamma = MLP(input_dim, 1, hidden_dim, activation, n_layers)

    def forward(self, x: torch.Tensor) -> D.Poisson:
        rate = torch.exp(self.gamma(x)).squeeze()
        assert rate.shape == (x.shape[0], ), rate.shape
        return D.Poisson(rate=rate)
