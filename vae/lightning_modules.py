import torch
import torch.nn.functional as F
import torch.distributions as D
from torch import nn, optim
import lightning as L
from typing import List, Dict, Any
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm
import wandb
import pandas as pd
import numpy as np
from torch.utils.data import random_split
from lightning.pytorch.utilities import grad_norm

from sdtd.data.util import Domain
from sdtd import data
from sdtd.vae.modules import (
    Encoder_W, Encoder_Z, Decoder_X_cont, Decoder_X_disc, Decoder_X_bin,
    Decoder_X_real, Decoder_X_cat, Decoder_Y

)
from sdtd.vae.util import thermometer
from sdtd.vae.distributions import Mixture


class SDTDModule(L.LightningModule):

    def __init__(self,
                 # dataset attributes
                 domains: Dict[int, Domain],
                 n_classes: Dict[int, int],

                 # hyperparameters
                 alpha: float = 1.0,

                 # z
                 output_dim_z: int = 10,
                 hidden_dim_z: int = None,
                 hidden_dim_z_same_as_output_dim_z: bool = True,
                 n_layers_z: int = 1,

                 # y
                 output_dim_y: int = None,
                 hidden_dim_y: int = None,
                 hidden_dim_y_same_as_output_dim_y: bool = True,
                 n_layers_y: int = None,
                 share_y: bool = True,

                 # x
                 hidden_dim_x: int = None,
                 n_layers_x: int = 1,

                 # architecture in general
                 activation: str = "relu",
                 global_thresholds: bool = False,

                 # optimization
                 lr: float = 1e-3,
                 beta_z: float = 1.0,
                 beta_w: float = 1.0,
                 beta_w_annealing_epochs: int = None,
                 beta_w_decaying_epochs: int = None,

                 # other
                 **kwargs):
        super().__init__()
        # input validation
        assert not (beta_w_annealing_epochs and beta_w_decaying_epochs), "Cannot anneal and decay beta_w at the same time"
        assert not (beta_w_annealing_epochs and beta_w_annealing_epochs < 0), "Beta_w_annealing_epochs must be positive"
        assert not (beta_w_decaying_epochs and beta_w_decaying_epochs < 0), "Beta_w_decaying_epochs must be positive"
        assert not (hidden_dim_z_same_as_output_dim_z and hidden_dim_z is not None), "hidden_dim_z_same_as_output_dim_z ignores hidden_dim_z, which was set to a value"
        assert not (hidden_dim_y_same_as_output_dim_y and hidden_dim_y is not None), "hidden_dim_y_same_as_output_dim_y ignores hidden_dim_y, which was set to a value"
        assert not (n_layers_y is None and (output_dim_y is not None or hidden_dim_y is not None)), "n_layers_y == 0 ignores output_dim_y, which was set to a value"

        # hyperparameters
        self.save_hyperparameters(ignore=['domains', 'n_classes'])
        self.n_features = len(domains)
        self.domains = domains
        self.n_classes = n_classes

        # batch norm
        normalized_dim = 0
        for m in range(self.n_features):
            if domains[m].is_positive_real():
                normalized_dim += 2
            elif domains[m].is_real():
                normalized_dim += 1
            elif domains[m].is_binary():
                normalized_dim += 2
            elif domains[m].is_discrete():
                normalized_dim += 2 * self.n_classes[m] + 1

        # W
        self.prior_w_alpha = nn.ParameterList()
        self.encoder_w = nn.ModuleList()
        for m in range(self.n_features):
            if domains[m].is_real() or domains[m].is_binary():
                # no weights here, but append None to keep the indices consistent, ModuleDicts require strings as keysa, so I consider this to be the better option
                self.prior_w_alpha.append(None)
                self.encoder_w.append(None)
            else:
                if domains[m].is_positive_real():
                    prior_alpha_m = [alpha] * 2
                elif domains[m].is_discrete():
                    prior_alpha_m = [alpha] * 3
                self.prior_w_alpha.append(nn.Parameter(torch.tensor(prior_alpha_m), requires_grad=False))
                self.encoder_w.append(Encoder_W(prior_alpha_m))

        # Z
        if hidden_dim_z_same_as_output_dim_z:
            hidden_dim_z = output_dim_z
        self.prior_loc = nn.Parameter(torch.zeros(output_dim_z), requires_grad=False)
        self.encoder_z = Encoder_Z(normalized_dim, output_dim_z, hidden_dim_z, activation, n_layers_z)

        # Y
        if hidden_dim_y_same_as_output_dim_y:
            hidden_dim_y = output_dim_y
        if n_layers_y is None:
            output_dim_y = output_dim_z
        self.decoder_y = Decoder_Y(self.n_features, share_y, output_dim_z, output_dim_y, hidden_dim_y, activation, n_layers_y)

        # X
        self.decoder_x = nn.ModuleList()
        for m in range(self.n_features):
            if self.domains[m].is_real():
                self.decoder_x.append(Decoder_X_real(output_dim_y, hidden_dim_x, activation, n_layers_x))
            elif self.domains[m].is_positive_real():
                self.decoder_x.append(Decoder_X_cont(output_dim_y, hidden_dim_x, activation, n_layers_x))
            elif self.domains[m].is_binary():
                self.decoder_x.append(Decoder_X_bin(output_dim_y, hidden_dim_x, activation, n_layers_x))
            elif self.domains[m].is_discrete():
                self.decoder_x.append(Decoder_X_disc(n_classes[m], global_thresholds, output_dim_y, hidden_dim_x, activation, n_layers_x))

    # PRIORS

    def prior_w(self, m: int) -> D.Dirichlet:
        assert not self.domains[m].is_binary() and not self.domains[m].is_real(), f"Feature X{m} with domain {self.domains[m]} has no weights"
        assert m >= 0 and m < self.n_features
        return D.Dirichlet(self.prior_w_alpha[m])

    def prior_z(self) -> D.Normal:
        return D.Normal(self.prior_loc, 1.0)

    # FORWARD PASS

    def normalize_variables(self, x: List[torch.Tensor]) -> torch.Tensor:
        x_normalized = []
        batch_loc_real = {}
        batch_scale_real = {}
        batch_loc_pos = {}
        batch_scale_pos = {}
        for m in range(self.n_features):
            if self.domains[m].is_continuous():
                batch_loc_real[m] = x[m].mean(axis=0)
                batch_scale_real[m] = x[m].std(axis=0)
                x_normalized.append((x[m] - batch_loc_real[m]) / batch_scale_real[m])
                if self.domains[m].is_positive_real():
                    assert (x[m] >= 0).all()
                    batch_loc_pos[m] = x[m].log1p().mean(axis=0)
                    batch_scale_pos[m] = x[m].log1p().std(axis=0)
                    x_normalized.append((x[m].log1p() - batch_loc_pos[m]) / batch_scale_pos[m])
            elif self.domains[m].is_binary():
                x_normalized.append(F.one_hot(x[m].long(), 2).float())
            elif self.domains[m].is_discrete():
                x_normalized.append(F.one_hot(x[m].long(), self.n_classes[m]).float())
                x_normalized.append(thermometer(x[m].long(), self.n_classes[m]).float())
                x_normalized.append(x[m].log1p())
        return torch.column_stack(x_normalized), batch_loc_real, batch_scale_real, batch_loc_pos, batch_scale_pos

    def forward(self, x):
        # normalize and encode variables
        x_normalized, batch_loc_real, batch_scale_real, batch_loc_pos, batch_scale_pos = self.normalize_variables(x)

        # sample z
        q_z = self.encoder_z(x_normalized)
        z = q_z.rsample()
        y = self.decoder_y(z)

        # create mixtures
        q_w = {}
        w = {}
        p_x = []
        for m in range(self.n_features):
            if self.domains[m].is_real():
                p_x.append(self.decoder_x[m](y[:, m], batch_loc_real[m], batch_scale_real[m]))
            elif self.domains[m].is_binary():
                p_x.append(self.decoder_x[m](y[:, m]))
            else:
                # mixture weights
                q_w[m] = self.encoder_w[m](x[m])  # x is just used for the shape here
                w[m] = q_w[m].rsample()

                # likelihood
                if self.domains[m].is_positive_real():
                    p_x_m = self.decoder_x[m](y[:, m], w[m],
                                              batch_loc_real[m],
                                              batch_scale_real[m],
                                              batch_loc_pos[m],
                                              batch_scale_pos[m])
                elif self.domains[m].is_discrete():
                    p_x_m = self.decoder_x[m](y[:, m], w[m])

                p_x.append(p_x_m)

        # result
        return {'p_x': p_x, 'q_w': q_w, 'q_z': q_z}

    def generate(self, n: int):
        z = self.prior_z().sample((n, ))
        y = self.decoder_y(z)
        x = []
        for m in range(self.n_features):
            if self.domains[m].is_real() or self.domains[m].is_binary():
                p_x_m = self.decoder_x[m](y[:, m])
            else:
                w_m = self.prior_w(m).sample((n, ))
                p_x_m = self.decoder_x[m](y[:, m], w_m)
            x.append(p_x_m.sample())
        return x

    # TRAINING

    def on_train_epoch_start(self):
        # anneal beta_w
        if self.hparams['beta_w_annealing_epochs']:
            self.hparams['beta_w'] = min(1.0, self.hparams['beta_w'] + 1.0 / self.hparams['beta_w_annealing_epochs'])

        # decay beta_w
        if self.hparams['beta_w_decaying_epochs']:
            self.hparams['beta_w'] = max(0.0, self.hparams['beta_w'] - 1.0 / self.hparams['beta_w_decaying_epochs'])

        # log to make sure the decaying and annealing actually works
        self.log('Hyperparameters/Beta_W', self.hparams['beta_w'])

    def on_train_batch_start(self, batch, batch_idx):
        if self.example_input_array is None:
            self.example_input_array = [batch[m].detach() for m in range(self.n_features)]

    def _step(self, batch, batch_idx):
        # forward pass
        output = self.forward(batch)

        # loss
        output["nll_loss"] = self.nll(batch, **output)
        output["kl_loss_w"], output["kl_loss_z"] = self.kl(**output)
        output["loss"] = output["nll_loss"] + self.hparams['beta_w'] * output["kl_loss_w"] + self.hparams['beta_z'] * output["kl_loss_z"]

        # interpretable metrics
        with torch.no_grad():
            for m in range(self.n_features):
                output[f"reconstruction_error_{m}"] = self.reconstruction_error(m=m, x=batch, **output)

        # all metrics stored in output dict, so callbacks have access to all outputs
        return output

    def training_step(self, batch, batch_idx):
        # step
        output = self._step(batch, batch_idx)

        # log losses
        self.log('Training Metrics/Total Loss', output["loss"])
        self.log('Training Metrics/NLL Loss', output["nll_loss"])
        self.log('Training Metrics/KL_W Loss', output["kl_loss_w"])
        self.log('Training Metrics/KL_Z Loss', output["kl_loss_z"])

        # interpretable metrics
        for m in range(self.n_features):
            self.log(f'Training Metrics/Reconstruction Error X({m})', output[f"reconstruction_error_{m}"])

        return output

    # VALIDATION

    def validation_step(self, batch, batch_idx):
        # forward pass
        output = self._step(batch, batch_idx)

        # losses
        self.log('Validation Metrics/NLL Loss', output["nll_loss"], on_epoch=True, on_step=False)

        # interpretable metrics
        for m in range(self.n_features):
            self.log(f'Validation Metrics/Reconstruction Error X({m})', output[f"reconstruction_error_{m}"], on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        # forward pass
        output = self._step(batch, batch_idx)

        # losses
        self.log('Test Metrics/NLL Loss', output["nll_loss"])

        # interpretable metrics
        self.log(f'Test Metrics/Reconstruction Error X({m})', output[f"reconstruction_error_{m}"])

    # METRICS

    def nll(self, x, p_x, **kwargs):
        log_p_x = torch.column_stack([p_x_m.log_prob(x[m]) for m, p_x_m in enumerate(p_x)])
        assert all([log_p_x[:, m].shape == x[m].shape for m in range(self.n_features)]), log_p_x.shape
        return -log_p_x.mean(dim=0).sum()

    def kl(self, q_w, q_z, **kwargs):
        kl_w_per_feature = []
        for m in range(self.n_features):
            if not self.domains[m].is_binary() and not self.domains[m].is_real():
                kl_w_per_feature.append(D.kl_divergence(q_w[m], self.prior_w(m)))
        kl_w = sum(kl_w_per_feature)
        kl_z = D.kl_divergence(q_z, self.prior_z())
        return kl_w.mean(), kl_z.mean()

    def reconstruction_error(self, x: torch.Tensor, p_x: D.Distribution, m: int, **kwargs):  # p_x, x would match the sklearn way of pred, true, but then I could not do batch, **output
        x_hat_m = p_x[m].sample()
        if self.domains[m].is_continuous():
            return F.mse_loss(x_hat_m, x[m])
        if self.domains[m].is_discrete() or self.domains[m].is_binary():
            return (x_hat_m != x[m]).float().mean()
        assert False, f"Domain {self.domains[m]} not supported"

    # OPTIMIZERS

    def on_before_optimizer_step(self, optimizer):
        for name, param in self.named_parameters():
            for m in range(self.n_features):
                if self.domains[m].is_positive_real() and param.grad is not None and 'gamma' in name:
                    self.log("Gradients/{}".format(name), param.grad.abs().mean(), on_step=True, on_epoch=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams['lr'])


class BaselineModule(L.LightningModule):

    def __init__(self,
                 n_classes: Dict[int, int],
                 domains: Dict[int, Domain],

                 # z
                 output_dim_z: int = 10,
                 hidden_dim_z: int = None,
                 n_layers_z: int = 1,

                 # x
                 hidden_dim_x: int = None,
                 n_layers_x: int = 1,

                 # architecture in general
                 activation: str = "relu",

                 # optimization
                 lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['n_classes', 'n_features'])
        self.domains = domains
        self.n_features = len(domains)
        self.n_classes = n_classes

        normalized_dim = 0
        for m in range(self.n_features):
            if self.domains[m].is_continuous():
                normalized_dim += 1
            elif self.domains[m].is_binary():
                normalized_dim += 2
            elif self.domains[m].is_discrete():
                normalized_dim += self.n_classes[m]

        # Z
        self.prior_loc = nn.Parameter(torch.zeros(output_dim_z), requires_grad=False)
        self.encoder_z = Encoder_Z(normalized_dim, output_dim_z, hidden_dim_z, activation, n_layers_z)

        # X
        self.decoder_x = nn.ModuleList()
        for m in range(self.n_features):
            if self.domains[m].is_continuous():
                self.decoder_x.append(Decoder_X_real(output_dim_z, hidden_dim_x, activation, n_layers_x))
            elif self.domains[m].is_binary():
                self.decoder_x.append(Decoder_X_bin(output_dim_z, hidden_dim_x, activation, n_layers_x))
            elif self.domains[m].is_discrete():
                self.decoder_x.append(Decoder_X_cat(self.n_classes[m], output_dim_z, hidden_dim_x, activation, n_layers_x))

    def prior_z(self) -> D.Normal:
        return D.Normal(self.prior_loc, 1.0)

    def normalize_variables(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = []
        mean = {}
        std = {}
        for m in range(self.n_features):
            if self.domains[m].is_continuous():
                mean[m] = x[m].mean(axis=0)
                std[m] = x[m].std(axis=0)
                x_normalized.append((x[m] - mean[m]) / std[m])
            if self.domains[m].is_binary():
                x_normalized.append(F.one_hot(x[m].long(), 2).float())
            if self.domains[m].is_discrete():
                x_normalized.append(F.one_hot(x[m].long(), self.n_classes[m]).float())
        return torch.column_stack(x_normalized), mean, std

    def forward(self, x: torch.Tensor):
        # normalize and encode variables
        x_normalized, mean, std = self.normalize_variables(x)

        # sample z
        q_z = self.encoder_z(x_normalized)
        z = q_z.rsample()

        # create likelihoods
        p_x = []
        for m in range(self.n_features):
            if self.domains[m].is_continuous():
                p_x_m = self.decoder_x[m](z, mean[m], std[m])
            elif self.domains[m].is_binary():
                p_x_m = self.decoder_x[m](z)
            elif self.domains[m].is_discrete():
                p_x_m = self.decoder_x[m](z)
            p_x.append(p_x_m)

        # result
        return {'p_x': p_x, 'q_z': q_z}

    def on_train_batch_start(self, batch, batch_idx):
        if self.example_input_array is None:
            self.example_input_array = [batch[m].detach() for m in range(self.n_features)]

    def _step(self, batch, batch_idx):
        # forward pass
        output = self.forward(batch)

        # loss
        output["loss"] = self.nll(batch, **output)
        output["kl_loss"] = self.kl(**output)

        # interpretable metrics
        with torch.no_grad():
            for m in range(self.n_features):
                output[f"reconstruction_error_{m}"] = self.reconstruction_error(m=m, x=batch, **output)

        # all metrics stored in output dict, so callbacks have access to all outputs
        return output

    def training_step(self, batch, batch_idx):
        # step
        output = self._step(batch, batch_idx)

        # log losses
        self.log('Training Metrics/NLL Loss', output["loss"])
        self.log('Training Metrics/KL_Z Loss', output["kl_loss"])

        # interpretable metrics
        for m in range(self.n_features):
            self.log(f'Training Metrics/Reconstruction Error X({m})', output[f"reconstruction_error_{m}"])

        return output

    def validation_step(self, batch, batch_idx):
        # forward pass
        output = self._step(batch, batch_idx)

        # losses
        self.log('Validation Metrics/NLL Loss', output["loss"])

        # interpretable metrics
        for m in range(self.n_features):
            self.log(f'Validation Metrics/Reconstruction Error X({m})', output[f"reconstruction_error_{m}"])

    def test_step(self, batch, batch_idx):
        # forward pass
        output = self._step(batch, batch_idx)

        # losses
        self.log('Test Metrics/NLL Loss', output["loss"])

        # interpretable metrics
        for m in range(self.n_features):
            self.log(f'Test Metrics/Reconstruction Error X({m})', output[f"reconstruction_error_{m}"])

    def nll(self, x, p_x, **kwargs):
        log_p_x = torch.column_stack([p_x_m.log_prob(x[m]) for m, p_x_m in enumerate(p_x)])
        assert all(log_p_x[:, m].shape == x[m].shape for m in range(self.n_features)), log_p_x.shape
        return -log_p_x.mean(dim=0).sum()

    def kl(self, q_z, **kwargs):
        kl_z = D.kl_divergence(q_z, self.prior_z())
        return kl_z.mean()

    def reconstruction_error(self, x: torch.Tensor, p_x: D.Distribution, m: int, **kwargs):  # p_x, x would match the sklearn way of pred, true, but then I could not do batch, **output
        x_hat_m = p_x[m].sample()
        if self.domains[m].is_continuous():
            return F.mse_loss(x_hat_m, x[m])
        if self.domains[m].is_discrete() or self.domains[m].is_binary():
            return (x_hat_m != x[m]).float().mean()
        assert False, f"Domain {self.domains[m]} not supported"

    # OPTIMIZERS

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams['lr'])


class SDTDDataModule(L.LightningDataModule):

    def __init__(self,
                 # dataset
                 name: str,
                 dataset_cfg: Dict[str, Any],

                 # hyperparameters
                 batch_size: int = 1000):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = data.load_dataset(name, **dataset_cfg)

    def train_dataloader(self):
        return DataLoader(self.dataset.to_tensor_dataset('train'),
                          batch_size=self.hparams['batch_size'],
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset.to_tensor_dataset('val'),
                          batch_size=self.hparams['batch_size'])

    def test_dataloader(self):
        return DataLoader(self.dataset.to_tensor_dataset('test'),
                          batch_size=self.hparams['batch_size'])
