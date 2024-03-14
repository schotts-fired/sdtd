import torch
import lightning as L
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA

from sdtd.data.util import Domain


# plot settings
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=1.5)
sns.set_palette(sns.color_palette())
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"


def thermometer(x: torch.Tensor, n_classes: torch.float32):
    """Input is assumed to be label encoded starting at 0. For three classes,
    the first class is encoded as 000 and the last as 110. Technically, we could make the
    output two dimensional, but this clashes with the categorical one-hot encoding."""
    if x.max() >= n_classes:
        print(f"Warning: x.max()={x.max()} >= n_classes={n_classes}")
    assert x.max() < n_classes, f'{x.max()} >= {n_classes}'
    assert x.min() >= 0, f'{x.min()} < 0'
    assert x.dtype == torch.long
    assert x.ndim == 1
    batch_size = len(x)
    labels = torch.arange(n_classes, device=x.device).view(1, -1).expand(batch_size, -1)
    batch_values = x.view(-1, 1)
    encoding = (labels < batch_values).type(torch.float32)
    assert (encoding[:, -1] == 0).all()
    return encoding


def _create_marginals_fig(df: pd.DataFrame | pd.Series, domain: Domain):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    fig, ax = plt.subplots()
    if domain.is_continuous():
        g = sns.kdeplot(data=df, x='$x^d$', fill=True, ax=ax)
    if domain.is_discrete():
        g = sns.histplot(data=df, x='$x^d$', stat='probability', discrete=True, ax=ax)
        if len(df['$x^d$'].unique()) > 25:
            plt.xticks(rotation=45)
    g.set_ylabel('$p(x^d)$')
    plt.tight_layout()
    return fig


class PlotMarginals(L.Callback):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def on_train_start(self, trainer, pl_module):
        x_df = trainer.datamodule.dataset.to_df('train')
        for m in range(pl_module.n_features):
            fig = _create_marginals_fig(x_df.iloc[:, m].rename('$x^d$'), pl_module.domains[m])
            pl_module.logger.experiment.log({f'Dataset/X{m}': wandb.Image(fig)})
            plt.close()


class PlotBatchMarginals(L.Callback):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if pl_module.current_epoch == 0 and batch_idx == 0:
            N = batch[0].shape
            M = len(batch)
            for m in range(pl_module.n_features):
                x_df = pd.DataFrame(batch[m].cpu().numpy(), columns=[f'$x^d$'])
                fig = _create_marginals_fig(x_df, pl_module.domains[m])
                trainer.logger.experiment.log({f'Input Batch/X{m}': wandb.Image(fig)})
                plt.close()


class PlotGeneratedMarginals(L.Callback):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def on_train_end(self, trainer, pl_module):
        # set in eval mode for generation
        pl_module.eval()

        # generate
        N = pl_module.example_input_array[0].shape[0]
        M = pl_module.n_features
        x = pl_module.generate(N)
        for m in range(M):
            x_df = pd.DataFrame(x[m].cpu().numpy(), columns=[f'$x^d$'])
            fig = _create_marginals_fig(x_df, pl_module.domains[m])
            trainer.logger.experiment.log({f'Generative Samples/X{m}': wandb.Image(fig)})
            plt.close()

        # set back to training mode
        pl_module.train()


class PlotLatentSpace(L.Callback):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def on_train_end(self, trainer, pl_module):
        pl_module.eval()
        output = pl_module(pl_module.example_input_array)
        z_sample = output['q_z'].sample().cpu()
        z_df = pd.DataFrame(z_sample.numpy(), columns=[f'Z{k}' for k in range(z_sample.shape[1])])
        if z_sample.shape[1] >= 2:
            z_df = pd.DataFrame(PCA(n_components=2).fit_transform(z_df), columns=["$z_1$", "$z_2$"])
            for m in range(pl_module.n_features):
                if pl_module.domains[m].is_discrete():
                    z_df['Class'] = pl_module.example_input_array[m].cpu()
                    plt.figure()
                    sns.scatterplot(
                        x='$z_1$',
                        y='$z_2$',
                        hue='Class',
                        data=z_df
                    )
                    trainer.logger.experiment.log({f'Sample Z/Colored X{m}': wandb.Image(plt)})
                    plt.close()
        pl_module.train()


class PlotReconstructedMarginals(L.Callback):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def on_train_end(self, trainer, pl_module):
        pl_module.eval()
        output = pl_module(pl_module.example_input_array)
        for m in range(pl_module.n_features):
            if pl_module.domains[m].is_real() or pl_module.domains[m].is_binary():
                x_df = pd.DataFrame(output['p_x'][m].sample().cpu().numpy(), columns=[f'$x^d$'])
                fig = _create_marginals_fig(x_df, pl_module.domains[m])
                trainer.logger.experiment.log({f'Reconstructed Batch/X{m}': wandb.Image(fig)})
            else:
                mixture_sample_m = output['p_x'][m].sample()
                component_sample_m = output['p_x'][m].sample_components()

                # full mixture reconstruction
                x_df = pd.DataFrame(mixture_sample_m.cpu().numpy(), columns=[f'$x^d$'])
                fig = _create_marginals_fig(x_df, pl_module.domains[m])
                trainer.logger.experiment.log({f'Reconstructed Batch/X{m}': wandb.Image(fig)})

                # component reconstructions
                if pl_module.domains[m].is_positive_real():
                    x_df = pd.DataFrame(component_sample_m[:, 0].cpu().numpy(), columns=[f'$x^d$'])
                    fig = _create_marginals_fig(x_df, pl_module.domains[m])
                    trainer.logger.experiment.log({f'Reconstructed Real Samples/X{m}': wandb.Image(fig)})
                    x_df = pd.DataFrame(component_sample_m[:, 1].cpu().numpy(), columns=[f'$x^d$'])
                    fig = _create_marginals_fig(x_df, pl_module.domains[m])
                    trainer.logger.experiment.log({f'Reconstructed Positive Samples/X{m}': wandb.Image(fig)})
                elif pl_module.domains[m].is_discrete():
                    x_df = pd.DataFrame(component_sample_m[:, 0].cpu().numpy(), columns=[f'$x^d$'])
                    fig = _create_marginals_fig(x_df, pl_module.domains[m])
                    trainer.logger.experiment.log({f'Reconstructed Categorical Samples/X{m}': wandb.Image(fig)})
                    x_df = pd.DataFrame(component_sample_m[:, 1].cpu().numpy(), columns=[f'$x^d$'])
                    fig = _create_marginals_fig(x_df, pl_module.domains[m])
                    trainer.logger.experiment.log({f'Reconstructed Ordinal Samples/X{m}': wandb.Image(fig)})
                    x_df = pd.DataFrame(component_sample_m[:, 2].cpu().numpy(), columns=[f'$x^d$'])
                    fig = _create_marginals_fig(x_df, pl_module.domains[m])
                    trainer.logger.experiment.log({f'Reconstructed Count Samples/X{m}': wandb.Image(fig)})

            plt.close()
        pl_module.train()


class PlotInferedWeights(L.Callback):

    def __init__(self, log_plot: bool = False, log_csv: bool = False):
        assert log_plot or log_csv, f"With log_plot={log_plot} and log_csv={log_csv} both False, this callback is useless."
        super().__init__()
        self.log_plot = log_plot
        self.log_csv = log_csv

    @torch.no_grad()
    def on_train_end(self, trainer, pl_module):
        pl_module.eval()
        # collect infered weights for all batches
        infered_weights = [[] for m in range(pl_module.n_features)]
        for batch in trainer.train_dataloader:
            batch = [batch[m].to(pl_module.device) for m in range(pl_module.n_features)]
            output = pl_module(batch)
            for m in range(pl_module.n_features):
                if not pl_module.domains[m].is_real() and not pl_module.domains[m].is_binary():
                    infered_weights[m].append(output['q_w'][m].sample().cpu())

        for m in range(pl_module.n_features):
            if not pl_module.domains[m].is_real() and not pl_module.domains[m].is_binary():
                infered_weights_m = torch.cat(infered_weights[m])
                if pl_module.domains[m].is_positive_real():
                    columns = ['Real', 'Positive']
                elif pl_module.domains[m].is_discrete():
                    columns = ['Categorical', 'Ordinal', 'Count']
                df = pd.DataFrame(infered_weights_m.numpy(), columns=columns)
                df = df.melt(var_name='Data Type', value_name=r'$\mathbf{w}^d$')
                df['Variable'] = f'X{m}'

                if self.log_plot:
                    plt.figure()
                    g = sns.boxplot(x='Data Type', y=r'$\mathbf{w}^d$', data=df, hue='Data Type', legend=True)
                    g.set_ylim(0, 1)
                    g.set_xticklabels('')
                    trainer.logger.experiment.log({f'Samples W/X{m}': wandb.Image(plt)})
                    plt.close()

                if self.log_csv:
                    weights_table = wandb.Table(dataframe=df)
                    trainer.logger.experiment.log({f'Table W/X{m}': weights_table})
        pl_module.train()


class TrackLikelihoodParams(L.Callback):

    def __init__(self):
        super().__init__()
        self.mle_mean_real = {}
        self.mle_std_real = {}
        self.mle_mean_pos = {}
        self.mle_std_pos = {}
        self.mle_probs_cat = {}
        self.mle_probs_ord = {}
        self.mle_mean_count = {}
        self.mle_std_count = {}

    @torch.no_grad()
    def on_train_start(self, trainer, pl_module):
        dataset = torch.from_numpy(trainer.datamodule.dataset.to_numpy_dataset('train'))
        for m in range(pl_module.n_features):
            if pl_module.domains[m].is_continuous():
                self.mle_mean_real[m] = dataset[:, m].mean(axis=0)
                self.mle_std_real[m] = dataset[:, m].std(axis=0)
            if pl_module.domains[m].is_positive_real():
                self.mle_mean_pos[m] = dataset[:, m].log1p().mean(axis=0)
                self.mle_std_pos[m] = dataset[:, m].log1p().std(axis=0)
            if pl_module.domains[m].is_binary():
                self.mle_probs_bin[m] = torch.bincount(dataset[:, m].int()) / len(dataset)
            if pl_module.domains[m].is_discrete():
                self.mle_probs_cat[m] = torch.bincount(dataset[:, m].int()) / len(dataset)
                self.mle_probs_ord[m] = torch.bincount(dataset[:, m].int()) / len(dataset)
                self.mle_mean_count[m] = dataset[:, m].mean(axis=0)

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        for m in range(pl_module.n_features):
            if pl_module.domains[m].is_real():
                pl_module.log(f'Parameters/Real_{m}(Mean)', outputs['p_x'][m].mean.detach().mean(axis=0), on_step=False, on_epoch=True)
                pl_module.log(f'Parameters/Real_{m}(Std)', outputs['p_x'][m].stddev.detach().mean(axis=0), on_step=False, on_epoch=True)
                pl_module.log(f'MLE/Real_{m}(Mean)', self.mle_mean_real[m], on_step=False, on_epoch=True)
                pl_module.log(f'MLE/Real_{m}(Std)', self.mle_std_real[m], on_step=False, on_epoch=True)
            elif pl_module.domains[m].is_binary():
                for k in range(2):
                    pl_module.log(f"Parameters/Binary_{m}(P(1))", outputs['p_x'][m].probs[:, k].detach().mean(axis=0), on_step=False, on_epoch=True)
                    pl_module.log(f"MLE/Binary_{m}(P(1))", self.mle_probs_bin[m][k], on_step=False, on_epoch=True)
            else:
                components = outputs['p_x'][m].component_distributions
                if pl_module.domains[m].is_positive_real():
                    pl_module.log(f'Parameters/Real_{m}(Mean)', components[0].mean.detach().mean(axis=0), on_step=False, on_epoch=True)
                    pl_module.log(f'Parameters/Real_{m}(Std)', components[0].stddev.detach().mean(axis=0), on_step=False, on_epoch=True)
                    pl_module.log(f'MLE/Real_{m}(Mean)', self.mle_mean_real[m], on_step=False, on_epoch=True)
                    pl_module.log(f'MLE/Real_{m}(Std)', self.mle_std_real[m], on_step=False, on_epoch=True)
                    pl_module.log(f'Parameters/Positive_{m}(Mean)', components[1].loc.detach().mean(axis=0), on_step=False, on_epoch=True)
                    pl_module.log(f'Parameters/Positive_{m}(Std)', components[1].scale.detach().mean(axis=0), on_step=False, on_epoch=True)
                    pl_module.log(f'MLE/Positive_{m}(Mean)', self.mle_mean_pos[m], on_step=False, on_epoch=True)
                    pl_module.log(f'MLE/Positive_{m}(Std)', self.mle_std_pos[m], on_step=False, on_epoch=True)
                if pl_module.domains[m].is_discrete():
                    for k in range(pl_module.n_classes[m]):
                        pl_module.log(f"Parameters/Categorical_{m}(P({k}))", components[0].probs[:, k].detach().mean(), on_step=False, on_epoch=True)
                        pl_module.log(f"Parameters/Ordinal_{m}(P({k}))", components[1].probs[:, k].detach().mean(), on_step=False, on_epoch=True)
                        pl_module.log(f"MLE/Categorical_{m}(P({k}))", self.mle_probs_cat[m][k], on_step=False, on_epoch=True)
                        pl_module.log(f"MLE/Ordinal_{m}(P({k}))", self.mle_probs_ord[m][k], on_step=False, on_epoch=True)
                    pl_module.log(f'Parameters/Count_{m}(Mean)', components[2].mean.detach().mean(axis=0), on_step=False, on_epoch=True)
                    pl_module.log(f'MLE/Count_{m}(Mean)', self.mle_mean_count[m], on_step=False, on_epoch=True)
