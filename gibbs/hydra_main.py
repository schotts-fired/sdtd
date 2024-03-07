import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
import logging
from sdtd import data
from sdtd.data import datasets
from sdtd.gibbs.discovery import Discovery
import numpy as np
import pandas as pd
import os


def initialize_alpha(dataset: datasets.SDTDDataset, cfg: DictConfig) -> np.ndarray:
    alpha = np.full((len(dataset.domains), 3), cfg.alpha)
    for m, domain in dataset.domains.items():
        if domain.is_real():
            alpha[m, -1] = 0
        if domain.is_binary():
            alpha[m, :] = 0  # no type discovery here
    return alpha


def load_model(dataset: datasets.SDTDDataset,
               prior_alpha: np.ndarray,
               cfg: DictConfig) -> Discovery:
    # model expects domains to be indexed by integers not strings
    model = Discovery(
        X=dataset.to_numpy_dataset(),
        domains=dataset.domains,
        prior_alpha=prior_alpha,
        s2_b=cfg.s2B,
        s2_y=cfg.s2Y,
        s2_z=cfg.s2Z,
        s2_theta=cfg.s2theta,
        s2_u=cfg.s2u,
        K=cfg.K,
        n_iterations=cfg.n_iterations,
        seed=cfg.seed,
        dataset=cfg.dataset.name,
        logging=cfg.logging,
    )
    return model


def run(cfg: DictConfig, log: logging.Logger) -> None:
    results = []
    for simulation in range(cfg.n_simulations):
        log.info(">>> Starting simulation %d...", simulation)
        log.info(">>> Creating/Loading dataset...")
        dataset = data.load_dataset(**cfg.dataset, label_encoding_starts_at_one=True)

        log.info(">>> Creating model...")
        prior_alpha = initialize_alpha(dataset, cfg)
        model = load_model(dataset, prior_alpha, cfg)

        log.info(">>> Fitting model...")
        model = model.fit()

        for m in range(dataset.n_features):
            simulation_results = {}
            simulation_results["simulation"] = simulation
            simulation_results["variable"] = dataset.columns[m]
            simulation_results["dtype"] = dataset.domains[m]
            simulation_results['W1'] = model.W[m, 0]
            simulation_results['W2'] = model.W[m, 1]
            simulation_results['W3'] = model.W[m, 2]
            results.append(simulation_results)
    results = pd.DataFrame(results)

    if cfg.output_path:
        log.info(">>> writing results to %s...", cfg.output_path)
        results.to_csv(cfg.output_path, index=False)
    log.info(">>> Done.")


def run_reference(cfg: DictConfig, log) -> None:
    from .datatypes_reference import discovery as reference
    results = []
    for simulation in range(cfg.n_simulations):
        log.info(">>> Starting simulation %s...", simulation)

        # create dataset
        log.info(">>> Creating dataset...")
        dataset = data.load_dataset(**cfg.dataset,
                                    label_encoding_starts_at_one=True,
                                    na_value=-1.0)

        # prepare inputs
        X = dataset.to_numpy_dataset()
        assert all(np.nanmin(X[:, m]) >= 1.0 for m in range(dataset.n_features) if dataset.domains[m].is_discrete() or dataset.domains[m].is_binary())
        R = np.nanmax(X, axis=0)
        C = np.empty((dataset.n_features, ))
        for m in range(dataset.n_features):
            if dataset.domains[m].is_positive_real():
                C[m] = 1
            elif dataset.domains[m].is_real():
                C[m] = 2
            elif dataset.domains[m].is_binary():
                C[m] = 3
            elif dataset.domains[m].is_discrete():
                C[m] = 4
        prior_alpha = initialize_alpha(dataset, cfg)
        if cfg.logging and not os.path.exists("traces"):
            os.makedirs("traces")

        # run reference
        log.info(f">>> Fitting model for {cfg['n_iterations']} iterations...")
        weights = reference(
            X=X.astype(float), XT=X.astype(float),
            N=X.shape[0], D=X.shape[1],
            R=R.astype(float),
            C=C.astype(float),
            W=prior_alpha.astype(float),
            s2B=cfg["s2B"],
            s2Y=cfg["s2Y"],
            s2Z=cfg["s2Z"],
            s2theta=cfg["s2theta"],
            s2u=cfg["s2u"],
            Kest=cfg["K"],
            Nsim=cfg["n_iterations"],
            gsl_seed=42,
            # variable_dir=f"../dataset={cfg.dataset.name},logging=true,n_iterations=1,n_simulations=1,reference=False/variables/"
            # variable_dir=f"../dataset={cfg.dataset.name},logging=true,n_simulations=1,reference=False/variables/"
            variable_dir=""
        )

        # save results
        for m in range(dataset.n_features):
            simulation_results = {}
            simulation_results["simulation"] = simulation
            simulation_results["variable"] = dataset.columns[m]
            simulation_results["dtype"] = dataset.domains[m]
            simulation_results['W1'] = weights[m, 0]
            simulation_results['W2'] = weights[m, 1]
            simulation_results['W3'] = weights[m, 2]
            results.append(simulation_results)
    results = pd.DataFrame(results)

    if cfg.output_path != "":
        log.info(">>> writing results to %s...", cfg.output_path)
        results.to_csv(cfg.output_path, index=False)
    log.info(">>> Done.")


@hydra.main(version_base=None, config_path="configs", config_name="defaults")
def main(cfg: DictConfig):
    log = logging.getLogger(__name__)
    if cfg.reference:
        run_reference(cfg, log)
    else:
        run(cfg, log)


if __name__ == "__main__":
    main()
