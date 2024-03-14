# Discovering the Statistical Data Types of Variables with Probabilistic Matrix Factorization and Variational Autoencoders

This repo contains the code for my Bachelor thesis:

>On (Deep) Probabilistic Models for Data Type Discovery

It is not intended to be used as a library but may serve as inspiration for similar projects.

## Setup

To setup the required packages for both algorithms, run the following commands:

```bash
conda env create -f environment.yml
conda activate sdtd
```

## Probabilistic Matrix Factorization

### Overview 
Part of my Bachelor thesis involved reproducing the experiments from [this publication](https://proceedings.mlr.press/v70/valera17a.html):

> I. Valera and Z. Ghahramani, 
> "Automatic Discovery of the Statistical Types of Variables in a Dataset", 
> 34th International Conference on Machine Learning (ICML 2017). Sydney (Australia), 2017.

The code for this is located in the [sdtd.gibbs](gibbs) module and is based on the [original implementation](https://github.com/ivaleraM/DataTypes/tree/master)
released by the authors of the paper.
It implements a Gibbs sampling algorithm for this probabilistic graphical model:

<p align="center">
<img src="./images/pgm1.png" width="200">
</p>

### Experiments

Below are all the commands to reproduce the experiments for this part of my thesis.
The results will appear as csv files in a separate folder for each run. The folder is named
according to the hyperparameters defining this run. This looks as follows:

```
.
├── sdtd
└── experiments/
    └── gibbs/
        ├── real/
        │   ├── K=1,dataset.loc=0,dataset.scale=10,dataset=real/
        │   │   └── results.csv
        │   ├── K=1,dataset.loc=10,dataset.scale=10,dataset=real/
        │   │   └── results.csv
        │   └── ...
        ├── positive
        └── ...
```

#### Synthetic Data
##### Real-valued data

|Results|N(0,10)|N(10,10)|N(10,100)|
|-------|-------|--------|---------|
|<img src="./images/real.png" width="200">|<img src="./images/normal010.png" width="200">|<img src="./images/normal1010.png" width="200">|<img src="./images/normal10100.png" width="200">|


```bash
python -m sdtd.gibbs.hydra_main dataset=real K=1 dataset.loc=0 dataset.scale=10
python -m sdtd.gibbs.hydra_main dataset=real K=1 dataset.loc=10 dataset.scale=10
python -m sdtd.gibbs.hydra_main dataset=real K=1 dataset.loc=10 dataset.scale=100
```

##### Positive real-valued data

|Results|Gamma(1,1)|Gamma(3,1)|Gamma(5,1)|
|-------|-------|--------|---------|
|<img src="./images/positive.png" width="200">|<img src="./images/gamma11.png" width="200">|<img src="./images/gamma31.png" width="200">|<img src="./images/gamma51.png" width="200">|


```bash
python -m sdtd.gibbs.hydra_main dataset=positive K=1 dataset.a=1.0 dataset.scale=1.0
python -m sdtd.gibbs.hydra_main dataset=positive K=1 dataset.a=3.0 dataset.scale=1.0
python -m sdtd.gibbs.hydra_main dataset=positive K=1 dataset.a=5.0 dataset.scale=1.0
```

##### Interval-valued data

|Results|Beta(0.5,0.5)|Beta(0.5,1.0)|Beta(0.5,3.0)|
|-------|-------|--------|---------|
|<img src="./images/interval.png" width="200">|<img src="./images/beta0.50.5.png" width="200">|<img src="./images/beta0.51.png" width="200">|<img src="./images/beta0.53.png" width="200">|

```bash
python -m sdtd.gibbs.hydra_main dataset=interval K=1 dataset.a=0.5 dataset.b=0.5
python -m sdtd.gibbs.hydra_main dataset=interval K=1 dataset.a=0.5 dataset.b=1.0
python -m sdtd.gibbs.hydra_main dataset=interval K=1 dataset.a=0.5 dataset.b=3.0
```

##### Categorical data

<table>
    <thead>
        <tr>
            <th>Results</th>
            <th>Data</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="./images/categorical_classes.png" width="200"></td>
            <td rowspan=2><img src="./images/categorical_data.png" width="300"></td>
        </tr>
        <tr>
            <td><img src="./images/categorical_latent_features.png" width="200"></td>
        </tr>
    </tbody>
</table>


```bash
python -m sdtd.gibbs.hydra_main -m dataset=categorical K=1,2,3,4,5 dataset.n_classes=3,4,5,6,7,8,9
```

##### Ordinal data

<table>
    <thead>
        <tr>
            <th>Results</th>
            <th>Data</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="./images/ordinal_classes.png" width="200"></td>
            <td rowspan=2><img src="./images/ordinal_data.png" width="300"></td>
        </tr>
        <tr>
            <td><img src="./images/ordinal_latent_features.png" width="200"></td>
        </tr>
    </tbody>
</table>


```bash
python -m sdtd.gibbs.hydra_main -m dataset=ordinal K=1,2,3,4,5 dataset.n_classes=3,4,5,6,7,8,9
```

##### Count data

<table>
    <thead>
        <tr>
            <th>Results</th>
            <th>Data</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><img src="./images/count_a.png" width="200"></td>
            <td rowspan=2><img src="./images/count_data.png" width="300"></td>
        </tr>
        <tr>
            <td><img src="./images/count_latent_features.png" width="200"></td>
        </tr>
    </tbody>
</table>


```bash
python -m sdtd.gibbs.hydra_main -m dataset=count K=1,2,3,4,5 dataset.a=2,3,4,5,6,7,8
```

#### Real-world Data
For the real-world datasets, the number of simulations must explicitly be set to 1, since the default is 10. They can be run with following commands:

##### German Credit Dataset

```bash
python -m sdtd.gibbs.hydra_main dataset=german K=10 n_simulations=1
```
##### Adult Dataset

```bash
python -m sdtd.gibbs.hydra_main dataset=adult K=10 n_simulations=1
```

## Variational Autoencoder
This part of the project is located in the [sdtd.vae](vae) module. It implements the probabilistic graphical model

<p align="center">
<img src="./images/pgm2.png" width="200">
</p>

It takes advantage of [lightning](https://lightning.ai/docs/pytorch/stable/)
and [wandb](https://wandb.ai) to run the experiments and log the results in a structured way.
As a result, however, they need to be run in a two-step process. First, the sweep needs to be
initialized with a configuration file specifying the hyperparameters:

```bash
wandb sweep --project vae <path-to-config-file>
```

This will output a `sweep-id` that can then be used to start the sweep:

```bash
wandb agent <sweep-id>
```

This process can be used to run all the experiments specified in the
[sweeps directory](vae/configs/sweep). The outputs for each of the experiments
will be logged in the `vae` project and can be found in the `Sweeps`
tab. For details on how to setup `wandb`, please refer to the [setup
guide](https://docs.wandb.ai/guides/hosting/how-to-guides/basic-setup).
