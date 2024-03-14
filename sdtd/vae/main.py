from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from .lightning_modules import SDTDModule, BaselineModule, SDTDDataModule


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data", "model.init_args.domains", compute_fn=lambda dm: dm.dataset.domains, apply_on="instantiate")
        parser.link_arguments("data", "model.init_args.n_classes", compute_fn=lambda dm: dm.dataset.n_classes, apply_on="instantiate")


def cli_main():
    cli = MyLightningCLI(datamodule_class=SDTDDataModule,
                         parser_kwargs={"parser_mode": "omegaconf", "fit": {"default_config_files": ["sdtd/vae/configs/defaults.yaml"]}},
                         save_config_callback=None)


if __name__ == "__main__":
    cli_main()
