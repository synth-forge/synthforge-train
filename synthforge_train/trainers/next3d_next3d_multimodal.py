from ..src.lightning.modules import MultiModalLightningModule
from ..src.lightning.datamodules import SynthForgeDataModule
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI


class Trainer(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "data", "model.datamodule", apply_on="instantiate"
        )
        return super().add_arguments_to_parser(parser)


def cli_main():
    cli = Trainer(
        MultiModalLightningModule,
        SynthForgeDataModule,
        subclass_mode_model=False,
        subclass_mode_data=False,
    )


if __name__ == "__main__":
    cli_main()
