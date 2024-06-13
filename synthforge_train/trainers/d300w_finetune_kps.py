from ..src.lightning.modules import LandmarksFinetuneLightningModule
from ..src.lightning.datamodules import BaseDataModule
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI


class Trainer(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "data", "model.datamodule", apply_on="instantiate"
        )
        return super().add_arguments_to_parser(parser)


def cli_main():
    cli = Trainer(
        LandmarksFinetuneLightningModule,
        BaseDataModule,
        subclass_mode_model=False,
        subclass_mode_data=True,
    )


if __name__ == "__main__":
    cli_main()
