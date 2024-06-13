from lightning.pytorch import LightningDataModule


class BaseDataModule(LightningDataModule):
    _registry = {}
    def __init_subclass__(cls) -> None:
        if cls not in BaseDataModule._registry.keys():
            BaseDataModule._registry[cls.__name__] = cls
        return super().__init_subclass__()

    @classmethod
    @property
    def registered_datamodules(cls):
        assert cls == BaseDataModule, \
            'get_datamodules should be called from BaseDataModule'
        return BaseDataModule._registry

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        return self.__class__.__name__
