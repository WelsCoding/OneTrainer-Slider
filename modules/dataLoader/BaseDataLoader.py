import copy
from abc import ABCMeta, abstractmethod

from modules.dataLoader.mixin.DataLoaderMgdsMixin import DataLoaderMgdsMixin
from modules.model.BaseModel import BaseModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.util.config.TrainConfig import TrainConfig
from modules.util.TrainProgress import TrainProgress

from mgds.MGDS import MGDS, TrainDataLoader

import torch
from torch.utils.data import DataLoader, default_collate


def _slider_safe_collate(batch: list[dict]) -> dict:
    if not batch:
        return {}
    all_keys = list(dict.fromkeys(key for sample in batch for key in sample))
    none_keys = {key for key in all_keys if any(sample.get(key) is None for sample in batch)}
    if not none_keys:
        return default_collate(batch)

    collated = default_collate([
        {key: value for key, value in sample.items() if key not in none_keys}
        for sample in batch
    ])
    for key in none_keys:
        collated[key] = [sample.get(key) for sample in batch]
    return collated


class SliderSafeTrainDataLoader(TrainDataLoader):
    def __init__(self, dataset: MGDS, batch_size: int):
        DataLoader.__init__(self, dataset, batch_size=batch_size, drop_last=True, collate_fn=_slider_safe_collate)


class BaseDataLoader(
    DataLoaderMgdsMixin,
    metaclass=ABCMeta,
):

    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: BaseModel,
            model_setup: BaseModelSetup,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        super().__init__()

        self.train_device = train_device
        self.temp_device = temp_device

        if is_validation:
            config = copy.copy(config)
            config.batch_size = 1
            config.multi_gpu = False

        self.__ds = self._create_dataset(
            config=config,
            model=model,
            model_setup=model_setup,
            train_progress=train_progress,
            is_validation=is_validation,
        )
        self.__dl = SliderSafeTrainDataLoader(self.__ds, config.batch_size)

    def get_data_set(self) -> MGDS:
        return self.__ds

    def get_data_loader(self) -> TrainDataLoader:
        return self.__dl

    @abstractmethod
    def _create_dataset(
            self,
            config: TrainConfig,
            model: BaseModel,
            model_setup: BaseModelSetup,
            train_progress: TrainProgress,
            is_validation,
    ):
        pass
