import os
import logging
import warnings

from pipeline.common.registry import registry
from pipeline.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from pipeline.datasets.datasets.imageMol_dataset import ImageMolDataset


@registry.register_builder("image_mol")
class ImageMolBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageMolDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/image_mol/default.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        # self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            datapath=build_info.storage,
        )

        return datasets
