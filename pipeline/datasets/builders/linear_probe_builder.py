import os
import logging
import warnings

from pipeline.common.registry import registry
from pipeline.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from pipeline.datasets.datasets.linear_probe_dataset import LinearProbeFeatDataset


@registry.register_builder("linear_probe_dataset")
class LinearProbeBuilder(BaseDatasetBuilder):
    train_dataset_cls = LinearProbeFeatDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/linear_probe.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        # self.build_processors()

        build_info = self.config.build_info

        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets["train"] = dataset_cls(
            datapath=build_info.storage["train"],
        )
        datasets["valid"] = dataset_cls(
            datapath=build_info.storage["valid"],
        )

        return datasets

