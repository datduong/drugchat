import os
import logging
import warnings

from pipeline.common.registry import registry
from pipeline.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from pipeline.datasets.datasets.feat_dataset import FeatDataset, GraphImageFeatDataset


@registry.register_builder("feat_dataset")
class FeatBuilder(BaseDatasetBuilder):
    train_dataset_cls = FeatDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/feat_default.yaml"}

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


@registry.register_builder("graph_img_feat_dataset")
class GraphImgFeatBuilder(BaseDatasetBuilder):
    train_dataset_cls = GraphImageFeatDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/graph_img_feat_default.yaml"}

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
