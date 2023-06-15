import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from pipeline.common.registry import registry

from pipeline.models.gnn import GNN
import contextlib
from pipeline.models.base_model import BaseModel


@registry.register_model("linear_probe")
class LinearProbe(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {"linear_probe": "configs/models/linear_probe.yaml"}

    def __init__(
            self,
            in_dim=300,
            num_bi_heads=13,  # #binary heads
            num_num_heads=12,  # #numerical heads
        ):
        super().__init__()
        self.num_bi_heads = num_bi_heads
        self.num_num_heads = num_num_heads
        self.num_heads = num_bi_heads + num_num_heads
        self.fc = nn.Linear(in_dim, self.num_heads)
        self.criterion_bi = nn.BCEWithLogitsLoss(reduction="none")
        self.criterion_num = nn.MSELoss(reduction="none")

    def forward(self, sample):
        feat = sample["feat"]
        logits = self.fc(feat)

        loss_weight = sample["loss_weight"]
        labels = sample["labels"]

        logits_bi = logits[:, :self.num_bi_heads].flatten()
        weight_bi = loss_weight[:, :self.num_bi_heads].flatten()
        labels_bi = labels[:, :self.num_bi_heads].flatten()
        loss_bi0 = self.criterion_bi(logits_bi, labels_bi)
        loss_bi = (loss_bi0 * weight_bi).sum() / weight_bi.sum()
        
        logits_num = logits[:, self.num_bi_heads:].flatten()
        weight_num = loss_weight[:, self.num_bi_heads:].flatten()
        labels_num = labels[:, self.num_bi_heads:].flatten()
        loss_num0 = self.criterion_num(logits_num, labels_num)
        loss_num = (loss_num0 * weight_num).mean()

        loss = loss_bi + loss_num * 0# 1e-4
        return {"loss": loss, "loss_bi": loss_bi, "loss_num": loss_num,  "logits_bi": logits_bi.clone().detach().cpu(), "labels_bi": labels_bi.clone().detach().cpu()}


    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def from_config(cls, cfg):

        num_bi_heads = cfg.get("num_bi_heads", 13)
        num_num_heads = cfg.get("num_num_heads", 12)
        in_dim = cfg.get("in_dim", 300)

        model = cls(
            in_dim,
            num_bi_heads,
            num_num_heads,
        )

        return model
