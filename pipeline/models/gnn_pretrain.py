import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from pipeline.common.registry import registry

from pipeline.models.gnn import GNN
import contextlib
from pipeline.models.base_model import BaseModel


@registry.register_model("gnn_pretrain")
class GNN_ENCODER(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {"gnn0": "configs/models/gnn.yaml"}

    def __init__(
            self,
            vocab_size,
            use_graph_agg=True,
            gnn_ckpt=None,
        ):
        super().__init__()
        self.use_graph_agg = use_graph_agg
        self.create_gnn(gnn_ckpt)
        self.fc = nn.Linear(self.encoder_out_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss()

    def create_gnn(self, model_path):
        print('Loading GNN')
        print(f"use_graph_agg={self.use_graph_agg}")
        self.gnn = GNN(num_layer=5, emb_dim=300, gnn_type='gcn', use_graph_agg=self.use_graph_agg)
        if model_path is not None:
            self.gnn.load_from_pretrained(url_or_filename=model_path)
        self.encoder_out_dim = self.gnn.out_dim
        
        print('Loaded GNN')

    def forward(self, sample):
        graph = sample["graph"]
        word_vec = sample["word_vec"]

        emb = self.gnn(graph)
        emb = emb.squeeze(1)
        logits = self.fc(emb)

        loss = self.criterion(logits, word_vec)

        return {"loss": loss}


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

        vocab_size = cfg.get("vocab_size", None)
        use_graph_agg = cfg.get("use_graph_agg", True)
        gnn_ckpt = cfg.get("gnn_ckpt", None)

        model = cls(
            vocab_size,
            use_graph_agg=use_graph_agg,
            gnn_ckpt=gnn_ckpt,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            print("Loaded checkpoint from {}: {}".format(ckpt_path, msg))

        return model
