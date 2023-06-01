
import pickle
import torch
import numpy as np

from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, ConcatDataset
from torch_geometric.data import Batch


class GraphWordVec(Dataset):
    def __init__(self, datapath) -> None:
        super().__init__()
        with open(datapath, "rb") as f:
            data = pickle.load(f)
        for x in data:
            tmp = x['word_vec'] / x['word_vec'].sum()  # normalization, np.float64
            x['word_vec'] = tmp.astype(np.float32)  # shape [1, N] sparse matrix
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        rec = self.data[index]
        return rec
    
    @staticmethod
    def collater(samples):
        g = Batch.from_data_list([x["graph"] for x in samples])
        words = default_collate([torch.asarray(x["word_vec"].todense()).flatten() for x in samples])
        out = {"graph": g, "word_vec": words}
        return out
