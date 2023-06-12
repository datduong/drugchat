import pickle
import json

from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, ConcatDataset


class FeatDataset(Dataset):
    def __init__(self, datapath) -> None:
        super().__init__()
        with open(datapath, "rb") as f:
            data = pickle.load(f)

        self.feat = {}
        self.data = []
        for idx, (smi, dd) in enumerate(data.items()):
            self.feat[idx] = dd["graph_feat"]
            qa = [(idx, qa_pair) for qa_pair in dd["QA"]]
            self.data.extend(qa)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        idx, qa_pair = self.data[index]
        feat = self.feat[idx]
        if feat.ndim == 1:
            feat = feat.unsqueeze(0)  # add token dim
        return {"graph_feat": feat, "question": qa_pair[0], "text_input": str(qa_pair[1])}

    @staticmethod
    def collater(samples):
        graph_feat = default_collate([x["graph_feat"] for x in samples])
        qq = [x["question"] for x in samples]
        aa = [x["text_input"] for x in samples]
        out = {"graph_feat": graph_feat, "question": qq, "text_input": aa}
        return out



class GraphImageFeatDataset(Dataset):
    def __init__(self, datapath) -> None:
        super().__init__()
        with open(datapath, "rb") as f:
            data = pickle.load(f)

        self.feat = {}
        self.data = []
        for idx, (smi, dd) in enumerate(data.items()):
            self.feat[idx] = {"graph_feat": dd["graph_feat"], "img_feat": dd["img_feat"]}
            qa = [(idx, qa_pair) for qa_pair in dd["QA"]]
            self.data.extend(qa)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        idx, qa_pair = self.data[index]
        feat = self.feat[idx]
        if feat["graph_feat"].ndim == 1:
            feat["graph_feat"] = feat["graph_feat"].unsqueeze(0)  # add token dim
        if feat["img_feat"].ndim == 1:
            feat["img_feat"] = feat["img_feat"].unsqueeze(0)  # add token dim
        return {"graph_feat": feat["graph_feat"], "img_feat": feat["img_feat"], "question": qa_pair[0], "text_input": str(qa_pair[1])}

    @staticmethod
    def collater(samples):
        graph_feat = default_collate([x["graph_feat"] for x in samples])
        img_feat = default_collate([x["img_feat"] for x in samples])
        qq = [x["question"] for x in samples]
        aa = [x["text_input"] for x in samples]
        out = {"graph_feat": graph_feat, "image_feat": img_feat, "question": qq, "text_input": aa}
        return out
