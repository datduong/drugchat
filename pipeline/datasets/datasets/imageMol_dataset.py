import os
import json

from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms


class ImageMolDataset(Dataset):
    def __init__(self, datapath) -> None:
        super().__init__()
        jsonpath = os.path.join(datapath, "smiles_img_qa.json")
        with open(jsonpath, "rt") as f:
            meta = json.load(f)
        totensor = transforms.ToTensor()
        self.images = {}
        self.data = []
        for idx, rec in meta.items():
            img_file = 'img_{}.png'.format(idx)
            image_path = os.path.join(datapath, img_file)
            image = Image.open(image_path).convert("RGB")
            img = totensor(image)
            self.images[idx] = img
            smi, qa = rec
            qa = [(idx, qa_pair) for qa_pair in qa]
            self.data.extend(qa)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        idx, qa_pair = self.data[index]
        img = self.images[idx]
        return {"img": img, "question": qa_pair[0], "text_input": str(qa_pair[1])}
    
    @staticmethod
    def collater(samples):
        imgs = default_collate([x["img"] for x in samples])
        qq = [x["question"] for x in samples]
        aa = [x["text_input"] for x in samples]
        out = {"image": imgs, "question": qq, "text_input": aa}
        return out
