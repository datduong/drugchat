import os
import json

from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms


class ImageMolDataset(Dataset):
    def __init__(self, datapath, image_size=224) -> None:
        super().__init__()
        jsonpath = os.path.join(datapath, "smiles_img_qa.json")
        print(f"Using {jsonpath=}")
        with open(jsonpath, "rt") as f:
            meta = json.load(f)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        self.transforms = transforms.Compose([
            transforms.CenterCrop(image_size), 
            transforms.ToTensor(),
            normalize,
        ])
        self.images = {}
        self.data = []
        for idx, rec in meta.items():
            img_file = 'img_{}.png'.format(idx)
            image_path = os.path.join(datapath, img_file)
            image = Image.open(image_path).convert("RGB")
            img = self.transforms(image)
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
