import pickle
import json
import torch
from PIL import Image
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from torch_geometric.data import Data, Batch
import tqdm
from torchvision import transforms

from pipeline.models.gnn import GNN
from pipeline.models.image_mol import ImageMol

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model_path = "ckpt/gcn_contextpred.pth"
        self.use_graph_agg = True
        self.create_gnn(model_path)

    def create_gnn(self, model_path):
        print('Loading GNN')
        print(f"use_graph_agg={self.use_graph_agg}")
        self.gnn = GNN(num_layer=5, emb_dim=300, gnn_type='gcn', use_graph_agg=self.use_graph_agg)
        self.gnn.load_from_pretrained(url_or_filename=model_path)
        self.gnn.eval()

    def forward(self, graph):
        return self.gnn(graph)
    

class NetImg(nn.Module):
    def __init__(self, image_size=224) -> None:
        super().__init__()
        model_path = "ckpt/ImageMol.pth.tar"
        self.create_image_mol(model_path)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        self.transforms = transforms.Compose([
            transforms.CenterCrop(image_size), 
            transforms.ToTensor(),
            normalize,
        ])

    def create_image_mol(self, model_path):
        model = ImageMol()
        model.load_from_pretrained(url_or_filename=model_path)
        model.eval()
        self.image_mol = model

    def forward(self, img):
        return self.image_mol(img)

def convert_graph(infile, outfile):
    device = "cuda"
    net = Net().to(device)

    with open(infile, "rb") as f:
        data = pickle.load(f)
    with torch.no_grad():
        for smi, dd in tqdm.tqdm(data.items()):
            g = dd.pop("graph")
            graph0 = Data(x=torch.asarray(g['node_feat']), edge_index=torch.asarray(g['edge_index']), edge_attr=torch.asarray(g['edge_feat']))
            inputs = Batch.from_data_list([graph0]).to(device)
            feat = net(inputs)
            feat = feat.flatten().cpu()
            dd["graph_feat"] = feat

    with open(outfile, "wb") as f:
        pickle.dump(data, f)


def convert_graph_and_img():
    device = "cuda:1"
    net = Net().to(device)
    net_img = NetImg().to(device)

    with open("dataset/ChEMBL_QA_train_graph_smi.pkl", "rb") as f:
        data = pickle.load(f)

    with open("data/ChEMBL_QA_image/smiles_img_qa.json", "rt") as f:
        img_data = json.load(f)

    img_base = "data/ChEMBL_QA_image/img_{}.png"
    def get_img(smi):
        for idx, (smi_, _) in img_data.items():
            if smi == smi_:
                img_save_path = img_base.format(idx)
                img = Image.open(img_save_path).convert("RGB")
                inputs = net_img.transforms(img).unsqueeze(0).to(device)
                return inputs
        print(f"Warning: {smi} not found.")

    with torch.no_grad():
        for smi, dd in tqdm.tqdm(data.items()):
            g = dd.pop("graph")
            graph0 = Data(x=torch.asarray(g['node_feat']), edge_index=torch.asarray(g['edge_index']), edge_attr=torch.asarray(g['edge_feat']))
            inputs = Batch.from_data_list([graph0]).to(device)
            feat = net(inputs)
            feat = feat.flatten().cpu()
            dd["graph_feat"] = feat

            img = get_img(smi)
            if img is not None:
                img_feat = net_img(img)
                img_feat = img_feat.flatten().cpu()
            else:
                img_feat = None
            dd["img_feat"] = img_feat

    with open("dataset/chembl_train_graph_img_feat.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    # convert_graph("dataset/ChEMBL_PubChem_QA_train_graph_smi.pkl", "dataset/chembl_pubchem_train_graph_feat.pkl")
    # convert_graph("dataset/ChEMBL_QA_train_graph_smi.pkl", "dataset/chembl_train_graph_feat.pkl")
    convert_graph_and_img()
