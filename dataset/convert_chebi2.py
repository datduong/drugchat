import json
import pickle
import torch
from torch_geometric.data import Data, Batch


def main():

    with open("../data/chebi_0.pkl", "rb") as f:
        data = pickle.load(f)
    for rec in data:
        g = rec["graph"]
        graph = Data(x=torch.asarray(g['node_feat']), edge_index=torch.asarray(g['edge_index']), edge_attr=torch.asarray(g['edge_feat']))
        rec["graph"] = graph

    with open("../data/chebi_graph.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
