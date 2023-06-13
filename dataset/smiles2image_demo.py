import pickle
import json
import os
from rdkit import Chem
from rdkit.Chem import Draw
import time
import datetime


def Smiles2Img(smis, size=224, savePath=None):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
        path: E:/a/b/c.png
    '''
    # try:
    mol = Chem.MolFromSmiles(smis)
    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size, size))
    if savePath is not None:
        img.save(savePath)
    return img
    # except:
    #     return None


def convert_chembl():
    """
    Once started, constantly checks a text file. 
    If it finds new content, convert it to a graph and save it.
    """
    old_t0 = 0
    txt = "dataset/tmp_smiles.txt"
    save_path = "dataset/smiles_image.png"
    while True:
        time.sleep(1)
        if not os.path.isfile(txt):
            continue
        with open(txt, "rt") as f:
            res = f.read().strip("\n ")
        if not res:
            continue
        tmp = res.split(" ")
        t0 = float(tmp[0])
        if t0 <= old_t0:
            continue
        smi = " ".join(tmp[1:]).strip("\n ")
        tt = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        print(f"At time {tt}: {repr(res)}")
        old_t0 = t0
        g = Smiles2Img(smi, savePath=save_path)
        out = {"timestamp": time.time(), "img_save_path": save_path}
        with open("dataset/tmp_smiles.pkl", "wb") as f:
            pickle.dump(out, f)


if __name__ == '__main__':
    convert_chembl()
