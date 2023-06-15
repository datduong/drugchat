import pickle
import torch
from collections import defaultdict

from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, ConcatDataset
import logging

q_bi=['Is it known whether this drug is administered parenterally?',
 'Is it known whether this drug is applied topically?',
 'Is this compound a small molecule polymer, such as polystyrene sulfonate?',
 'Is this molecule characterized by a small molecular structure or a protein sequence?',
 'Does this compound satisfy the rule-of-three criteria?',
 'Determine if this molecule is inorganic, meaning it contains only metal atoms and fewer than two carbon atoms.',
 'Is there a black box warning associated with this drug?',
 'Is this drug used for therapeutic purposes, rather than for imaging, additives, or other non-therapeutic applications?',
 'Has this approved drug been withdrawn due to toxicity reasons for all indications, populations, and doses in at least one country (not necessarily the US)?',
 'Is it known if this drug is the first approved in its class, regardless of the indication or route of administration, acting on a specific target?',
 'Is it known whether this drug is taken orally?',
 'Is the drug administered in this specific form, such as a particular salt?',
 'Determine if this compound is a prodrug.',
 ]
q_clf=['What is the highest development stage achieved for this compound across all indications? Please respond with Approved, Phase 3 Clinical Trials, Phase 2 Clinical Trials, Phase 1 Clinical Trials, Early Phase 1 Clinical Trials, or Clinical Phase Unknown.',
 'Determine if this drug is administered as a racemic mixture, a single stereoisomer, an achiral molecule, or has an unknown chirality.',
 'Determine the type of availability for this drug.',
 'Is this compound an acid, a base, or neutral?',
 'What is the classification of this molecule? Please respond with Small Molecule, Protein, Antibody, Oligosaccharide, Oligonucleotide, Cell, Enzyme, Gene, or Unknown.',
 ]
q_num=['What is the polar surface area (PSA) value of this compound?',
 "How many violations of Lipinski's Rule of Five are there for this compound, using the HBA_LIPINSKI and HBD_LIPINSKI counts?",
 'What is the calculated ALogP value for this compound?',
 'How many heavy (non-hydrogen) atoms does this compound have?',
 'How many rotatable bonds does this compound have?',
 'How many aromatic rings does this compound have?',
 "How many hydrogen bond acceptors are there in this compound, calculated according to Lipinski's original rules (i.e., counting N and O atoms)?",
 "How many violations of Lipinski's Rule of Five (using HBA and HBD definitions) are there for this compound?",
 'How many hydrogen bond acceptors does this compound have?',
 "How many hydrogen bond donors are there in this compound, calculated according to Lipinski's original rules (i.e., counting NH and OH groups)?",
 'How many hydrogen bond donors does this compound have?',
 'What is the molecular weight of this compound\'s parent molecule?',
 ]

q_all = q_bi + q_num  # 13 + 12 = 25

q_idx = {'Is this molecule characterized by a small molecular structure or a protein sequence?': [
    "It has a small molecule structure.",
    "It has both."
]}


def get_bi_label(q, ans):
    if q in q_idx:
        ans_list = q_idx[q]
        # if ans not in ans_list:
        #     ans_list.append(ans)
        return ans_list.index(ans)
    ans = ans.lower().strip(" .")
    if ans == "yes":
        return 1
    if ans == "no":
        return 0
    logging.warning(f"Got ans='{ans}' for binary question='{q}'")

def get_label(qa_pairs):
    labels = []
    loss_weight = []
    for q in q_all:
        found = False
        for q_, a_ in qa_pairs:
            if q_ == q:
                weight = 1
                if q in q_bi:
                    target = get_bi_label(q, a_)
                    if target is None:
                        target = 0
                        weight = 0
                elif q in q_num:
                    target = float(a_)
                else:
                    raise RuntimeError
                labels.append(target)
                loss_weight.append(weight)
                found = True
                break
        if not found:
            labels.append(0)
            loss_weight.append(0)
    return labels, loss_weight


class LinearProbeFeatDataset(Dataset):
    def __init__(self, datapath) -> None:
        super().__init__()
        with open(datapath, "rb") as f:
            data = pickle.load(f)

        self.data = []
        for idx, (smi, dd) in enumerate(data.items()):
            labels, loss_weight = get_label(dd["QA"])
            labels = torch.tensor(labels, dtype=torch.float)
            loss_weight = torch.tensor(loss_weight, dtype=torch.float)
            qa = {"feat": dd["img_feat"], "labels": labels, "loss_weight": loss_weight}
            self.data.append(qa)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        ret = self.data[index]
        return ret

    @staticmethod
    def collater(samples):
        out = default_collate(samples)
        return out
