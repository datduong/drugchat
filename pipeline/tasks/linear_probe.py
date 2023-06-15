"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch
from pipeline.common.registry import registry
from pipeline.tasks.base_task import BaseTask
from sklearn.metrics import classification_report


def get_bi_cls_report(logits, labels):
    pred = (logits.detach() > 0).to(torch.int32)
    pred = pred.cpu().numpy()
    report = classification_report(labels, pred, output_dict=True)
    # report is a dict {'0': {}, '1': {}, 'macro avg': {}, 'weighted avg': {}}
    # return report['macro avg']
    return report


@registry.register_task("linear_probe")
class LinearProbeTask(BaseTask):
    def __init__(self):
        super().__init__()

    def valid_step(self, model, samples):
        ret = model(samples)
        # returns a dict with {key: tensor}
        return ret

    def before_evaluation(self, model, dataset, **kwargs):
        pass
        # model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, val_result, split_name, epoch):
        """
        val_result: a list of `ret` returned by valid_step
        """
        out = {}
        keys = list(val_result[0].keys())
        for key in keys:
            tensors = [res[key] for res in val_result]
            if tensors[0].ndim == 0:
                out[key] = torch.stack(tensors)
            else:
                out[key] = torch.cat(tensors)
        logits = out.pop("logits_bi")
        labels = out.pop("labels_bi")

        q_rec = {}
        num_q = 13
        assert len(logits) % num_q == 0
        for i in range(num_q):
            lo = logits[i::num_q]
            la = labels[i::num_q]
            rep = get_bi_cls_report(lo, la)
            q_rec[f"q{i}"] = rep
        rep = get_bi_cls_report(logits, labels)

        for k in list(out.keys()):
            out[k] = out[k].mean().item()
        out.update(rep)
        out.update(q_rec)

        # "agg_metrics" is needed for selecting best checkpoint
        out["agg_metrics"] = -out["loss"]

        return out