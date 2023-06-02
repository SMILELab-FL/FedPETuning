"""define evaluation scripts for FedETuning """

import torch
import numpy as np
from abc import ABC

from utils.register import registry


class BaseEval(ABC):
    def __init__(self, device, metric):
        self.device = device
        self.metric = metric
        self.task_name = metric.task_name
        self.logger = registry.get("logger")

    def test_and_eval(self, valid_dl, model, model_type, model_output_mode):
        raise NotImplementedError


@registry.register_eval("glue")
class GlueEval(BaseEval, ABC):
    def __init__(self, device, metric):
        super(GlueEval, self).__init__(device, metric)

    def test_and_eval(self, valid_dl, model, model_type, model_output_mode):
        model.to(self.device)

        eval_loss, nb_eval_steps = 0.0, 0
        preds, out_label_ids = None, None
        results = {}

        for batch in valid_dl:
            model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]
                          }
                if model_type != 'distilbert':
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                    inputs['token_type_ids'] = \
                        batch[2] if model_type in ['bert', 'xlnet'] else None
                outputs = model(inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results["eval_loss"] = eval_loss
        if model_output_mode == "seq_classification":
            preds = np.argmax(preds, axis=1)
        elif model_output_mode == "regression":
            preds = np.squeeze(preds)

        result = self.metric.calculate_metric(preds, out_label_ids, False)
        results.update(result)

        return results


@registry.register_eval("conll")
class CoNLLEval(BaseEval, ABC):
    def __init__(self, device, metric):
        super(CoNLLEval, self).__init__(device, metric)

    def test_and_eval(self, valid_dl, model, model_type, model_output_mode):
        model.to(self.device)

        eval_loss, nb_eval_steps = 0.0, 0
        preds, out_label_ids = None, None
        results = {}

        for batch in valid_dl:
            model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]
                          }
                if model_type != 'distilbert':
                    # XLM, DistilBERT and RoBERTa don't use segment_ids
                    inputs['token_type_ids'] = \
                        batch[2] if model_type in ['bert', 'xlnet'] else None
                outputs = model(inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results["eval_loss"] = eval_loss
        if model_output_mode == "seq_classification":
            preds = np.argmax(preds, axis=1)
        elif model_output_mode == "regression":
            preds = np.squeeze(preds)
        elif model_output_mode == "token_classification":
            preds = np.argmax(preds, axis=-1)

        label_list = registry.get("id2label")
        result = self.metric.calculate_metric(preds, out_label_ids, label_list, False)
        results.update(result)

        return results



