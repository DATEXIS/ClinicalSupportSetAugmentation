from typing import Optional, Dict, Any, List, Union

import pytorch_lightning as pl
import torch
import torchmetrics
import transformers
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torchmetrics.functional import retrieval_precision
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, BertModel
from torchmetrics.functional.retrieval import retrieval_recall
from simple_attention_model.auroc import SafeAUROC
import torch.nn.functional as F
class ClassificationModel(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 encoder_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 warmup_steps: int = 0,
                 decay_steps: int = 50_000,
                 weight_decay: float = 0.01,
                 lr: float = 2e-5,
                 optimizer_name="adam",
                 readmission_task: bool=False
                 ):
        super().__init__()
        self.encoder = BertModel.from_pretrained(encoder_model_name)
        self.encoder.pooler = None
        self.num_classes = num_classes
        self.classification_layer = torch.nn.Linear(768, num_classes)
        self.recall = torchmetrics.Recall(num_classes=num_classes)
        self.prec_metric = torchmetrics.Precision(num_classes=num_classes)
        self.f1 = torchmetrics.F1(num_classes=num_classes)
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.auroc = SafeAUROC(num_classes=self.num_classes)
        self.readmission_task = readmission_task

    def forward(self,
                input_ids,
                attention_mask):
        encoded = self.encoder(input_ids, attention_mask, return_dict=True)['last_hidden_state'][:, 0]
        logits = self.classification_layer(encoded)
        return logits

    def training_step(self, batch, batch_idx):
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)['last_hidden_state'][:,0]
        logits = self.classification_layer(encoded)
        loss = F.binary_cross_entropy_with_logits(logits, batch['labels'])
        self.log("Train/Loss", loss)
        return loss

    def test_step(self, batch, batch_idx, **kwargs) -> Optional[STEP_OUTPUT]:
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)['last_hidden_state'][:, 0]
        logits = self.classification_layer(encoded)
        loss = F.binary_cross_entropy_with_logits(logits, batch['labels'])

        self.log("Val/Loss", loss)
        '''
        for k in [10, 20, 30]:
            for pred, target, added_embeddings in zip(logits, batch['labels'], batch['mask']):
                if self.readmission_task:
                    pred_ = pred
                    target_ = target
                else:
                    pred_ = pred[added_embeddings]
                    target_ = target[added_embeddings]

                self.log("Test/Recall@" + str(k), retrieval_recall(pred_, target_, k=k),
                         on_epoch=True,
                         on_step=False,
                         sync_dist=True)
                self.log("Test/Precision@" + str(k), retrieval_precision(pred_, target_, k=k),
                         sync_dist=True,
                         on_step=False,
                         on_epoch=True)
        '''

        for pred, target, added_embeddings in zip(logits, batch['labels'], batch['mask']):
            if self.readmission_task:
                pred_ = pred
                target_ = target
            else:
                pred_ = pred[added_embeddings]
                target_ = target[added_embeddings]
            map = torchmetrics.functional.retrieval.retrieval_average_precision(pred_, target_)
            self.log("Test/mAP",
                     map,
                     sync_dist=True,
                     on_step=False,
                     on_epoch=True)
            ndcg = torchmetrics.functional.retrieval.retrieval_normalized_dcg(pred_, target_, k=30)
            self.log("Test/nDCG@30",
                     ndcg,
                     sync_dist=True,
                     on_step=False,
                     on_epoch=True)
            self.log("hp_metric",
                     ndcg,
                     sync_dist=True,
                     on_step=False,
                     on_epoch=True)
        return {"logits": logits,
                "labels": batch["labels"],
                "mask": batch['mask']}

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        logits = torch.cat([x["logits"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs]).int()
        mask = torch.cat([x["mask"] for x in outputs])

        auroc = torchmetrics.functional.auroc(logits, labels, num_classes=self.num_classes, average=None)
        auroc = auroc[auroc > 0].mean()
        self.log("Test/AUROC", auroc)
        if not self.readmission_task:
            logits[~mask] = 0
            labels[~mask] = 0
        masked_auroc = torchmetrics.functional.auroc(logits, labels, num_classes=self.num_classes, average=None)
        masked_auroc = masked_auroc[masked_auroc > 0].mean()
        self.log("Test/Masked AUROC", masked_auroc)


    def validation_step(self, batch, batch_idx, **kwargs) -> Optional[STEP_OUTPUT]:
        encoded = self.encoder(batch['input_ids'], batch['attention_mask'], return_dict=True)['last_hidden_state'][:, 0]
        logits = self.classification_layer(encoded)
        loss = F.binary_cross_entropy_with_logits(logits, batch['labels'])

        self.log("Val/Loss", loss)

        for k in [10, 20, 30]:
            for pred, target, added_embeddings in zip(logits, batch['labels'], batch['mask']):
                if self.readmission_task:
                    pred_ = pred
                    target_ = target
                else:
                    pred_ = pred[added_embeddings]
                    target_ = target[added_embeddings]

                self.log("Val/Recall@" + str(k), retrieval_recall(pred_, target_, k=k),
                         on_epoch=True,
                         on_step=False,
                         sync_dist=True)
                self.log("Val/Precision@" + str(k), retrieval_precision(pred_, target_, k=k),
                         sync_dist=True,
                         on_step=False,
                         on_epoch=True)
        for pred, target, added_embeddings in zip(logits, batch['labels'], batch['mask']):
            if self.readmission_task:
                pred_ = pred
                target_ = target
            else:
                pred_ = pred[added_embeddings]
                target_ = target[added_embeddings]
            ndcg = torchmetrics.functional.retrieval.retrieval_normalized_dcg(pred_, target_, k=30)
            map = torchmetrics.functional.retrieval.retrieval_average_precision(pred_, target_)

            self.log("Val/nDCG@30",
                     ndcg,
                     sync_dist=True,
                     on_step=False,
                     on_epoch=True)
            self.log("Val/mAP", map,
                     sync_dist=True,
                     on_step=False,
                     on_epoch=True)
            self.log("hp_metric",
                     ndcg,
                     sync_dist=True,
                     on_step=False,
                     on_epoch=True)
        return {"logits": logits,
                "labels": batch["labels"],
                "mask": batch['mask']}

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        logits = torch.cat([x["logits"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs]).int()
        mask = torch.cat([x["mask"] for x in outputs])

        auroc = torchmetrics.functional.auroc(logits, labels, num_classes=self.num_classes, average=None)
        auroc = auroc[auroc > 0].mean()
        self.log("Val/AUROC", auroc)
        if not self.readmission_task:
            logits[~mask] = 0
            labels[~mask] = 0
        masked_auroc = torchmetrics.functional.auroc(logits, labels, num_classes=self.num_classes, average=None)
        masked_auroc = masked_auroc[masked_auroc > 0].mean()
        self.log("Val/Masked AUROC", masked_auroc)

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        weight_decay = 0.01
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
                weight_decay
        }, {
            'params':
                [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
                0.0
        }]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr, weight_decay=weight_decay)

        scheduler = transformers.get_polynomial_decay_schedule_with_warmup(optimizer, self.warmup_steps,
                                                                           self.decay_steps)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
        }

        return [optimizer], [scheduler]
