from typing import Optional

import pytorch_lightning as pl
import torch
import torch.distributed
import torch.nn.functional as F
import torchmetrics
import transformers
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics.functional.retrieval import retrieval_recall, retrieval_precision

class SupportSetOnlyModel(pl.LightningModule):
    def __init__(self,
                 num_icd_codes: int = 3963,  # PAD,BOS,EOS
                 warmup_steps: int = 0,
                 decay_steps: int = 500000,
                 lr: float = 2e-5,
                 icd_embedding_dim: int=768,
                 num_mh_icd_attn_heads=8,
                 num_mh_icd_attn_layers=2,
                 readmission_task=False
                 ):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.lr = lr
        self.readmission_task = readmission_task
        self.num_icd_codes=num_icd_codes
        self.icd_embedding_layer = torch.nn.Embedding(num_icd_codes + 1, icd_embedding_dim)
        output_dim = icd_embedding_dim
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=icd_embedding_dim, nhead=num_mh_icd_attn_heads, batch_first=True)
        self.mh_aggregation_layer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_mh_icd_attn_layers)
        self.output_layer = torch.nn.Linear(output_dim, num_icd_codes)
        self.save_hyperparameters()

    def forward(self, icd_embeddings, icd_embedding_attention_mask):
        BS = icd_embeddings.shape[0]
        NUM_ICD_EMBEDDINGS = icd_embeddings.shape[1]
        mask = icd_embedding_attention_mask
        icd_embeddings = self.icd_embedding_layer(icd_embeddings).view(BS, NUM_ICD_EMBEDDINGS, -1)
        self_attention_icd = self.mh_aggregation_layer(icd_embeddings,
                                                       src_key_padding_mask=~mask.view(len(mask), -1))[:, 0]
        output = self.output_layer(self_attention_icd)
        return output

    def training_step(self, batch, batch_idx):
        
        output = self(batch["icd_embedding_ids"],
                      batch["icd_embedding_attention_mask"])
        loss = F.binary_cross_entropy_with_logits(output, batch["labels"])
        self.log("Train/Loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, **kwargs) -> Optional[STEP_OUTPUT]:
        logits = self(batch["icd_embedding_ids"],
                      batch["icd_embedding_attention_mask"])
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
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

    def test_step(self, batch, batch_idx, **kwargs) -> Optional[STEP_OUTPUT]:
        logits = self(batch["icd_embedding_ids"],
                      batch["icd_embedding_attention_mask"])
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
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
        for pred, target, added_embeddings in zip(logits, batch['labels'], batch['mask']):
            if self.readmission_task:
                pred_ = pred
                target_ = target
            else:
                pred_ = pred[added_embeddings]
                target_ = target[added_embeddings]
            ndcg = torchmetrics.functional.retrieval.retrieval_normalized_dcg(pred_, target_, k=30)
            map = torchmetrics.functional.retrieval.retrieval_average_precision(pred_, target_)
            self.log("Test/mAP",
                     map,
                     sync_dist=True,
                     on_step=False,
                     on_epoch=True)
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

    def validation_epoch_end(self,outputs) -> None:
        logits = torch.cat([x["logits"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs]).int()
        mask = torch.cat([x["mask"] for x in outputs])

        if self.trainer.num_gpus > 1:
            # all_gather logits, labels and masks
            logits_list = [torch.empty_like(logits) for _ in range(self.trainer.num_gpus)]
            labels_list = [torch.empty_like(labels) for _ in range(self.trainer.num_gpus)]
            mask_list = [torch.empty_like(mask) for _ in range(self.trainer.num_gpus)]

            torch.distributed.all_gather(logits_list, logits)
            torch.distributed.all_gather(labels_list, labels)
            torch.distributed.all_gather(mask_list, mask)
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)
            mask = torch.cat(mask_list)

        auroc = torchmetrics.functional.auroc(logits, labels, num_classes=self.num_icd_codes, average=None)
        auroc = auroc[auroc > 0].mean()
        self.log("Val/AUROC", auroc, sync_dist=False, rank_zero_only=True)

        logits[~mask] = 0
        labels[~mask] = 0
        masked_auroc = torchmetrics.functional.auroc(logits, labels, num_classes=self.num_icd_codes, average=None)
        masked_auroc = masked_auroc[masked_auroc > 0].mean()
        self.log("Val/Masked AUROC", masked_auroc)

    def test_epoch_end(self,outputs) -> None:
        logits = torch.cat([x["logits"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs]).int()
        mask = torch.cat([x["mask"] for x in outputs])

        if self.trainer.num_gpus > 1:
            # all_gather logits, labels and masks
            logits_list = [torch.empty_like(logits) for _ in range(self.trainer.num_gpus)]
            labels_list = [torch.empty_like(labels) for _ in range(self.trainer.num_gpus)]
            mask_list = [torch.empty_like(mask) for _ in range(self.trainer.num_gpus)]

            torch.distributed.all_gather(logits_list, logits)
            torch.distributed.all_gather(labels_list, labels)
            torch.distributed.all_gather(mask_list, mask)
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)
            mask = torch.cat(mask_list)

        auroc = torchmetrics.functional.auroc(logits, labels, num_classes=self.num_icd_codes, average=None)
        auroc = auroc[auroc > 0].mean()
        self.log("Test/AUROC", auroc, sync_dist=False, rank_zero_only=True)

        logits[~mask] = 0
        labels[~mask] = 0
        masked_auroc = torchmetrics.functional.auroc(logits, labels, num_classes=self.num_icd_codes, average=None)
        masked_auroc = masked_auroc[masked_auroc > 0].mean()
        self.log("Test/Masked AUROC", masked_auroc)


    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = transformers.get_polynomial_decay_schedule_with_warmup(optimizer, self.warmup_steps,
                                                                           self.decay_steps)
        scheduler = {'scheduler': scheduler,
                     'interval': 'step',
                     }

        return [optimizer], [scheduler]
