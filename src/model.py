from typing import Optional

import pytorch_lightning as pl
import torch
import torch.distributed
import torch.nn.functional as F
import torchmetrics
import transformers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torchmetrics.functional.retrieval import retrieval_recall, retrieval_precision
from transformers import AutoModel

from code_only_transformer import SupportSetOnlyModel


class QKVAttentionLayer(torch.nn.Module):
    def __init__(self, dimension: int = 768, use_projection: bool = True, use_sigmoid_attention: bool = True):
        super().__init__()
        if use_projection:
            self.value_layer = torch.nn.Linear(dimension, dimension)
            self.query_layer = torch.nn.Linear(dimension, dimension)
            self.key_layer = torch.nn.Linear(dimension, dimension)
        else:
            self.value_layer = torch.nn.Identity()
            self.query_layer = torch.nn.Identity()
            self.key_layer = torch.nn.Identity()
        self.use_sigmoid_attention = use_sigmoid_attention

    def forward(self, queries, values, mask=None):
        torch.nn.Identity()
        queries = self.query_layer(queries)
        values = self.value_layer(values)
        keys = self.key_layer(values)
        score = torch.bmm(queries, keys.transpose(1, 2))
        if mask is not None:
            score = score.masked_fill(~mask, -1e9)
        if self.use_sigmoid_attention:
            attn = torch.sigmoid(score)
        else:
            attn = torch.softmax(score, dim=-1)
        context = torch.bmm(attn, values)
        return context


class QVAttentionLayer(torch.nn.Module):
    def __init__(self, dimension: int = 768, use_projection: bool = True, use_sigmoid_attention: bool = True):
        super().__init__()
        if use_projection:
            self.value_layer = torch.nn.Linear(dimension, dimension)
            self.query_layer = torch.nn.Linear(dimension, dimension)
        else:
            self.value_layer = torch.nn.Identity()
            self.query_layer = torch.nn.Identity()
        self.use_sigmoid_attention = use_sigmoid_attention

    def forward(self, queries, values, mask=None):
        queries = self.query_layer(queries)
        values = self.value_layer(values)
        score = torch.bmm(queries, values.transpose(1, 2))

        if mask is not None:
            score = score.masked_fill(~mask, -1e9)

        if self.use_sigmoid_attention:
            attn = torch.sigmoid(score)
        else:
            attn = torch.softmax(score, dim=-1)

        context = torch.bmm(attn, values)
        return context


class MultiHeadICDEncoder(torch.nn.Module):
    def __init__(self,
                 num_codes: int = 3964,
                 embedding_dim: int = 128,
                 num_attn_layers: int = 1,
                 num_attention_heads: int = 4
                 ):
        super().__init__()
        self.icd_embedding_layer = torch.nn.Embedding(num_codes, embedding_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dim, dim_feedforward=4 * embedding_dim,
                                                         nhead=num_attention_heads,
                                                         batch_first=True)
        self.mh_aggregation_layer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_attn_layers)

    def forward(self, embedding_ids, attention_mask):
        BS = len(embedding_ids)
        icd_embeddings = self.icd_embedding_layer(embedding_ids).view(BS, -1)
        mask = attention_mask[:, 0, :]
        self_attention_icd = self.mh_aggregation_layer(icd_embeddings,
                                                       src_key_padding_mask=~mask.view(len(mask), -1))[:, 0]
        return self_attention_icd


class CLSICDAttention(torch.nn.Module):
    def __init__(self,
                 dim: int = 768,
                 use_sigmoid_attention: bool = True
                 ):
        super().__init__()
        self.cross_attention_layer = QVAttentionLayer(dim,
                                                      use_projection=True,
                                                      use_sigmoid_attention=use_sigmoid_attention)

    def forward(self, bert_embeddings, icd_embeddings, attention_mask):
        attended_bert_embeddings = self.cross_attention_layer(bert_embeddings, icd_embeddings, mask=attention_mask)
        aggregation = torch.cat((attended_bert_embeddings, bert_embeddings), dim=-1)
        return torch.cat(bert_embeddings, attended_bert_embeddings)


class SupportSetModel(pl.LightningModule):
    def __init__(self,
                 num_icd_codes: int = 1228,  # PAD,BOS,EOS
                 encoder_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 warmup_steps: int = 0,
                 decay_steps: int = 500000,
                 lr: float = 2e-5,
                 encoder_lr: float = 2e-5,
                 use_qkv_attention: bool = False,
                 use_attention_projection: bool = True,
                 use_intermediate_ff: bool = True,
                 use_intermediate_activation: bool = True,
                 intermediate_projection_dim: int = 768,
                 use_aggregation: bool = False,
                 use_sigmoid_attention: bool = True,
                 num_attention_heads=12,
                 num_attention_layers=2,
                 use_mh_icd_aggregation: bool = False,
                 mh_icd_checkpoint: str = "",
                 num_mh_icd_attn_heads: int = 12,
                 num_mh_icd_attn_layers: int = 2,
                 icd_embedding_dim: int = 768,
                 readmission_task: bool = False,
                 use_bert_for_embeddings: bool = False,
                 use_dedicated_bert_for_embeddings: bool = False,
                 ignore_added_error: bool = False,
                 aggregation_red_dim: int = 768
                 ):
        super().__init__()
        if encoder_model_name == "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" or encoder_model_name == "bert-base-uncased":
            self.encoder = AutoModel.from_pretrained(encoder_model_name, add_pooling_layer=False)
        else:
            self.encoder = AutoModel.from_pretrained(encoder_model_name)
        self.num_icd_codes = num_icd_codes
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.lr = lr
        self.encoder_lr = encoder_lr
        self.use_qkv_attention = use_qkv_attention
        self.use_projection = use_attention_projection
        self.use_intermediate_ff = use_intermediate_ff
        self.use_intermediate_activation = use_intermediate_activation
        self.intermediate_projection_dim = intermediate_projection_dim
        self.use_aggregation = use_aggregation
        self.use_sigmoid_attention = use_sigmoid_attention
        self.use_mh_icd_aggregation = use_mh_icd_aggregation
        self.readmission_task = readmission_task
        self.mh_icd_checkpoint = mh_icd_checkpoint
        self.pretrained_mh_icd = False
        self.use_bert_for_embeddings = use_bert_for_embeddings
        self.use_dedicated_bert_for_embeddings = use_dedicated_bert_for_embeddings
        self.ignore_added_error = ignore_added_error

        # 1. Attention between ICD NULL Embedding and CLS Embedding

        # 2. ICD-Embedding Multihead-Attention + Concat

        # 3. All Token Attention between all BERT
        self.icd_embedding_dim = icd_embedding_dim
        if use_bert_for_embeddings:
            if use_dedicated_bert_for_embeddings:
                self.icd_embedding_layer = AutoModel.from_pretrained(encoder_model_name, add_pooling_layer=False)
            if self.icd_embedding_dim != 768:
                self.bert_downprojection = torch.nn.Linear(768, self.icd_embedding_dim)
        else:
            self.icd_embedding_dim = icd_embedding_dim
            if icd_embedding_dim != 768 and not use_mh_icd_aggregation:
                self.icd_embedding_layer = torch.nn.Embedding(num_icd_codes + 1, self.icd_embedding_dim)
                self.bert_downprojection = torch.nn.Linear(768, self.icd_embedding_dim)
            else:
                self.icd_embedding_layer = torch.nn.Embedding(num_icd_codes + 1, icd_embedding_dim)
        output_dim = self.icd_embedding_dim
        if self.use_aggregation:
            output_dim = self.icd_embedding_dim * 2
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=aggregation_red_dim,
                                                             dim_feedforward=4 * aggregation_red_dim,
                                                             nhead=num_attention_heads,
                                                             batch_first=True)
            self.attention_downprojection = torch.nn.Linear(output_dim, aggregation_red_dim)
            self.aggregation_transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_attention_layers)
            output_dim = aggregation_red_dim

        if use_mh_icd_aggregation:
            if self.mh_icd_checkpoint != "":
                self.mh_aggregation_layer = SupportSetOnlyModel.load_from_checkpoint(self.mh_icd_checkpoint)
                output_dim = self.mh_aggregation_layer.output_layer.out_features + 768
                self.pretrained_mh_icd = True
            else:
                output_dim = self.icd_embedding_dim * 2
                encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.icd_embedding_dim,
                                                                 dim_feedforward=4 * self.icd_embedding_dim,
                                                                 nhead=num_mh_icd_attn_heads,
                                                                 batch_first=True)
                self.mh_aggregation_layer = torch.nn.TransformerEncoder(encoder_layer,
                                                                        num_layers=num_mh_icd_attn_layers)
        else:
            self.cross_attention_layer = QVAttentionLayer(self.icd_embedding_dim, use_projection=self.use_projection,
                                                          use_sigmoid_attention=use_sigmoid_attention)
        # self.layer_norm = torch.nn.LayerNorm(self.icd_embedding_dim)
        if self.use_intermediate_activation:
            self.projection = torch.nn.Linear(output_dim, self.intermediate_projection_dim)
            self.output_layer = torch.nn.Linear(self.intermediate_projection_dim, num_icd_codes)
        else:
            self.output_layer = torch.nn.Linear(output_dim, num_icd_codes)
            self.projection = torch.nn.Identity()

        self.save_hyperparameters()

    def forward(self,
                input_ids,
                attention_mask,
                icd_embeddings,
                icd_embedding_attention_mask=None,
                code_description_input_ids=None,
                code_description_attention_mask=None,
                num_codes_per_admission=None):
        BS = icd_embeddings.shape[0]
        NUM_ICD_EMBEDDINGS = icd_embeddings.shape[1]

        bert_embeddings = self.encoder(input_ids,
                                       attention_mask,
                                       return_dict=True)["last_hidden_state"]

        mask = icd_embedding_attention_mask
        if not self.use_aggregation:
            bert_embeddings = bert_embeddings[:, 0].view(len(input_ids), 1, -1)
            if self.use_mh_icd_aggregation:
                mask = mask[:, 0, :]
            else:
                mask = icd_embedding_attention_mask[:, 0, :].reshape(bert_embeddings.shape[0], 1, -1)

        if self.use_mh_icd_aggregation:
            if self.pretrained_mh_icd:
                self_attention_icd = self.mh_aggregation_layer(icd_embeddings, mask)
            else:
                if self.use_bert_for_embeddings:
                    if self.use_dedicated_bert_for_embeddings:
                        icd_embeddings = self.icd_embedding_layer(code_description_input_ids,
                                                                  code_description_attention_mask,
                                                                  return_dict=True)['last_hidden_state'][:, 0]
                        if self.icd_embedding_dim != 768:
                            icd_embeddings = self.bert_downprojection(icd_embeddings)
                    else:
                        icd_embeddings = self.encoder(code_description_input_ids,
                                                      code_description_attention_mask,
                                                      return_dict=True)['last_hidden_state'][:, 0]
                        if self.icd_embedding_dim != 768:
                            icd_embeddings = self.bert_downprojection(icd_embeddings)
                    # pad
                    i = 0
                    embeddings_per_admission = []
                    for length in num_codes_per_admission:
                        embeddings_per_admission.append(icd_embeddings[i:i + length])
                        i += length
                    icd_embeddings = torch.nn.utils.rnn.pad_sequence(embeddings_per_admission, batch_first=True)

                else:
                    icd_embeddings = self.icd_embedding_layer(icd_embeddings).view(BS, NUM_ICD_EMBEDDINGS, -1)
                self_attention_icd = self.mh_aggregation_layer(icd_embeddings,
                                                               src_key_padding_mask=~mask.view(len(mask), -1))[:, 0]
            bert_embeddings = bert_embeddings.view(-1, 768)
            aggregation = torch.cat((bert_embeddings, self_attention_icd), dim=-1)
        else:
            if self.use_bert_for_embeddings:
                if self.use_dedicated_bert_for_embeddings:
                    icd_embeddings = self.icd_embedding_layer(code_description_input_ids,
                                                              code_description_attention_mask,
                                                              return_dict=True)['last_hidden_state'][:, 0]
                    if self.icd_embedding_dim != 768:
                        icd_embeddings = self.bert_downprojection(icd_embeddings)
                else:
                    icd_embeddings = self.encoder(code_description_input_ids,
                                                  code_description_attention_mask,
                                                  return_dict=True)['last_hidden_state'][:, 0]
                    if self.icd_embedding_dim != 768:
                        icd_embeddings = self.bert_downprojection(icd_embeddings)
                # pad
                i = 0
                embeddings_per_admission = []
                for length in num_codes_per_admission:
                    embeddings_per_admission.append(icd_embeddings[i:i + length])
                    i += length
                icd_embeddings = torch.nn.utils.rnn.pad_sequence(embeddings_per_admission, batch_first=True)
            else:
                icd_embeddings = self.icd_embedding_layer(icd_embeddings).view(BS, NUM_ICD_EMBEDDINGS, -1)
            if self.icd_embedding_dim != 768:
                bert_embeddings = self.bert_downprojection(bert_embeddings)
            attended_bert_embeddings = self.cross_attention_layer(bert_embeddings, icd_embeddings, mask=mask)
            aggregation = torch.cat((attended_bert_embeddings, bert_embeddings), dim=-1)
            if self.use_aggregation:
                aggregation = self.attention_downprojection(aggregation)
                aggregation = self.aggregation_transformer(aggregation, src_key_padding_mask=~attention_mask)[:, 0]
            else:
                aggregation = aggregation.view(len(input_ids), -1)
        if self.use_intermediate_activation:
            aggregation = torch.relu(self.projection(aggregation))
        else:
            aggregation = self.projection(aggregation)

        output = self.output_layer(aggregation)
        return output

    def training_step(self, batch, batch_idx):
        if self.use_bert_for_embeddings:
            output = self(batch["input_ids"], batch["attention_mask"], batch["icd_embedding_ids"],
                          batch["icd_embedding_attention_mask"],
                          code_description_input_ids=batch["code_description_tokens"],
                          code_description_attention_mask=batch["code_description_attention_masks"],
                          num_codes_per_admission=batch["codes_per_note"])
        else:
            output = self(batch["input_ids"], batch["attention_mask"], batch["icd_embedding_ids"],
                          batch["icd_embedding_attention_mask"])
        if self.ignore_added_error:
            output *= batch['mask']
            batch['labels'] *= batch['mask']
            loss = F.binary_cross_entropy_with_logits(output, batch["labels"])
        else:
            loss = F.binary_cross_entropy_with_logits(output, batch["labels"])
        self.log("Train/Loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, **kwargs) -> Optional[STEP_OUTPUT]:
        if self.use_bert_for_embeddings:
            logits = self(batch["input_ids"], batch["attention_mask"], batch["icd_embedding_ids"],
                          batch["icd_embedding_attention_mask"],
                          code_description_input_ids=batch["code_description_tokens"],
                          code_description_attention_mask=batch["code_description_attention_masks"],
                          num_codes_per_admission=batch["codes_per_note"])
        else:
            logits = self(batch["input_ids"], batch["attention_mask"], batch["icd_embedding_ids"],
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
        if self.use_bert_for_embeddings:
            logits = self(batch["input_ids"], batch["attention_mask"], batch["icd_embedding_ids"],
                          batch["icd_embedding_attention_mask"],
                          code_description_input_ids=batch["code_description_tokens"],
                          code_description_attention_mask=batch["code_description_attention_masks"],
                          num_codes_per_admission=batch["codes_per_note"])
        else:
            logits = self(batch["input_ids"], batch["attention_mask"], batch["icd_embedding_ids"],
                          batch["icd_embedding_attention_mask"])
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
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
            ndcg = torchmetrics.functional.retrieval.retrieval_normalized_dcg(pred_, target_, k=30)
            map = torchmetrics.functional.retrieval.retrieval_average_precision(pred_, target_)

            self.log("Test/nDCG@30",
                     ndcg,
                     sync_dist=True,
                     on_step=False,
                     on_epoch=True)
            self.log("Test/mAP", map,
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

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
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
        encoder_parameters = set(self.encoder.parameters())
        if self.use_dedicated_bert_for_embeddings and self.use_bert_for_embeddings:
            bert_embedding_params = set(self.icd_embedding_layer.parameters())
            rest_parameters = list(set(self.parameters()) - encoder_parameters - bert_embedding_params)
            encoder_parameters = list(encoder_parameters) + list(bert_embedding_params)
        else:
            rest_parameters = list(set(self.parameters()) - encoder_parameters)
            encoder_parameters = list(encoder_parameters)

        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": self.encoder_lr},
                {"params": rest_parameters, "lr": self.lr}
            ], lr=self.lr, weight_decay=0.01)
        scheduler = transformers.get_polynomial_decay_schedule_with_warmup(optimizer, self.warmup_steps,
                                                                           self.decay_steps)
        scheduler = {'scheduler': scheduler,
                     'interval': 'step'}

        return [optimizer], [scheduler]
