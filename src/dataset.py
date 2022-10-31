import json
import random
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np


class ClassificationWithDescriptionCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, data):
        admission_notes = [x['admission_note'] for x in data]
        labels = torch.stack([x['labels'] for x in data])
        code_descriptions = [x['code_descriptions'] for x in data]
        num_descs_per_note = torch.tensor([len(x) for x in code_descriptions], dtype=torch.long)

        # flatten and tokenize code_descriptions
        # look at this cool double list comprehension no one understands and everyone has to google
        # Here is the link for you: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
        code_descriptions = [item for sublist in code_descriptions for item in sublist]
        tokenized_code_descriptions = self.tokenizer(code_descriptions,
                                                     padding=True,
                                                     truncation=True,
                                                     max_length=self.max_seq_len,
                                                     return_tensors="pt")

        tokenized = self.tokenizer(admission_notes,
                                   padding=False,
                                   truncation=True,
                                   max_length=self.max_seq_len,
                                   )
        input_ids = [torch.tensor(x) for x in tokenized["input_ids"]]
        attention_masks = [torch.tensor(x, dtype=torch.bool) for x in tokenized["attention_mask"]]
        lengths = torch.tensor([len(x) for x in input_ids])
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks,
                                                          batch_first=True,
                                                          padding_value=0)

        embedding_ids = [x["label_idxs"] for x in data]
        max_embedding_id_count = torch.max(num_descs_per_note)

        embedding_id_attention_mask = torch.zeros((attention_masks.shape[0],
                                                   attention_masks.shape[1],
                                                   max_embedding_id_count), dtype=torch.bool)
        for i, num_descs in enumerate(num_descs_per_note):
            embedding_id_attention_mask[i, :, 0:num_descs] = 1

        mask = torch.ones(labels.shape, dtype=torch.bool)
        for i, id in enumerate(embedding_ids):
            x = id - 1
            x = x[x >= 0]
            mask[i, x] = 0

        embedding_ids = torch.nn.utils.rnn.pad_sequence(embedding_ids,
                                                        batch_first=True,
                                                        padding_value=0)

        return {"input_ids": input_ids,
                "attention_mask": attention_masks,
                "labels": labels,
                "icd_embedding_ids": embedding_ids,
                "icd_embedding_attention_mask": embedding_id_attention_mask,
                "lengths": lengths,
                "code_description_tokens": tokenized_code_descriptions.input_ids,
                "code_description_attention_masks": tokenized_code_descriptions.attention_mask,
                "codes_per_note": num_descs_per_note,
                "mask": mask}


class ClassifcationCollator:
    def __init__(self, tokenizer: AutoTokenizer, max_seq_len: int = 512, all_examples_with_null:bool=False):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.all_examples_with_null=all_examples_with_null

    def __call__(self, data):
        admission_notes = [x['admission_note'] for x in data]
        labels = torch.stack([x['labels'] for x in data])
        tokenized = self.tokenizer(admission_notes,
                                   padding=False,
                                   truncation=True,
                                   max_length=self.max_seq_len,
                                   )
        input_ids = [torch.tensor(x) for x in tokenized["input_ids"]]
        attention_masks = [torch.tensor(x, dtype=torch.bool) for x in tokenized["attention_mask"]]
        lengths = torch.tensor([len(x) for x in input_ids])
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                    batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks,
                                                          batch_first=True,
                                                          padding_value=0)

        embedding_ids = [x["label_idxs"] for x in data]
        max_embedding_id_count = max(len(x) for x in embedding_ids)
        embedding_id_attention_mask = torch.zeros((attention_masks.shape[0],
                                                   attention_masks.shape[1],
                                                   max_embedding_id_count), dtype=torch.bool)
        for i, embed_id in enumerate(embedding_ids):
            if len(embed_id) == 1:
                embedding_id_attention_mask[i, :, 0] = 1
            else:
                if self.all_examples_with_null:
                    embedding_id_attention_mask[i, :, 0:len(embed_id)] = 1
                else:
                    embedding_id_attention_mask[i, :, 1:len(embed_id)] = 1

        mask = torch.ones(labels.shape, dtype=torch.bool)
        for i, id in enumerate(embedding_ids):
            x = id - 1
            x = x[x >= 0]
            mask[i, x] = 0

        embedding_ids = torch.nn.utils.rnn.pad_sequence(embedding_ids,
                                                        batch_first=True,
                                                        padding_value=0)
        return {"input_ids": input_ids,
                "attention_mask": attention_masks,
                "labels": labels,
                "icd_embedding_ids": embedding_ids,
                "icd_embedding_attention_mask": embedding_id_attention_mask,
                "lengths": lengths,
                "mask": mask}


class ICDEmbeddingCollator:
    def __init__(self):
        pass

    def __call__(self, data):
        labels = torch.stack([x['labels'] for x in data])

        embedding_ids = [x["label_idxs"] for x in data]
        max_embedding_id_count = max(len(x) for x in embedding_ids)
        embedding_id_attention_mask = torch.zeros((len(embedding_ids),
                                                   max_embedding_id_count), dtype=torch.bool)
        for i, embed_id in enumerate(embedding_ids):
            embedding_id_attention_mask[i, 0:len(embed_id)] = 1

        embedding_ids = torch.nn.utils.rnn.pad_sequence(embedding_ids,
                                                        batch_first=True,
                                                        padding_value=0)
        mask = torch.ones(labels.shape, dtype=torch.bool)
        for i, id in enumerate(embedding_ids):
            x = id - 1
            x = x[x >= 0]
            mask[i, x] = 0

        return {"labels": labels,
                "icd_embedding_ids": embedding_ids,
                "icd_embedding_attention_mask": embedding_id_attention_mask,
                "mask": mask}


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 examples,
                 label_lookup,
                 label_distribution,
                 num_codes_to_add: int = 0,
                 sampling_strategy: str = "random",
                 use_ccs: bool = False,
                 use_code_descriptions: bool = False,
                 use_umls: bool=False,
                 add_null: bool=False):
        # tokenize admission notes
        self.examples = examples
        self.label_lookup = label_lookup
        self.inverse_label_lookup = {v: k for k, v in label_lookup.items()}
        self.num_codes_to_add = num_codes_to_add
        self.label_distribution = label_distribution
        self.sampling_strategy = sampling_strategy
        self.use_code_descriptions = use_code_descriptions
        self.use_ccs = use_ccs
        self.use_umls = use_umls
        self.add_null = add_null

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        note = example['admission_note']
        key = "short_codes"
        if self.use_ccs:
            key = "ccs_codes"
        labels = example[key]
        hadm_id = example['hadm_id']

        label_ids = [self.label_lookup[x] for x in labels]
        label_idxs = torch.tensor([int(x) for x in label_ids])
        labels = torch.zeros(len(self.label_lookup), dtype=torch.float32)
        labels[label_idxs] = 1
        label_idxs = torch.tensor([int(x) for x in label_ids])
        # get corresponding distribution
        code_distribution = self.label_distribution[label_idxs]

        if self.sampling_strategy == "random" or self.sampling_strategy == "random_randamount":
            sample_ids = torch.randperm(len(label_idxs))
            label_idxs = label_idxs[sample_ids]
        elif self.sampling_strategy == "main":
            # order by appearance / count in distribution descending
            sorted_idxs = torch.argsort(code_distribution, dim=-1, descending=True)
            label_idxs = label_idxs[sorted_idxs]
        elif self.sampling_strategy == "weighted":
            alpha = 0.001  # add some alpha term to have no code with 0 probability in the list
            weighting = 1.0 - (code_distribution.float() / (code_distribution.float().max() + alpha))
            weighting = weighting.numpy()
            weighting = weighting / np.linalg.norm(weighting, 1)
            try:
                label_idxs = np.random.choice(label_idxs, size=len(label_idxs), p=weighting, replace=False)
                label_idxs = torch.tensor(label_idxs)
            except:
                print(weighting, weighting.sum())
        elif self.sampling_strategy == "longtail":
            # order by appearance / count in distribution ascending
            sorted_idxs = torch.argsort(code_distribution, dim=-1, descending=False)
            label_idxs = label_idxs[sorted_idxs]
        elif self.sampling_strategy == "with_wrong":
            label_idxs = label_idxs[torch.randperm(len(label_idxs))]
            mask = torch.ones(len(labels), dtype=torch.bool)
            mask[label_idxs] = 0
            all_code_ids = torch.arange(0, len(labels))
            all_code_ids_without_true = all_code_ids[mask]
            all_code_ids_without_true = all_code_ids_without_true[torch.randperm(len(all_code_ids_without_true))]
            label_idxs = torch.cat([all_code_ids_without_true[0:4], label_idxs])

        num_add = self.num_codes_to_add
        if self.sampling_strategy == "random_randamount":
            num_add = random.randint(0, len(label_idxs))
        description_texts = []
        if self.use_code_descriptions:
            if num_add == 0:
                description_texts = ["No Information"]
            else:
                text_labels = [example[key][i] for i in sample_ids[0:num_add]]

                if self.use_umls:
                    for code in text_labels:
                        desc = example['ccs_code_definitions'][str(code)]['long_names'][0]
                        desc += " : " + example['ccs_code_definitions'][str(code)]['definitions'][0]
                        description_texts.append(desc)
                else:
                    description_texts = [example['ccs_code_definitions'][str(code)]['long_names'][0] for code in text_labels]

        if num_add == 0:
            annotated_labels = torch.tensor([0])
        else:
            if self.add_null:
                annotated_labels = torch.cat((torch.tensor([0]),(label_idxs[0:num_add] + 1)))
            else:
                annotated_labels = label_idxs[0:num_add] + 1

        return {"admission_note": note,
                "labels": labels,
                "label_idxs": annotated_labels,
                "code_descriptions": description_texts,
                "hadm_id": hadm_id}


class MIMICClassificationDataModule(pl.LightningDataModule):
    def __init__(self,
                 use_code_descriptions: bool = False,
                 data_dir: str = "../../data",
                 batch_size: int = 4,
                 eval_batch_size: int = 4,
                 tokenizer_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 num_workers: int = 0,
                 num_codes_to_add: int = 0,
                 num_codes_to_add_val: int = 0,
                 sampling_strategy: str = "random",
                 val_sampling_strategy: str = "random",
                 max_seq_len: int = 512,
                 use_umls:bool=False,
                 use_ccs: bool = False,
                 embedding_only: bool = False,
                 add_null_everywhere: bool=False):
        super().__init__()

        with open(data_dir + "/test_umls-wikidata_labels.jsonl", "r") as f:
            lines = f.readlines()
            test_data = [json.loads(l) for l in lines]
        with open(data_dir + "/val_umls-wikidata_labels.jsonl", "r") as f:
            lines = f.readlines()
            validation_data = [json.loads(l) for l in lines]
        with open(data_dir + "/train_umls-wikidata_labels.jsonl", "r") as f:
            lines = f.readlines()
            training_data = [json.loads(l) for l in lines]

        # build label index
        label_idx = {}
        id = 0
        for adm_note in (training_data + test_data + validation_data):
            key = "short_codes"
            if use_ccs:
                key = "ccs_codes"

            for code in adm_note[key]:
                if code not in label_idx:
                    label_idx[code] = id
                    id += 1
        self.use_umls = use_umls
        self.training_data = training_data
        self.test_data = test_data
        self.val_data = validation_data
        self.use_code_descriptions = use_code_descriptions
        self.use_ccs = use_ccs
        self.label_idx = label_idx
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.add_null_everywhere = add_null_everywhere
        if embedding_only:
            self.collator = ICDEmbeddingCollator()
        elif self.use_code_descriptions:
            self.collator = ClassificationWithDescriptionCollator(self.tokenizer, max_seq_len)
        else:
            self.collator = ClassifcationCollator(self.tokenizer, max_seq_len, all_examples_with_null=self.add_null_everywhere)

        self.num_workers = num_workers
        self.num_codes_to_add = num_codes_to_add
        self.num_codes_to_add_val = num_codes_to_add_val
        self.sampling_strategy = sampling_strategy
        self.val_sampling_strategy = val_sampling_strategy
        self.embedding_only = embedding_only

    def setup(self, stage: Optional[str] = None):
        # calculate label distribution

        distribution = torch.zeros(len(self.label_idx), dtype=torch.long)
        for example in self.test_data + self.val_data + self.training_data:
            key = "short_codes"
            if self.use_ccs:
                key = "ccs_codes"
            codes = torch.tensor([self.label_idx[x] for x in example[key]])
            distribution[codes] += 1

        mimic_train = ClassificationDataset(self.training_data,
                                            label_lookup=self.label_idx,
                                            label_distribution=distribution,
                                            num_codes_to_add=self.num_codes_to_add,
                                            sampling_strategy=self.sampling_strategy,
                                            use_ccs=self.use_ccs,
                                            use_umls=self.use_umls,
                                            use_code_descriptions=self.use_code_descriptions,
                                            add_null=self.add_null_everywhere
                                            )

        mimic_val = ClassificationDataset(self.val_data,
                                          label_lookup=self.label_idx,
                                          label_distribution=distribution,
                                          num_codes_to_add=self.num_codes_to_add_val,
                                          sampling_strategy=self.val_sampling_strategy,
                                          use_ccs=self.use_ccs,
                                          use_umls=self.use_umls,
                                          use_code_descriptions=self.use_code_descriptions,
                                          add_null=self.add_null_everywhere)

        mimic_test = ClassificationDataset(self.test_data,
                                           label_lookup=self.label_idx,
                                           label_distribution=distribution,
                                           num_codes_to_add=self.num_codes_to_add_val,
                                           sampling_strategy=self.val_sampling_strategy,
                                           use_ccs=self.use_ccs,
                                           use_umls=self.use_umls,
                                           use_code_descriptions=self.use_code_descriptions,
                                           add_null=self.add_null_everywhere)
        self.mimic_train = mimic_train
        self.mimic_val = mimic_val
        self.mimic_test = mimic_test
        print("Val length: ", len(self.mimic_val))
        print("Train Length: ", len(self.mimic_train))

    def train_dataloader(self):
        return DataLoader(self.mimic_train,
                          batch_size=self.batch_size,
                          collate_fn=self.collator,
                          pin_memory=True,
                          num_workers=self.num_workers,
                          shuffle=True,
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.mimic_val,
                          batch_size=self.eval_batch_size,
                          collate_fn=self.collator,
                          pin_memory=True,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)

    def test_dataloader(self):
        return DataLoader(self.mimic_test,
                          batch_size=self.eval_batch_size,
                          collate_fn=self.collator,
                          pin_memory=True,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)
