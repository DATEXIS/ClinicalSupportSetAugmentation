import json
import random
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ICDEmbeddingCollator, ClassifcationCollator


class ReadmissionCollator:
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
            embedding_id_attention_mask[i, 0:attention_masks[i].sum(), 0:num_descs] = 1

        embedding_ids = torch.nn.utils.rnn.pad_sequence(embedding_ids,
                                                        batch_first=True,
                                                        padding_value=0)
        mask = torch.ones(labels.shape, dtype=torch.bool)
        # for i, id in enumerate(embedding_ids):
        #    x = id - 1
        #    x = x[x >= 0]
        #    mask[i, x] = 0

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


class ReadmissionDataset(torch.utils.data.Dataset):
    def __init__(self,
                 examples,
                 admission_notes,
                 short_code_index,
                 add_codes: bool = True,
                 umls_text: bool = False,
                 code_descriptions=None,
                 use_ccs=False,
                 add_null_everywhere: bool = False):
        # tokenize admission notes
        self.examples = examples
        self.admission_notes = admission_notes
        self.short_code_index = short_code_index
        self.add_codes = add_codes
        self.umls_text = umls_text
        self.code_descriptions = code_descriptions
        self.use_ccs = use_ccs
        self.add_null_everywhere = add_null_everywhere

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        if self.use_ccs:
            new_diagnoses_key = "new_diagnoses_ccs"
            persistent_diagnoses_key = "persistent_diagnoses_ccs"
            lost_diagnoses_key = "lost_diagnoses_ccs"
        else:
            new_diagnoses_key = "new_diagnoses_short_code"
            persistent_diagnoses_key = "persistent_diagnoses_short_code"
            lost_diagnoses_key = "lost_diagnoses_short_code"

        annotated_diagnoses = example[persistent_diagnoses_key] + example[lost_diagnoses_key]
        code_ids = []
        if self.add_null_everywhere:
            code_ids = [0]
        if self.add_codes:
            for code in annotated_diagnoses:
                if code is not None:
                    code_ids.append(self.short_code_index[str(code)] + 1)  # plus 1 because of the NULL embedding
        if len(code_ids) == 0:
            code_ids = [0]
        annotated_labels = torch.tensor(code_ids, dtype=torch.long)

        label_ids = [self.short_code_index[str(x)] for x in
                     example[new_diagnoses_key] + example[persistent_diagnoses_key] if x is not None]
        labels = torch.zeros(len(self.short_code_index), dtype=torch.float32)
        texts = []

        for code in code_ids:
            if code == 0:
                continue
            if str(code) in example['lost_diagnoses_ccs_descriptions']:
                desc = example['lost_diagnoses_ccs_descriptions'][str(code)]['long_name']
                if desc is None:
                    continue
                if self.umls_text:
                    desc += " : " + example['lost_diagnoses_ccs_descriptions'][str(code)]['definition'][0]
                texts.append(desc)
            if str(code) in example['persistent_diagnoses_ccs_descriptions']:
                desc = example['persistent_diagnoses_ccs_descriptions'][str(code)]['long_name']
                if desc is None:
                    continue
                if self.umls_text:
                    desc += " : " + example['persistent_diagnoses_ccs_descriptions'][str(code)]['definition'][0]
                texts.append(desc)
        if len(texts) == 0:
            texts.append("No Information")
        if len(label_ids) > 0:
            label_ids = torch.tensor(label_ids, dtype=torch.long)
            if label_ids.max() == 1083:
                print("max:", torch.max(torch.tensor(list(self.short_code_index.values()))))
                print("min:", torch.min(torch.tensor(list(self.short_code_index.values()))))

            labels[label_ids] = 1
        note = self.admission_notes[example["hadm_id"]]
        return {"admission_note": note,
                "labels": labels,
                "label_idxs": annotated_labels,
                "code_descriptions": texts}


class ReadmissionDatamodule(pl.LightningDataModule):
    def __init__(self,
                 filename: str = "../../data/mimic3_readmissions_UMLS2021AB.json",
                 admission_notes: str = "../../data/admission_notes.json",
                 code_descriptions: str = "../../data/icd_desc.txt",
                 batch_size: int = 4,
                 eval_batch_size: int = 4,
                 tokenizer_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 num_workers=0,
                 add_codes: bool = True,
                 umls_text: bool = False,
                 use_text_descriptions: bool = False,
                 use_ccs: bool = False,
                 add_null_everywhere: bool = False,
                 embedding_only:bool=False,
                 max_seq_len: int = 512):
        super().__init__()
        self.add_null_everywhere = add_null_everywhere
        self.embedding_only = embedding_only
        with open(filename, "r") as f:
            data = json.load(f)
        # flatten list of readmissions

        with open(admission_notes, "r") as f:
            admission_notes_list = json.load(f)
            self.admission_notes = {x["id"]: x["text"] for x in admission_notes_list}

        with open(code_descriptions, encoding="ISO-8859-1", mode="r") as f:
            lines = f.readlines()
            splitted = [x.split(' ', 1) for x in lines]
            self.code_descriptions = {x[0]: x[1] for x in splitted}

        admissions = []
        for patient in data:
            # sliding window
            for admission in patient["patient_history"]:
                if admission["hadm_id"] in self.admission_notes:
                    admissions.append(admission)
        self.admissions = admissions

        all_short_codes_index = {}
        idx = 0
        if use_ccs:
            for admission in self.admissions:
                for code in admission["new_diagnoses_ccs"] + admission["persistent_diagnoses_ccs"] + \
                            admission["lost_diagnoses_ccs"]:
                    if str(code) not in all_short_codes_index:
                        all_short_codes_index[str(code)] = idx
                        idx += 1
        else:
            for admission in self.admissions:
                for code in admission["new_diagnoses_short_code"] + admission["persistent_diagnoses_short_code"] + \
                            admission["lost_diagnoses_short_code"]:
                    if str(code) not in all_short_codes_index:
                        all_short_codes_index[str(code)] = idx
                        idx += 1
        self.all_short_codes_index = all_short_codes_index
        self.training_dataset = ReadmissionDataset(self.admissions[0:-5000], self.admission_notes,
                                                   self.all_short_codes_index,
                                                   add_codes=add_codes,
                                                   umls_text=umls_text,
                                                   code_descriptions=self.code_descriptions,
                                                   use_ccs=use_ccs, add_null_everywhere=self.add_null_everywhere)
        self.test_dataset = ReadmissionDataset(self.admissions[-5000:], self.admission_notes,
                                               self.all_short_codes_index, add_codes=add_codes, umls_text=umls_text,
                                               code_descriptions=self.code_descriptions,
                                               use_ccs=use_ccs, add_null_everywhere=self.add_null_everywhere)
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        if use_text_descriptions:
            self.collator = ReadmissionCollator(AutoTokenizer.from_pretrained(tokenizer_name))
        elif self.embedding_only:
            self.collator=ICDEmbeddingCollator()
        else:
            self.collator = ClassifcationCollator(AutoTokenizer.from_pretrained(tokenizer_name),
                                                  all_examples_with_null=self.add_null_everywhere)

    def train_dataloader(self):
        return DataLoader(self.training_dataset,
                          batch_size=self.batch_size,
                          collate_fn=self.collator,
                          pin_memory=True,
                          num_workers=self.num_workers,
                          shuffle=True,
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.eval_batch_size,
                          collate_fn=self.collator,
                          pin_memory=True,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.eval_batch_size,
                          collate_fn=self.collator,
                          pin_memory=True,
                          num_workers=self.num_workers,
                          persistent_workers=self.num_workers > 0)
