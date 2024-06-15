# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wikipedia dataset from DPR code for ORQA."""

import csv
import random
import time
from abc import ABC

import numpy as np
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger("wikipedia-dataset")

def get_open_retrieval_wiki_dataset(args, tokenizer, text_format):

    dataset = OpenRetrievalEvidenceDataset(args, '2018 Wikipedia from DPR codebase',
                                           'evidence',
                                           args.evidence_data_path,
                                           tokenizer,
                                           args.retriever_seq_length,
                                           text_format)
    return dataset

class OpenRetrievalEvidenceDataset(ABC, Dataset):
    """Open Retrieval Evidence dataset class."""

    def __init__(self, args, task_name, dataset_name, datapath, tokenizer, max_seq_length, text_format):
        # Store inputs.
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.text_format = text_format
        logger.info(' > building {} dataset for {} from {}'.format(self.task_name,
                                                             self.dataset_name, datapath))
        # Process the files.
        self.samples, self.id2text = self.process_samples_from_single_path(datapath, args.evidence_number)

        logger.info('  >> total number of samples: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        text = self.text_format.format(**{'title': row['title'], 'text': row['text']})
        tokenized_output = self.tokenizer(text, truncation=True,
                                            padding="max_length",
                                            max_length=self.max_seq_length,
                                            return_tensors='pt')
        input_ids = tokenized_output['input_ids']
        attention_mask = tokenized_output['attention_mask']
        token_type_ids = tokenized_output['token_type_ids']
        sample = {
            'row_id': row['doc_id'],
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }
        return sample
    
    def get_evidence(self, idx):
        return self.id2text[idx]

    @staticmethod
    def process_samples_from_single_path(filename, evidence_number=None):
        start_time = time.time()

        logger.info(' > Processing {} ...'.format(filename))
        total = 0

        rows = []
        id2text = {}

        with open(filename, "r", encoding="utf-8") as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            next(reader, None)  # skip the headers
            for row in reader:
                # file format: doc_id, doc_text, title
                doc_id = int(row[0])
                text = row[1]
                title = row[2]

                rows.append({'doc_id': doc_id,
                             'text': text,
                             'title': title})

                assert doc_id not in id2text
                id2text[doc_id] = (text, title)

                total += 1
                if total % 100000 == 0:
                    logger.info('  > processed {} rows so far ...'.format(total))
                if evidence_number is not None and total == evidence_number:
                    break

        logger.info(' >> processed {} samples.'.format(len(rows)))

        return rows, id2text
