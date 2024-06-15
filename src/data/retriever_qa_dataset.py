"""Dataset for qa pairs."""

import csv
import random
import time
from abc import ABC

import numpy as np
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger("retriever-qa-dataset")

def get_retriever_qa_dataset(args, dataset_name, split, qa_file, tokenizer):

    dataset = QAPairDataset(args, dataset_name, split, qa_file,
                                           tokenizer,
                                           args.retriever_query_seq_length)
    return dataset


class QAPairDataset(ABC, Dataset):
    """Open Retrieval Evidence dataset class."""

    def __init__(self, args, dataset_name, split, datapath, tokenizer, max_seq_length):
        # Store inputs.
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        logger.info(' > building {} dataset for {} from {}'.format(self.dataset_name,
                                                             self.split, datapath))
        # Process the files.
        self.samples = self.process_samples_from_single_path(datapath)

        logger.info('  >> total number of samples: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        text = row['question']
        tokenized_output = self.tokenizer(text, truncation=True,
                                            padding="max_length",
                                            max_length=self.max_seq_length,
                                            return_tensors='pt')
        input_ids = tokenized_output['input_ids']
        attention_mask = tokenized_output['attention_mask']
        token_type_ids = tokenized_output['token_type_ids']
        sample = {
            'idx': row['idx'],
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'answers': row['answers'],
            'question': text,
        }
        if "long_answers" in row:
            sample["long_answers"] = row["long_answers"]
        return sample

    @staticmethod
    def process_samples_from_single_path(filename):
        logger.info(' > Processing {} ...'.format(filename))
        total = 0
        rows = []
        with open(filename, 'r', encoding="utf-8") as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for index, row in enumerate(reader):
                question = row[0]
                if "eli5" in filename:
                    question = eval(question)[0]
                if "asqa" in filename:
                    long_answers = eval(row[2])
                else:
                    long_answers = None
                answers = eval(row[1])
                idx = index + 1
                current_data = {
                    'idx': index + 1,
                    'question': question,
                    'answers': answers
                }
                if long_answers is not None:
                    current_data['long_answers'] = long_answers
                rows.append(current_data)

                total += 1
                if total % 100000 == 0:
                    logger.info('  > processed {} rows so far ...'.format(total))
                # if total % 101 == 0:
                #     break

        logger.info(' >> processed {} samples.'.format(len(rows)))

        return rows
    
    @staticmethod
    def collate_fn(batch):
        new_batch = {}
        # print(batch[0])
        for key in batch[0]:
            new_batch[key] = []
        for b in batch:
            for key in b:
                new_batch[key].append(b[key])
        for key in batch[0]:
            if isinstance(batch[0][key], torch.Tensor):
                new_batch[key] = torch.cat(new_batch[key], dim=0)
        return new_batch

            
