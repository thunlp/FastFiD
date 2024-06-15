import torch
from abc import ABC
import json
from torch.utils.data import Dataset
import logging

logger = logging.getLogger("llama-dataset")

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_qap_dataset(args, tokenizer, before_context_pattern, no_answer_token, return_token_type_ids):
    data_paths = {
        'train': args.qa_file_train,
        'dev': args.qa_file_dev,
        'test': args.qa_file_test,
    }
    datasets = {}
    for key in data_paths:
        if data_paths[key] is not None:
            datasets[key] = OpenQAPDataset(args, data_paths[key], key, tokenizer=tokenizer)
        else:
            datasets[key] = None
    return datasets


class OpenQAPDataset(ABC, Dataset):
    def __init__(self, args, path, split, tokenizer):
        logger.info(' > building dataset for {}:'.format(split))
        logger.info(' >> Processing {} ...'.format(path))
        with open(path, 'r', encoding="utf-8") as f:
            if args.iterable_load:
                import ijson
                logger.info(' >> Processing dataset in iterable way ...')
                samples = ijson.items(f, 'item')
                self.samples = []
                for sample in samples:
                    self.samples.append(sample)
            else:
                self.samples = json.load(f)
        logger.info(' >> Processed {} samples.'.format(len(self.samples)))
        self.tokenizer = tokenizer
        self.topk = args.topk_retrievals
        self.encoder_seq_length = args.encoder_seq_length
        self.decoder_seq_length = args.decoder_seq_length
        self.with_title = args.with_title
        if self.with_title:
            # self.before_context_pattern = 'Question: {question:s} Title: {title:s} Context: '
            self.context_pattern = "Document [{doc_idx:d}](Title: {title:s}): {text:s}\n"
        else:
            # self.before_context_pattern = 'Question: {question:s} Context: '
            self.context_pattern = "Document [{doc_idx:d}]: {text:s}\n"
        self.prompt_pattern = "\n\nQuestion: {question:s}\nAnswers:"
        self.before_context_pattern = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\n"
        self.before_context_pattern = B_INST + ' ' + B_SYS + self.before_context_pattern + E_SYS
        self.prompt_pattern = self.prompt_pattern + ' ' + E_INST

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        question = sample['question']
        answers = sample['answers']
        idx = sample['idx']
        item = {}
        input_text = "<s>" + self.before_context_pattern
        for ctx_index in range(self.topk):
            ctx = sample['ctxs'][ctx_index]
            text = ctx['text']
            title = ctx['title']
            input_text += self.context_pattern.format(doc_idx=ctx_index + 1, title=title, text=text)
        input_text = input_text[:-1] # remove last \n
        tokenized_output = self.tokenizer(input_text,
                                        padding=False,
                                        truncation="only_first",
                                        max_length=self.encoder_seq_length,
                                        add_special_tokens=False)
        context_ids = tokenized_output['input_ids']
        label_text = self.prompt_pattern.format(question=question) + " " + answers[0]
        tokenized_output = self.tokenizer(label_text,
                                        padding=False,
                                        truncation="only_first",
                                        max_length=self.encoder_seq_length,
                                        add_special_tokens=False)
        label_ids = tokenized_output['input_ids'][1:] # remove space
        label_prompt = self.tokenizer(self.prompt_pattern.format(question=question),
                                      padding=False,
                                      truncation="only_first",
                                      max_length=self.encoder_seq_length,
                                      add_special_tokens=False)
        label_prompt_ids = label_prompt['input_ids'][1:] # remove space
        total_len = len(context_ids) + len(label_ids)
        # training data
        if total_len > self.encoder_seq_length:
            reduce_len = total_len - self.encoder_seq_length
            reduce_context_ids = context_ids[:-reduce_len]
        else:
            reduce_context_ids = context_ids
        input_ids = reduce_context_ids + label_ids
        labels = reduce_context_ids[1:] + label_ids + [self.tokenizer.eos_token_id]
        loss_mask_len = len(reduce_context_ids) + len(label_prompt_ids) - 1
        labels[:loss_mask_len] = [-100] * loss_mask_len
        attention_mask = [1] * len(input_ids)
        # generate data
        plus_generation_len = len(context_ids) + len(label_prompt_ids) + self.decoder_seq_length
        if plus_generation_len > self.encoder_seq_length:
            reduce_len = plus_generation_len - self.encoder_seq_length
            reduce_context_ids = context_ids[:-reduce_len]
        else:
            reduce_context_ids = context_ids
        generate_input_ids = reduce_context_ids + label_prompt_ids
        generate_attention_mask = [1] * len(generate_input_ids)
        
        item.update({
            'input_ids': input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "generate_input_ids": generate_input_ids,
            "generate_attention_mask": generate_attention_mask,
            'answers': answers,
            'question': question,
            'idx': idx
        })
        return item
    
    @staticmethod
    def collate_fn(batch):
        new_batch = {}
        # print(batch[0])
        for key in batch[0]:
            new_batch[key] = []
        for b in batch:
            for key in b:
                new_batch[key].append(b[key])
        # train data
        input_key = "input_ids"
        label_key = "labels"
        attention_mask_key = "attention_mask"
        pad_id = 0
        batch_size = len(batch)
        max_len = max([len(b) for b in new_batch[input_key]])
        for i in range(batch_size):
            current_len = len(new_batch[input_key][i])
            new_batch[input_key][i] = new_batch[input_key][i] + [pad_id] * (max_len - current_len)
            new_batch[label_key][i] = new_batch[label_key][i] + [-100] * (max_len - current_len)
            new_batch[attention_mask_key][i] = new_batch[attention_mask_key][i] + [0] * (max_len - current_len)
        new_batch[input_key] = torch.tensor(new_batch[input_key], dtype=torch.long)
        new_batch[label_key] = torch.tensor(new_batch[label_key], dtype=torch.long)
        new_batch[attention_mask_key] = torch.tensor(new_batch[attention_mask_key], dtype=torch.long)
        # generate data
        generate_input_ids = "generate_input_ids"
        generate_attention_mask = "generate_attention_mask"
        max_len = max([len(b) for b in new_batch[generate_input_ids]])
        for i in range(batch_size):
            current_len = len(new_batch[generate_input_ids][i])
            new_batch[generate_input_ids][i] = [pad_id] * (max_len - current_len) + new_batch[generate_input_ids][i]
            new_batch[generate_attention_mask][i] = [0] * (max_len - current_len) + new_batch[generate_attention_mask][i]
        new_batch[generate_input_ids] = torch.tensor(new_batch[generate_input_ids], dtype=torch.long)
        new_batch[generate_attention_mask] = torch.tensor(new_batch[generate_attention_mask], dtype=torch.long)
        return new_batch


