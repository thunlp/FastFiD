import torch
from abc import ABC
import json
from torch.utils.data import Dataset
import logging
import random
import copy

logger = logging.getLogger("reader-dataset")

def get_qap_dataset(args, tokenizer, before_context_pattern, no_answer_token, return_token_type_ids):
    data_paths = {
        'train': args.qa_file_train,
        'dev': args.qa_file_dev,
        'test': args.qa_file_test,
    }
    datasets = {}
    for key in data_paths:
        if data_paths[key] is not None:
            datasets[key] = OpenQAPDataset(args, before_context_pattern, no_answer_token, return_token_type_ids, data_paths[key], key, tokenizer=tokenizer)
        else:
            datasets[key] = None
    return datasets


class OpenQAPDataset(ABC, Dataset):
    def __init__(self, args,  before_context_pattern, no_answer_token, return_token_type_ids, path, split, tokenizer):
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
        self.with_extractive_loss = args.with_extractive_loss
        self.with_generative_loss = args.with_generative_loss
        self.with_rerank_loss = args.with_rerank_loss
        self.with_title = args.with_title
        self.return_token_type_ids = return_token_type_ids
        if self.with_title:
            # self.before_context_pattern = 'Question: {question:s} Title: {title:s} Context: '
            self.before_context_pattern = before_context_pattern['title']
        else:
            # self.before_context_pattern = 'Question: {question:s} Context: '
            self.before_context_pattern = before_context_pattern['no_title']
        self.no_answer_token = no_answer_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        question = sample['question']
        answers = sample['answers']
        short_answers = None
        if "long_answers" in sample:
            answers = sample['long_answers']
            short_answers = sample["answers"]
        idx = sample['idx']
        input_texts = []
        have_answers = []
        item = {}
        if self.with_extractive_loss:
            contexts = sample['ctxs'][:self.topk]
            extractive_info = self.mark_start_end_in_topk(answers, contexts, question)
            item['extractive'] = extractive_info
        for ctx_index in range(self.topk):
            ctx = sample['ctxs'][ctx_index]
            text = ctx['text']
            title = ctx['title']
            input_texts.append(self.before_context_pattern.format(**{'question': question, 'title': title}) + text)
            have_answers.append(ctx['has_answer'])
        if not self.return_token_type_ids:
            tokenized_output = self.tokenizer(input_texts,
                                            padding="max_length",
                                            truncation=True,
                                            max_length=self.encoder_seq_length,
                                            return_tensors='pt')
        else:
            questions = [copy.deepcopy(question) for _ in input_texts]
            tokenized_output = self.tokenizer(questions, input_texts,
                                                padding="max_length",
                                                truncation=True,
                                                max_length=self.encoder_seq_length,
                                                return_token_type_ids=True,
                                                return_tensors='pt')
        input_ids = tokenized_output['input_ids']
        attention_mask = tokenized_output['attention_mask']
        item.update({
            'input_ids': input_ids.unsqueeze(0),
            'attention_mask': attention_mask.unsqueeze(0),
            'answers': answers,
            'short_answers': short_answers,
            'question': question,
            'idx': idx
        })
        if self.return_token_type_ids:
            item['token_type_ids'] = tokenized_output['token_type_ids'].unsqueeze(0)
        if self.with_generative_loss:
            decoder_tokenized_output = self.tokenizer(text_target='<extra_id_0> ' + answers[0],
                                                    padding="max_length",
                                                    truncation=True,
                                                    max_length=self.decoder_seq_length,
                                                    return_tensors='pt')
            decoder_input_ids = decoder_tokenized_output['input_ids']
            decoder_attention_mask = decoder_tokenized_output['attention_mask']
            generative_item = {
                'labels': decoder_input_ids.masked_fill_(decoder_input_ids == 0, -100),
                'decoder_attention_mask': decoder_attention_mask,
            }
            item.update(generative_item)
        if self.with_rerank_loss:
            have_answers = torch.tensor(have_answers, dtype=torch.long)
            item['have_answers'] = have_answers.unsqueeze(0)
        return item
    
    def mark_start_end_in_topk(self, answers, contexts, question):
        local_context_starts = []
        local_context_ends = []
        global_context_starts = []
        global_context_ends = []
        total_answer_num = 0
        global_have_answer = 1
        local_have_answer = []
        for context in contexts:
            current_starts, current_ends, eos_index = self.align_start_end_with_tokenizer(answers, context, question)
            total_answer_num += len(current_starts)
            if len(current_starts) == 0:
                local_have_answer.append(0)
                current_starts.append(eos_index)
                current_ends.append(eos_index)
            else:
                local_have_answer.append(1)
            local_context_starts.append(current_starts)
            local_context_ends.append(current_ends)
        if total_answer_num == 0:
            # no answer
            global_have_answer = 0
        for context_index, (current_starts, current_ends) in enumerate(zip(local_context_starts, local_context_ends)):
            global_current_starts = [context_index * self.encoder_seq_length + s for s in current_starts]
            global_current_ends = [context_index * self.encoder_seq_length + e for e in current_ends]
            global_context_starts.append(global_current_starts)
            global_context_ends.append(global_current_ends)
        return {
            "global_have_answer": global_have_answer,
            "local_have_answer": local_have_answer, 
            "local_context_starts": local_context_starts,
            "local_context_ends": local_context_ends,
            "global_context_starts": global_context_starts,
            "global_context_ends": global_context_ends,
        }
    
    def align_start_end_with_tokenizer(self, answers, context, question):
        title = context['title']
        text = context['text']
        start_positions_text = []
        end_positions_text = []
        before_context = self.before_context_pattern.format(**{'question': question, 'title': title})
        for s, e in zip(context['start'], context['end']):
            start_positions_text.append(len(before_context) + s)
            end_positions_text.append(len(before_context) + e)
        input_text = before_context + text
        if not self.return_token_type_ids:
            tokenizer_output = self.tokenizer(input_text, padding="max_length", truncation=True, max_length=self.encoder_seq_length, return_offsets_mapping=True)
            special_number = 0
        else:
            tokenizer_output = self.tokenizer(question, input_text, padding="max_length", truncation=True, max_length=self.encoder_seq_length, return_offsets_mapping=True, return_token_type_ids=True)
            special_number = 2
        eos_index = tokenizer_output['input_ids'].index(self.no_answer_token) # 1 is magic number referring to </s>
        offset_maps = tokenizer_output['offset_mapping']
        t5_tokenized_starts = []
        t5_tokenized_ends = []
        current_special_number = 0
        for s, e in zip(start_positions_text, end_positions_text):
            assert e >= s
            t5_s = -1
            t5_e = -1
            for index, offset in enumerate(offset_maps):
                if offset == (0, 0):
                    current_special_number += 1
                if current_special_number < special_number:
                    continue
                if offset == (0, 0):
                    continue
                if offset[0] == s and t5_s < 0:
                    t5_s = index
                if offset[0] > s and t5_s < 0:
                    t5_s = index - 1
                if offset[1] > e:  # the last one of offset is not included in this span
                    t5_e = index
                    break
            if t5_s < 0 or t5_e < 0:
                continue
            assert t5_s >= 0 and t5_e >= 0
            t5_tokenized_starts.append(t5_s)
            t5_tokenized_ends.append(t5_e)
        return t5_tokenized_starts, t5_tokenized_ends, eos_index
    
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
            elif isinstance(batch[0][key], dict): # extractive info
                extractive_info = new_batch.pop(key)
                extractive_info = OpenQAPDataset.collate_fn_extractive(extractive_info)
                new_batch.update(extractive_info)
        return new_batch
    
    def collate_fn_extractive(batch):
        new_batch = {}
        for key in batch[0]:
            new_batch[key] = []
        for b in batch:
            for key in b:
                new_batch[key].append(b[key])
        batch = new_batch
        global_have_answer = new_batch['global_have_answer'] # batch_size
        local_have_answer = new_batch['local_have_answer'] # batch_size, topk
        max_local_n_answer = 1
        for current_local_context_starts in batch['local_context_starts']:
            for topi_context_starts in current_local_context_starts:
                max_local_n_answer = max(max_local_n_answer, len(topi_context_starts))
        local_context_starts = []
        local_context_ends = []
        local_mask = []
        for current_local_context_starts in batch['local_context_starts']:
            local_context_starts.append([])
            local_mask.append([])
            for topi_context_starts in current_local_context_starts:
                local_context_starts[-1].append(topi_context_starts + [0] * (max_local_n_answer - len(topi_context_starts)))
                local_mask[-1].append([1] * len(topi_context_starts) + [0] * (max_local_n_answer - len(topi_context_starts)))
        for current_local_context_ends in batch['local_context_ends']:
            local_context_ends.append([])
            for topi_context_ends in current_local_context_ends:
                local_context_ends[-1].append(topi_context_ends + [0] * (max_local_n_answer - len(topi_context_ends)))
        global_context_starts = []
        global_context_ends = []
        global_mask = []
        for i, current_global_context_starts in enumerate(batch['global_context_starts']):
            global_context_starts.append([])
            if global_have_answer[i] > 0: # have answer in topk, add all answer in global starts
                for j, topi_global_starts in enumerate(current_global_context_starts):
                    if local_have_answer[i][j] > 0:
                        global_context_starts[-1].extend(topi_global_starts)
            else: # no answer in topk, add all NULL answer in global starts
                for j, topi_global_starts in enumerate(current_global_context_starts):
                    global_context_starts[-1].extend(topi_global_starts)
        for i, current_global_context_ends in enumerate(batch['global_context_ends']):
            global_context_ends.append([])
            if global_have_answer[i] > 0: # have answer in topk, add all answer in global ends
                for j, topi_global_ends in enumerate(current_global_context_ends):
                    if local_have_answer[i][j] > 0:
                        global_context_ends[-1].extend(topi_global_ends)
            else: # no answer in topk, add all NULL answer in global starts
                for j, topi_global_ends in enumerate(current_global_context_ends):
                    global_context_ends[-1].extend(topi_global_ends)
        max_global_answer = max([len(t) for t in global_context_starts])
        for i in range(len(global_context_starts)):
            global_mask.append([1] * len(global_context_starts[i]) + [0] * (max_global_answer - len(global_context_starts[i])))
            global_context_starts[i] += [0] * (max_global_answer - len(global_context_starts[i]))
            global_context_ends[i] += [0] * (max_global_answer - len(global_context_ends[i]))
        return {
            'global_have_answer': torch.tensor(global_have_answer, dtype=torch.int64), # batch_size
            'local_have_answer': torch.tensor(local_have_answer, dtype=torch.int64), # batch_size, topk
            'local_context_starts': torch.tensor(local_context_starts, dtype=torch.int64), # batch_size, topk, max_local_answer_number
            'local_context_ends': torch.tensor(local_context_ends, dtype=torch.int64), # batch_size, topk, max_local_answer_number
            'local_mask': torch.tensor(local_mask, dtype=torch.int64), # batch_size, topk, max_local_answer_number
            'global_context_starts': torch.tensor(global_context_starts, dtype=torch.int64), # batch_size, max_global_answer_number
            'global_context_ends': torch.tensor(global_context_ends, dtype=torch.int64), # batch_size, max_global_answer_number
            'global_mask': torch.tensor(global_mask, dtype=torch.int64), # batch_size, max_global_answer_number
        }


