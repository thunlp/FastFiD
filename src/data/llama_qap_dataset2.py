import torch
from abc import ABC
import json
from torch.utils.data import Dataset
import logging

logger = logging.getLogger("llama-dataset")

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
        self.with_extractive_loss = args.with_extractive_loss
        self.with_generative_loss = args.with_generative_loss
        self.with_title = args.with_title
        if self.with_title:
            # self.before_context_pattern = 'Question: {question:s} Title: {title:s} Context: '
            self.context_pattern = "Document [{doc_idx:d}](Title: {title:s}): {text:s}\n"
        else:
            # self.before_context_pattern = 'Question: {question:s} Context: '
            self.context_pattern = "Document [{doc_idx:d}]: {text:s}\n"
        self.prompt_pattern = "\n\nAnswer:"
        self.before_context_pattern = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\nQuestion: {question:s}\n\n"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        question = sample['question']
        answers = sample['answers']
        idx = sample['idx']
        item = {}
        input_text = "<s>" + self.before_context_pattern.format(question=question)
        start_positions_text = []
        end_positions_text = []
        context_pattern_prefix = self.context_pattern.split("{text:s}")[0]
        for ctx_index in range(self.topk):
            ctx = sample['ctxs'][ctx_index]
            text = ctx['text']
            title = ctx['title']
            original_len = len(input_text)
            input_text += self.context_pattern.format(doc_idx=ctx_index + 1, title=title, text=text)
            context_prefix = context_pattern_prefix.format(doc_idx=ctx_index + 1, title=title)
            original_len += len(context_prefix)
            if self.with_extractive_loss:
                for s, e in zip(ctx['start'], ctx['end']):
                    start_positions_text.append(s + original_len)
                    end_positions_text.append(e + original_len)
        input_text = input_text[:-1] # remove last \n
        tokenized_output = self.tokenizer(input_text,
                                        padding=False,
                                        truncation="only_first",
                                        max_length=self.encoder_seq_length,
                                        add_special_tokens=False,
                                        return_offsets_mapping=True)
        offset_mapping = tokenized_output["offset_mapping"]
        context_ids = tokenized_output['input_ids']
        label_text = self.prompt_pattern + " " + answers[0]
        tokenized_output = self.tokenizer(label_text,
                                        padding=False,
                                        truncation="only_first",
                                        max_length=self.encoder_seq_length,
                                        add_special_tokens=False)
        label_ids = tokenized_output['input_ids'][1:] # remove space
        label_prompt = self.tokenizer(self.prompt_pattern,
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
            offset_mapping = offset_mapping[:-reduce_len]
        else:
            reduce_context_ids = context_ids
        input_ids = reduce_context_ids + label_ids
        labels = reduce_context_ids[1:] + label_ids + [self.tokenizer.eos_token_id]
        loss_mask_len = len(reduce_context_ids) + len(label_prompt_ids) - 1
        labels[:loss_mask_len] = [-100] * loss_mask_len
        attention_mask = [1] * len(input_ids)
        # for select generative
        only_labels = labels[loss_mask_len:]
        only_labels_input = input_ids[loss_mask_len:]
        sys_prompt_ids = self.tokenizer(
            "<s>" + self.before_context_pattern.format(question=question),
            padding=False,
            truncation="only_first",
            max_length=self.encoder_seq_length,
            add_special_tokens=False,
        )['input_ids']
        prompt_mask = [1] * len(sys_prompt_ids) + [0] * (len(reduce_context_ids) - len(sys_prompt_ids)) + [1] * (len(label_prompt_ids) - 1)
        # extractive data
        if self.with_extractive_loss:
            extractive_info = self.align_start_end_with_tokenizer(start_positions_text, end_positions_text, offset_mapping)
            item['extractive'] = extractive_info
        # generate data
        plus_generation_len = len(context_ids) + len(label_prompt_ids) + self.decoder_seq_length
        if plus_generation_len > self.encoder_seq_length:
            reduce_len = plus_generation_len - self.encoder_seq_length
            reduce_gen_context_ids = context_ids[:-reduce_len]
        else:
            reduce_gen_context_ids = context_ids
        generate_input_ids = reduce_gen_context_ids + label_prompt_ids
        generate_attention_mask = [1] * len(generate_input_ids)
        generate_prompt_mask = [1] * len(sys_prompt_ids) + [0] * (len(reduce_gen_context_ids) - len(sys_prompt_ids)) + [1] * (len(label_prompt_ids) - 1)
        
        item.update({
            'input_ids': input_ids,
            "labels": labels,
            "context_ids": reduce_context_ids + label_prompt_ids[:-1],
            "prompt_mask": prompt_mask,
            "only_labels": only_labels,
            "only_labels_input": only_labels_input,
            "attention_mask": attention_mask,
            "generate_input_ids": generate_input_ids,
            "generate_attention_mask": generate_attention_mask,
            "generate_prompt_mask": generate_prompt_mask,
            'answers': answers,
            'question': question,
            'idx': idx,
            "total_len": total_len,
        })
        return item
    
    def align_start_end_with_tokenizer(self, start_positions_text, end_positions_text, offset_mapping):
        t5_tokenized_starts = []
        t5_tokenized_ends = []
        for s, e in zip(start_positions_text, end_positions_text):
            assert e >= s
            t5_s = -1
            t5_e = -1
            for index, offset in enumerate(offset_mapping):
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
        assert len(t5_tokenized_starts) == len(t5_tokenized_ends)
        if len(t5_tokenized_starts) == 0:
            t5_tokenized_starts.append(0)
            t5_tokenized_ends.append(0)
        return {
            "global_context_starts": t5_tokenized_starts,
            "global_context_ends": t5_tokenized_ends,
        }
    
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
        # generate prompt mask
        max_len = max([len(b) for b in new_batch["generate_prompt_mask"]])
        for i in range(batch_size):
            current_len = len(new_batch["generate_prompt_mask"][i])
            new_batch["generate_prompt_mask"][i] = [0] * (max_len - current_len) + new_batch["generate_prompt_mask"][i]
        new_batch["generate_prompt_mask"] = torch.tensor(new_batch["generate_prompt_mask"], dtype=torch.long)
        # context_ids
        context_ids = "context_ids"
        context_attention_mask = "context_attention_mask"
        max_len = max([len(b) for b in new_batch[context_ids]])
        new_batch[context_attention_mask] = [None] * batch_size
        for i in range(batch_size):
            current_len = len(new_batch[context_ids][i])
            new_batch[context_ids][i] = [pad_id] * (max_len - current_len) + new_batch[context_ids][i]
            new_batch[context_attention_mask][i] = [0] * (max_len - current_len) + [1] * current_len
        new_batch[context_ids] = torch.tensor(new_batch[context_ids], dtype=torch.long)
        new_batch[context_attention_mask] = torch.tensor(new_batch[context_attention_mask], dtype=torch.long)
        # prompt mask
        max_len = max([len(b) for b in new_batch["prompt_mask"]])
        for i in range(batch_size):
            current_len = len(new_batch["prompt_mask"][i])
            new_batch["prompt_mask"][i] = [0] * (max_len - current_len) + new_batch["prompt_mask"][i]
        new_batch["prompt_mask"] = torch.tensor(new_batch["prompt_mask"], dtype=torch.long)
        # select generative data
        max_len = max([len(b) for b in new_batch["only_labels"]])
        new_batch["labels_attention_mask"] = [None] * batch_size
        for i in range(batch_size):
            current_len = len(new_batch["only_labels"][i])
            new_batch["only_labels"][i] = new_batch["only_labels"][i] + [-100] * (max_len - current_len)
            new_batch["only_labels_input"][i] = new_batch["only_labels_input"][i] + [pad_id] * (max_len - current_len)
            new_batch["labels_attention_mask"][i] = [1] * current_len + [0] * (max_len - current_len)
        new_batch["only_labels"] = torch.tensor(new_batch["only_labels"], dtype=torch.long)
        new_batch["only_labels_input"] = torch.tensor(new_batch["only_labels_input"], dtype=torch.long)
        new_batch["labels_attention_mask"] = torch.tensor(new_batch["labels_attention_mask"], dtype=torch.long)
        # extractive info
        if "extractive" in new_batch:
            all_starts = []
            all_ends = []
            global_mask = []
            for i, extractive_info in enumerate(new_batch["extractive"]):
                starts = extractive_info["global_context_starts"]
                ends = extractive_info["global_context_ends"]
                all_starts.append(starts)
                all_ends.append(ends)
            max_starts = max([len(starts) for starts in all_starts])
            for i in range(batch_size):
                global_mask.append(len(all_starts[i]) * [1] + (max_starts - len(all_starts[i])) * [0])
                all_starts[i] = all_starts[i] + (max_starts - len(all_starts[i])) * [0]
                all_ends[i] = all_ends[i] + (max_starts - len(all_ends[i])) * [0]
            all_starts = torch.tensor(all_starts, dtype=torch.long)
            all_ends = torch.tensor(all_ends, dtype=torch.long)
            global_mask = torch.tensor(global_mask, dtype=torch.long)
            new_batch.pop("extractive")
            new_batch["global_context_starts"] = all_starts
            new_batch["global_context_ends"] = all_ends
            new_batch["global_mask"] = global_mask
        return new_batch


