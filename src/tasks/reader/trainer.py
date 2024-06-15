import os
import json
import torch
import math
import collections
from collections import OrderedDict
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    SequentialSampler,
    RandomSampler,
)
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import logging
from transformers import is_torch_available, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.modeling_outputs import Seq2SeqLMOutput
from data.qap_dataset import get_qap_dataset
from tasks.reader.eval_utils import exact_match_score, metric_max_over_ground_truths, postprocess
from tasks.reader.eval_utils import extraction_recall_score, rouge_match_score, answer_recall_score
from tasks.reader.eval_utils import metric_mean_over_ground_truths, answer_strem_score
READER_MODEL_TYPE=os.getenv('READER_MODEL_TYPE', 't5')
if READER_MODEL_TYPE == 't5':
    from tasks.reader.t5reader import Reader
elif READER_MODEL_TYPE == 'electra':
    from tasks.reader.electrareader import Reader
elif READER_MODEL_TYPE == "llama":
    from models.llama.llama_reader import Reader
    from data.llama_qap_dataset import get_qap_dataset
elif READER_MODEL_TYPE == "llama-chat":
    from models.llama.llama_reader import Reader
    from data.llama_chat_qap_dataset import get_qap_dataset
else:
    raise ValueError("No reader model based on {}".format(READER_MODEL_TYPE))
from utils import LossTracker
import profiler
import wandb
import deepspeed

logger = logging.getLogger("reader-trainer")

RawResult = collections.namedtuple("RawResult", ["unique_id", "input_ids", "sequence_lengths", "start_probs", "end_probs", "selected_probs"])

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available

def build_dataloader(args, dataset, batch_size, shuffle):
    world_size = args.world_size
    rank = args.rank
    num_workers = args.num_workers
    if world_size > 1:
        sampler = DistributedSampler(dataset, rank=rank, shuffle=shuffle)
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            sampler=sampler,
                            collate_fn=dataset.collate_fn,
                            pin_memory=True)
    return dataloader, sampler

def deepspeed_initialize(args, model, train_dataset_length=0):
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    with open(args.deepspeed, 'r') as f:
        ds_config = json.load(f)
    ds_config['bf16']['enabled'] = args.bf16
    ds_config['train_micro_batch_size_per_gpu'] = args.batch_size
    ds_config['steps_per_print'] = args.log_interval
    ds_config['gradient_clipping'] = args.clip_grad
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    ds_config['train_batch_size'] = args.global_batch_size
    if args.train:
        ds_config['optimizer']['params']['lr'] = args.lr
        ds_config['optimizer']['params']['weight_decay'] = args.weight_decay
        if args.max_train_steps is None:
            gradient_accumulation_steps = args.gradient_accumulation_steps
            micro_batch_size = args.global_batch_size // gradient_accumulation_steps
            one_epoch_micro_steps = (train_dataset_length - 1) // micro_batch_size + 1
            num_training_steps = math.ceil(one_epoch_micro_steps * args.train_epochs // gradient_accumulation_steps)
            warmup_steps = int(num_training_steps * args.warmup)
        else:
            num_training_steps = args.max_train_steps
            warmup_steps = int(num_training_steps * args.warmup)
        scheduler = {'type': None, 'params': {}}
        if args.lr_scheduler == 'linear':
            scheduler['type'] = 'WarmupDecayLR'
        elif args.lr_scheduler == 'constant':
            scheduler['type'] = 'WarmupLR'
        else:
            raise ValueError(f"{args.lr_scheduler} is not supported.")   
        scheduler['params']['warmup_min_lr'] = 0.0
        scheduler['params']['warmup_max_lr'] = args.lr
        scheduler['params']['warmup_num_steps'] = warmup_steps
        scheduler['params']["warmup_type"] = 'linear'
        if args.lr_scheduler == 'linear':
            scheduler['params']['total_num_steps'] = num_training_steps
        ds_config['scheduler'] = scheduler
    else:
        ds_config.pop('optimizer')
        ds_config.pop('scheduler')
    if "zero_optimization" in ds_config and ds_config['zero_optimization']['stage'] == 3:
        hidden_size = model.config.d_model
        ds_config["zero_optimization"]["reduce_bucket_size"] = hidden_size * hidden_size
        ds_config["zero_optimization"]["stage3_prefetch_bucket_size"] = 0.9 * hidden_size * hidden_size
        ds_config["zero_optimization"]["stage3_param_persistence_threshold"] = 10 * hidden_size
    deepspeed_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model_parameters,
        config_params=ds_config,
    )
    return deepspeed_engine, optimizer, lr_scheduler



class Trainer(object):
    def __init__(self, args):
        self._setup_wandb(args)
        self.args = args
        set_seed(args.seed)
        self.reader = Reader(args)
        self.tokenizer = self.reader.tokenizer
        if args.load is not None:
            # load model from checkpoint
            self.load_checkpoint(args, self.reader, args.load)
        self.world_size = args.world_size
        self.rank = args.rank
        self.datasets = get_qap_dataset(args, self.tokenizer, Reader.before_context_pattern, Reader.no_answer_token, Reader.return_token_type_ids)
        if args.record_cross_attention:
            self.reader.register_cross_attention_hook()
        self.reset_cross_attentions()
        if args.bf16 and not args.deepspeed:
            self.reader = self.reader.to(torch.bfloat16)
        if args.deepspeed is None:
            if args.world_size > 1:
                dist.barrier()
                logger.warning(f"Running DDP on rank {self.rank} / {self.world_size}")
                self.reader = self.reader.to(args.local_rank)
                self.wrape_reader = DDP(self.reader, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
                dist.barrier()
            else:
                logger.warning("Move model to GPU")
                self.reader.cuda()
                # self.reader.to('cpu')
                self.wrape_reader = self.reader
        else:
            # deepspeed initialize
            if args.train:
                self.wrape_reader, self.optimizer, self.lr_scheduler = deepspeed_initialize(args, self.reader, len(self.datasets['train']))
                self.reader = self.wrape_reader.module
            else:
                self.wrape_reader, self.optimizer, self.lr_scheduler = deepspeed_initialize(args, self.reader)
                self.reader = self.wrape_reader.module

    def train(self, args):
        self.wrape_reader.train()
        train_dataloader, train_sampler = build_dataloader(args, self.datasets['train'], batch_size=args.batch_size, shuffle=True)
        # if args.rank == 0:
        #     for index, first_batch in enumerate(train_dataloader):
        #         print("Batch {}".format(index))
        #         for key in first_batch:
        #             if isinstance(first_batch[key], torch.Tensor):
        #                 print(key, first_batch[key].shape)
        #             else:
        #                 print(key, type(key))
        #         i = 0
        #         print(self.tokenizer.decode(first_batch['input_ids'][i, :]))
        #         print(self.tokenizer.decode(first_batch['labels'][i, :].masked_fill(first_batch['labels'][i, :] == -100, 0)))
        #         print("answers:", first_batch['answers'][i])
                # print("short_answers:", first_batch["short_answers"])
                # print("local_mask:", first_batch['local_mask'][i])
                # print("local answers:")
                # for k in range(25):
                #     input_ids = first_batch['input_ids'][i, k, :]
                #     local_context_starts = first_batch['local_context_starts'][i, k, :].tolist()
                #     local_context_ends = first_batch['local_context_ends'][i,k,:].tolist()
                #     local_mask = first_batch['local_mask'][i, k, :].tolist()
                #     for start, end, mask in zip(local_context_starts, local_context_ends, local_mask):
                #         if mask > 0:
                #             print(k, start, end)
                #             print(self.tokenizer.decode(input_ids[start:end+1]))
        #         print("global answers:")
        #         print(first_batch.keys())
        #         global_context_starts = first_batch['global_context_starts'][i].tolist()
        #         global_context_ends = first_batch['global_context_ends'][i].tolist()
        #         global_mask = first_batch['global_mask'][i].tolist()
        #         input_ids = first_batch['input_ids']#.view(first_batch['input_ids'].shape[0], -1)
        #         print(input_ids.shape)
        #         for start, end, mask in zip(global_context_starts, global_context_ends, global_mask):
        #             if mask > 0:
        #                 print(start, end)
        #                 print(self.tokenizer.decode(input_ids[i, start:end+1]))
        #                 print(self.tokenizer.convert_ids_to_tokens(input_ids[i, start:end+1]))
        #         print(first_batch['global_context_starts'])
        #         print(first_batch['idx'])
        # torch.distributed.barrier()
        # quit()
        gradient_accumulation_steps = args.gradient_accumulation_steps
        total_step = math.ceil(len(train_dataloader) * args.start_epoch / gradient_accumulation_steps)
        total_micro_step = 0
        if args.max_train_steps is None:
            num_training_steps = math.ceil(len(train_dataloader) * args.train_epochs / gradient_accumulation_steps)
            start_epoch = args.start_epoch
            end_epoch = args.train_epochs
        else:
            num_training_steps = args.max_train_steps
            start_epoch = args.start_epoch
            end_epoch = num_training_steps * gradient_accumulation_steps // len(train_dataloader) + 1
        if args.deepspeed is None:
            optimizer, lr_scheduler = self._setup_optimizer(args, self.reader, num_training_steps)
            if not args.no_load_optim and args.load is not None:
                self.load_checkpoint(args, self.reader, args.load, no_load_model=True, optimizer=optimizer, lr_scheduler=lr_scheduler)
        else:
            optimizer = self.optimizer
            lr_scheduler = self.lr_scheduler
        best_metric = -1
        best_epoch = -1
        loss_tracker = LossTracker()
        logger.info(" > Begin Training ...")
        logger.info(f" >> num traning steps           = {num_training_steps}")
        logger.info(f" >> global batch size           = {args.global_batch_size}")
        logger.info(f" >> instances per epoch         = {len(self.datasets['train'])}")
        logger.info(f" >> gradient accumulation steps = {gradient_accumulation_steps}")
        with profiler.profile(enabled=args.time_analysis) as timer:
            for epoch in range(start_epoch, end_epoch):
                if total_step >= num_training_steps:
                    break
                if isinstance(train_sampler, DistributedSampler):
                    train_sampler.set_epoch(epoch + args.seed)
                epoch_loss = 0
                data_iter = enumerate(train_dataloader)
                while True:
                    with timer.record('get_batch_data'):
                        try:
                            local_step, batch = next(data_iter)
                        except StopIteration:
                            break
                    inputs = self.prepare_input(batch)
                    # for i in range(8):
                    #     torch.distributed.barrier()
                    #     if self.args.rank == i:
                    #         print(f"step {local_step} rank {self.args.rank}:")
                    #         print("keys:", inputs.keys())
                    #         print("context length:", inputs["context_ids"].shape, "prompt mask length:", inputs["prompt_mask"].shape)
                    #         print(inputs["prompt_mask"])
                    #         temp_ids = inputs["context_ids"][0].masked_select(inputs["prompt_mask"][0] == 1)
                    #         print(self.tokenizer.decode(temp_ids, skip_special_tokens=False))
                    #         print("context ids:")
                    #         print(self.tokenizer.decode(inputs["context_ids"][0], skip_special_tokens=False))
                    #         print("input_ids:")
                    #         print(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False))
                    #         print(inputs["only_labels_input"])
                    #         print(inputs["only_labels"])
                    #         print(inputs["labels_attention_mask"])
                    #         print(self.tokenizer.batch_decode(inputs["only_labels_input"], skip_special_tokens=False))
                    #         print(inputs["global_start_positions"])
                    torch.distributed.barrier()
                    # with torch.autograd.profiler.record_function("forward"): # label the block
                    with timer.record('forward'):
                        loss_dict = self.wrape_reader(
                            **inputs
                        )
                    loss = loss_dict['loss']
                    loss_tracker.track(**{k: loss_dict[k] for k in loss_dict if 'loss' in k})
                    epoch_loss += loss.item()
                    # print(total_micro_step, loss_dict)
                    total_micro_step += 1
                    if args.deepspeed is None:
                        is_gradient_accumulation = (total_micro_step % gradient_accumulation_steps == 0)
                    else:
                        is_gradient_accumulation = self.wrape_reader.is_gradient_accumulation_boundary()
                    if args.deepspeed is None:
                        loss = loss / args.gradient_accumulation_steps
                        # with torch.autograd.profiler.record_function("backward"): # label the block
                        with timer.record('backward'):
                            loss.backward()
                    else:
                        #runs backpropagation
                        with timer.record('backward'):
                            self.wrape_reader.backward(loss)
                        self.wrape_reader.step()
                    if is_gradient_accumulation:
                        total_step += 1
                        if args.deepspeed is None:
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.wrape_reader.parameters(), self.args.clip_grad, norm_type=2)
                            self.optimizer_step(optimizer, lr_scheduler)
                        else:
                            grad_norm = self.wrape_reader.get_global_grad_norm()
                        loss_tracker.track(grad_norm=grad_norm)
                        if total_step % args.log_interval == 0:
                            self.log_step(suffix="Training |",
                                        log_dict=loss_tracker.get_loss_dict(),
                                        wandb_suffix='train',
                                        num_updates=total_step,
                                        lr=lr_scheduler.get_last_lr()[0])
                            loss_tracker.reset()
                            if args.time_analysis:
                                timer.log(logger=logger)
                        # end of epoch training, begin save and evaluate
                        if args.save is not None and args.save_strategy == 'step' and total_step % args.save_step == 0:
                            logger.info("Epoch {:02d} | Save checkpoint into {}".format(epoch, os.path.join(args.save, 'step{}'.format(total_step))))
                            os.makedirs(args.save, exist_ok=True)
                            self.save_checkpoint(args, self.reader, optimizer, lr_scheduler, os.path.join(args.save, 'step{}'.format(total_step)))
                        if args.eval_strategy == 'step' and total_step % args.eval_step == 0:
                            results = self.evaluate(args, dataset=self.datasets['dev'])
                            metrics = results[0]
                            self.wrape_reader.train() # return to train mode
                            self.log_step(metrics, suffix='Evaluation Results |', epoch=epoch, wandb_suffix='dev', num_updates=total_step)
                            if not args.eval_only_loss:
                                # save best
                                em = metrics[args.eval_metric]
                                if em > best_metric:
                                    best_metric = em
                                    best_epoch = epoch
                                    if self.rank == 0 and args.save is not None:
                                        self.save_checkpoint(args, self.reader, optimizer, lr_scheduler, os.path.join(args.save, 'best'))
                                logger.info('Best Results | Epoch: {:02d} {}: {:.5f}'.format(best_epoch, args.eval_metric, best_metric))
                        if total_step == args.train_iters_epoch:
                            break
                # end of epoch training, begin save and evaluate
                if args.save is not None and args.save_strategy == 'epoch':
                    logger.info("Epoch {:02d} | Save checkpoint into {}".format(epoch, os.path.join(args.save, 'epoch{}'.format(epoch))))
                    os.makedirs(args.save, exist_ok=True)
                    self.save_checkpoint(args, self.reader, optimizer, lr_scheduler, os.path.join(args.save, 'epoch{}'.format(epoch)))
                if args.eval_strategy == 'epoch':
                    results = self.evaluate(args, dataset=self.datasets['dev'])
                    metrics = results[0]
                    self.wrape_reader.train() # return to train mode
                    self.log_step(metrics, suffix='Evaluation Results |', epoch=epoch, wandb_suffix='dev', num_updates=total_step)
                    if not args.eval_only_loss:
                        # save best
                        em = metrics[args.eval_metric]
                        if em > best_metric:
                            best_metric = em
                            best_epoch = epoch
                            if self.rank == 0 and args.save is not None:
                                self.save_checkpoint(args, self.reader, optimizer, lr_scheduler, os.path.join(args.save, 'best'))
                        logger.info('Best Results | Epoch: {:02d} {}: {:.5f}'.format(best_epoch, args.eval_metric, best_metric))
                epoch_loss = 0
                # end of epoch
        if args.wandb_project is not None and args.rank == 0:
            wandb.finish()
    
    @torch.no_grad()
    def evaluate(self, args, dataset, return_predictions=False):
        model = self.reader
        # model = self.wrape_reader
        model.eval()
        logger.info('Begin evaluation over {} instances'.format(len(dataset)))
        eval_dataloader, eval_sampler = build_dataloader(args, dataset, batch_size=args.eval_batch_size, shuffle=False)
        no_padding_num = ((len(dataset) - 1) % self.world_size) + 1
        predictions = []
        references = []
        idxs = []
        questions = []
        all_cross_attention_length = []
        max_cross_attention_length = []
        loss_tracker = LossTracker()
        with profiler.profile(enabled=args.time_analysis) as timer:
            for local_step, batch in enumerate(tqdm(eval_dataloader,desc='Evaluate generation', disable=(self.rank != 0 or args.disable_tqdm))):
                if args.eval_iters is not None and local_step == args.eval_iters:
                    break
                inputs = self.prepare_input(batch)
                # for i in range(8):
                #     torch.distributed.barrier()
                #     if self.args.rank == i:
                #         print(f"rank {self.args.rank}:")
                #         input_ids = inputs["input_ids"][0]
                #         input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
                #         for i in range(len(input_text)):
                #             print(input_text[i])
                #             labels = inputs["labels"][i].tolist()
                #             new_labels = []
                #             new_tokens = []
                #             for local_i, l in enumerate(labels):
                #                 if l != -100:
                #                     new_labels.append(l)
                #                     new_tokens.append(input_ids[i][local_i].item())
                #             print(self.tokenizer.convert_ids_to_tokens(new_labels))
                #             print(self.tokenizer.convert_ids_to_tokens(new_tokens))
                #             print("=" * 80)
                # torch.distributed.barrier()
                # quit()
                loss_dict = model(
                    **inputs,
                    record_cross_attention=False,
                )
                loss_tracker.track(**{k: loss_dict[k] for k in loss_dict if 'loss' in k})
                if 'cross_attentions' in loss_dict:
                    self.record_cross_attentions(loss_dict['cross_attentions'], loss_dict['cross_attention_masks'])
                # if self.rank == 0:
                #     print(local_step, len(eval_dataloader), batch['idx'])
                if args.eval_only_loss:
                    continue
                if args.eval_metric == "str-em":
                    # for asqa
                    references.extend(batch['short_answers'])
                else:
                    references.extend(batch['answers'])
                idxs.extend(batch['idx'])
                questions.extend(batch['question'])
                # generate answer
                inputs = self.prepare_input(batch, for_generate=True)
                # if self.args.rank == 0:
                #     print(inputs.keys())
                #     print(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False))
                #     print(inputs["attention_mask"][0])
                #     print(inputs["input_ids"].shape, inputs["prompt_mask"].shape)
                with timer.record('generation'):
                    output, cross_attention_length = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        prompt_mask=inputs.get('prompt_mask', None),
                        max_new_tokens=args.decoder_seq_length,
                        # max_length=args.decoder_seq_length,
                        timer=timer,
                        record_cross_attention=args.record_cross_attention,
                        num_beams=args.num_beams,
                    )
                if 'cross_attentions' in output:
                    self.record_cross_attentions(output['cross_attentions'], output['cross_attention_masks'], output.get('select_masks', None))
                if cross_attention_length is not None:
                    all_cross_attention_length.extend(cross_attention_length[0])
                    max_cross_attention_length.extend(cross_attention_length[1])
                input_len = inputs["input_ids"].shape[-1]
                output_sequences = output['sequences']
                if output_sequences.shape[1] > input_len:
                    output = output_sequences[:, input_len:]
                else:
                    output = output_sequences
                prediction = self.tokenizer.batch_decode(output, skip_special_tokens=True)
                predictions.extend([{'prediction': p} for p in prediction])
        if args.time_analysis:
            # record time and cross attention length
            time_dict = timer.to_dict()
            time_generation = time_dict['generation'] / len(predictions)
            time_encoder = time_dict['encoder'] / len(predictions) if "encoder" in time_dict else None
            loss_tracker.track(time_generation=time_generation, time_encoder=time_encoder)
            if len(all_cross_attention_length) > 0:
                best_context_length = sum(all_cross_attention_length[-len(predictions):]) / len(predictions)
                real_context_length = sum(max_cross_attention_length[-len(predictions):]) / len(predictions)
                loss_tracker.track(best_context_length=best_context_length, real_context_length=real_context_length)
        if self.world_size > 1:
            loss_tracker.all_reduce()
        loss_dict = loss_tracker.get_loss_dict()
        if not args.eval_only_loss:
            if args.rank >= no_padding_num and local_step == (len(eval_dataloader) - 1):
                predictions = predictions[:-1]
                references = references[:-1]
                idxs = idxs[:-1]
                questions = questions[:-1]
            scores = []
            for prediction, reference in zip(predictions, references):
                if args.eval_metric == "em":
                    scores.append(metric_max_over_ground_truths(exact_match_score, prediction['prediction'], reference))
                elif args.eval_metric == "rougel":
                    scores.append(metric_max_over_ground_truths(rouge_match_score, prediction['prediction'], reference))
                elif args.eval_metric == "recall":
                    scores.append(metric_max_over_ground_truths(answer_recall_score, prediction['prediction'], reference))
                elif args.eval_metric == "str-em":
                    scores.append(metric_mean_over_ground_truths(answer_strem_score, prediction['prediction'], reference))
                else:
                    raise NotImplementedError(f"{args.eval_metric} is not implemented.")
            score_num = len(scores)
            score_sum = sum(scores)
            if self.world_size > 1:
                score_sum_tensor = torch.tensor([score_sum], dtype=torch.float64).cuda()
                dist.all_reduce(score_sum_tensor, op=dist.ReduceOp.SUM)
                score_sum = score_sum_tensor.item()
                score_num_tensor = torch.tensor([score_num], dtype=torch.int64).cuda()
                dist.all_reduce(score_num_tensor, op=dist.ReduceOp.SUM)
                score_num = score_num_tensor.item()
                if return_predictions:
                    all_predictions = [None for _ in range(self.world_size)]
                    dist.all_gather_object(all_predictions, predictions)
                    reshape_predictions = []
                    if self.rank == 0:
                        for i in range(len(all_predictions[0])):
                            for j in range(self.world_size):
                                if i < len(all_predictions[j]):
                                    reshape_predictions.append(all_predictions[j][i])
                        predictions = reshape_predictions
                    else:
                        predictions = None
            # print("{}: {} {}".format(self.rank, score_sum, score_num))
            em = score_sum / score_num
            loss_dict[args.eval_metric] = em
        if args.record_cross_attention:
            if not args.with_extractive_loss:
                self.use_cross_attention_rerank(dataset)
            else:
                self.select_cross_attention()
            self.reset_cross_attentions()
        logger.info("Evaluation over!")
        if return_predictions:
            return (loss_dict, predictions)
        return (loss_dict,)
    
    @torch.no_grad()
    def evaluate_extractive(self, args, dataset, return_predictions=False):
        model = self.reader
        model.eval()
        logger.info('Begin evaluation extraction over {} instances'.format(len(dataset)))
        eval_dataloader, eval_sampler = build_dataloader(args, dataset, batch_size=args.eval_batch_size, shuffle=False)
        no_padding_num = ((len(dataset) - 1) % self.world_size) + 1
        predictions = []
        references = []
        idxs = []
        questions = []
        predictions = []
        loss_tracker = LossTracker()
        for local_step, batch in enumerate(tqdm(eval_dataloader,desc='Evaluate extraction', disable=(self.rank != 0 or args.disable_tqdm))):   
            if args.eval_iters is not None and local_step == args.eval_iters:
                break
            inputs = self.prepare_input(batch)
            loss_dict = model(
                **inputs
            )
            loss_tracker.track(**loss_dict)
            if args.eval_only_loss:
                continue
            references.extend(batch['answers'])
            idxs.extend(batch['idx'])
            questions.extend(batch['question'])
            inputs = self.prepare_input(batch, for_generate=True)
            # extract sentence
            if 'token_type_ids' not in inputs:
                topk_doc_index, topk_start, topk_end, topk_probs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            else:
                start_probs, end_probs, selected_probs = model.generate(input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'])
            # input_ids = inputs['input_ids'][0].tolist()
            # local_mask = inputs['local_mask'][0].tolist()
            # local_starts = inputs['local_start_positions'][0].tolist()
            # local_ends = inputs['local_end_positions'][0].tolist()
            # cnt = 0
            # print(batch['question'][0])
            # print(batch['answers'][0])
            # for input_id, masks, starts, ends in zip(input_ids, local_mask, local_starts, local_ends):
            #     print("Passage: " + str(cnt))
            #     print(self.tokenizer.decode(input_id))
            #     cnt += 1
            #     for mask, start, end in zip(masks, starts, ends):
            #         if mask != 0:
            #             print(self.tokenizer.decode(input_id[start:end + 1]))
            # quit()
            # selected_positive_probs = torch.unbind(selected_probs, dim=-1)[1]
            # span_probs = start_probs.unsqueeze(3) * end_probs.unsqueeze(2) * selected_positive_probs.unsqueeze(-1).unsqueeze(-1)
            idx = batch['idx']
            input_ids = inputs["input_ids"]
            if len(input_ids.shape) == 2:
                input_ids = input_ids.unsqueeze(1) # bs, 1, seq_len
            for index, unique_id in enumerate(idx):
                output = []
                prediction = []
                doc_indexes = topk_doc_index[index].tolist()
                starts = topk_start[index].tolist()
                ends = topk_end[index].tolist()
                probs = topk_probs[index].tolist()
                for doc_i, s, e, p in zip(doc_indexes, starts, ends, probs):
                    assert e >= s, f"rank: {self.args.rank} batch: {local_step} index: {index} start: {s} end: {e} prob: {p}"
                    output.append(input_ids[index, doc_i, s:e + 1])
                output = self.tokenizer.batch_decode(output, skip_special_tokens=False)
                for doc_i, s, e, o, p in zip(doc_indexes, starts, ends, output, probs):
                    prediction.append({'prediction': o, 'doc_index': doc_i, 'start': s, 'end': e, 'score': p})
                predictions.append(prediction)
        if self.world_size > 1:
            loss_tracker.all_reduce()
        loss_dict = loss_tracker.get_loss_dict()
        if not args.eval_only_loss:
            # caculate Recall
            if args.rank >= no_padding_num and local_step == (len(eval_dataloader) - 1):
                predictions = predictions[:-1]
                references = references[:-1]
                idxs = idxs[:-1]
                questions = questions[:-1]
            scores = []
            for prediction, reference in zip(predictions, references):
                scores.append(extraction_recall_score(prediction, reference, args.match))
            score_num = len(scores)
            score_tensor = torch.tensor(scores, dtype=torch.float64).cuda() # num, topk
            score_tensor = score_tensor.sum(dim=0) # topk
            if self.world_size > 1:
                score_num_tensor = torch.tensor([score_num], dtype=torch.int64).cuda()
                dist.all_reduce(score_num_tensor, op=dist.ReduceOp.SUM)
                score_num = score_num_tensor.item()
                dist.all_reduce(score_tensor, op=dist.ReduceOp.SUM)
                if return_predictions:
                    all_predictions = [None for _ in range(self.world_size)]
                    dist.all_gather_object(all_predictions, predictions)
                    reshape_predictions = []
                    if self.rank == 0:
                        for i in range(len(all_predictions[0])):
                            for j in range(self.world_size):
                                if i < len(all_predictions[j]):
                                    reshape_predictions.append(all_predictions[j][i])
                        predictions = reshape_predictions
                    else:
                        predictions = None
            # print("{}: {} {}".format(self.rank, score_sum, score_num))
            score_tensor = (score_tensor / score_num).tolist()
            for topk in args.support_sentence_topk_accuracies:
                loss_dict['Recall@{}'.format(topk)] = score_tensor[topk - 1]
        logger.info("Evaluation over!")
        if return_predictions:
            return (loss_dict, predictions)
        return (loss_dict,)
    
    def reset_cross_attentions(self):
        self.cross_attentions = {}
        self.attention_mask = []
        self.select_masks = []
        model = self.reader
        if isinstance(model, DDP):
            model = model.module
        if hasattr(model.model, 'decoder'):
            decoder_layer_num = len(model.model.decoder.block)
            for i in range(decoder_layer_num):
                self.cross_attentions[i] = []
    
    @torch.no_grad()
    def record_cross_attentions(self, cross_attentions, cross_attention_masks, select_masks=None):
        for layer_num, score in enumerate(cross_attentions):
            score = torch.clone(score.detach())
            score = torch.chunk(score, score.shape[0], dim=0) # split by batch_size
            for s in score:
                s = s.squeeze()[:, 1:, :] # remove <extra_id_0>
                # bmt.print_rank(layer_num, s.shape)
                self.cross_attentions[layer_num].append(s.cpu().detach())
        cross_attention_masks = torch.chunk(cross_attention_masks, cross_attention_masks.shape[0], dim=0) # split by batch size
        for a in cross_attention_masks:
            a = a.squeeze().cpu().detach()[1:, :] # remove <extra_id_0>
            self.attention_mask.append(a.squeeze().cpu().detach())
        if select_masks is not None:
            for s in select_masks:
                self.select_masks.append(s.cpu().detach())
    
    def use_cross_attention_rerank(self, dataset):
        logger.info("Use Cross Attention to rerank:")
        cross_attentions = self.cross_attentions
        attention_masks = self.attention_mask
        args = self.args
        world_size = self.world_size
        rank = self.rank
        topk_accuracy = [0 for _ in range(self.args.topk_retrievals)]
        topk_score = [0 for _ in range(self.args.topk_retrievals)]
        total_cnt = 0
        new_samples = []
        import copy
        for i in range(len(cross_attentions[0])):
            attention_mask = attention_masks[i]
            cross_attn = [cross_attentions[key][i] for key in cross_attentions] # all layer cross attention
            cross_attn = torch.stack(cross_attn, dim=0) # layer_num, num_heads, decoder_seq_len, encoder_cat_seq_len
            decoder_layer_num, num_heads, decoder_seq_len, encoder_cat_seq_len = cross_attn.shape
            topk_retrievals = self.args.topk_retrievals
            encoder_seq_len = self.args.encoder_seq_length
            assert topk_retrievals * encoder_seq_len == encoder_cat_seq_len
            cross_attn = cross_attn.view(decoder_layer_num, num_heads, decoder_seq_len, topk_retrievals, encoder_seq_len)
            attention_mask = attention_mask.view(1, 1, decoder_seq_len, topk_retrievals, encoder_seq_len).to(torch.int64)
            cross_attn = (cross_attn * attention_mask).permute(3, 0, 1, 2, 4)
            cross_attn = cross_attn.contiguous()
            # topk, layer_num, heads_num, decoder_seq_len, encoder_seq_len
            decoder_attention_mask = attention_mask.view(decoder_seq_len, topk_retrievals, encoder_seq_len).permute(1, 2, 0)[0, 0, :]
            decoder_real_len = torch.sum(decoder_attention_mask).item()
            cross_attn = cross_attn.view(topk_retrievals, -1)
            score = cross_attn.sum(dim=-1)
            sort_score, sort_indices = score.sort(descending=True)
            data_index = i * world_size + rank
            if data_index > len(dataset.samples):
                continue
            total_cnt += 1
            sample = dataset.samples[data_index]
            new_sample = copy.deepcopy(sample)
            new_sample['ctxs'] = []
            for j in range(topk_retrievals):
                cross_score = sort_score[j].item()
                topk_score[j] += cross_score / (decoder_layer_num * num_heads * decoder_real_len)
            for j in range(topk_retrievals):
                index = sort_indices[j].item()
                ctx = sample['ctxs'][index]
                if ctx['has_answer']:
                    for remain in range(j, topk_retrievals):
                        topk_accuracy[remain] += 1
                    break
            # save new rank
            for j in range(topk_retrievals):
                index = sort_indices[j].item()
                cross_score = sort_score[j].item() / (decoder_layer_num * num_heads * decoder_real_len)
                new_ctx = copy.deepcopy(sample['ctxs'][index])
                new_ctx['cross_attention_score'] = cross_score
                new_ctx['current_rank'] = j + 1
                new_ctx['original_rank'] = index + 1
                new_sample['ctxs'].append(new_ctx)
            new_samples.append(new_sample)
        if self.rank == 0:
            json.dump(new_samples, open("data/qaps_text_test_cross_attn.json", 'w'), indent=4, ensure_ascii=False)
        topk_accuracy = torch.tensor(topk_accuracy, dtype=torch.int64).to(rank)
        total_cnt = torch.tensor([total_cnt], dtype=torch.int64).to(rank)
        topk_score = torch.tensor(topk_score, dtype=torch.float64).to(rank)
        if world_size > 1:
            torch.distributed.all_reduce(topk_accuracy)
            torch.distributed.all_reduce(total_cnt)
            torch.distributed.all_reduce(topk_score)
        total_cnt = total_cnt.item()
        topk_accuracy = topk_accuracy.tolist()
        topk_score = topk_score.tolist()
        topk_accuracy = [t / total_cnt for t in topk_accuracy]
        topk_score = [t / total_cnt for t in topk_score]
        logger.info("Cross Attention Rerank Perfomance:")
        for i in [1, 5, 10, 20, 25]:
            logger.info("Top-{}: {:.4f}, average score: {:.4f}".format(i, topk_accuracy[i - 1], topk_score[i - 1]))
        logger.info("Score sum: {:.4f}".format(sum(topk_score)))
    
    @torch.no_grad()
    def select_cross_attention(self):
        logger.info("Count cross attention in select parts:")
        cross_attentions = self.cross_attentions
        attention_masks = self.attention_mask
        select_masks = self.select_masks
        world_size = self.world_size
        rank = self.rank
        select_cross_attn_sum = 0
        select_cross_attn_avg = 0
        non_select_cross_attn_sum = 0
        non_select_cross_attn_avg = 0
        total_cnt = 0
        for i in range(len(cross_attentions[0])):
            attention_mask = attention_masks[i]
            select_mask = select_masks[i] # encoder_cat_seq_len
            cross_attn = [cross_attentions[key][i] for key in cross_attentions] # all layer cross attention
            cross_attn = cross_attn
            cross_attn = torch.stack(cross_attn, dim=0) # layer_num, num_heads, decoder_seq_len, encoder_cat_seq_len
            decoder_layer_num, num_heads, decoder_seq_len, encoder_cat_seq_len = cross_attn.shape
            actual_decoder_seq_len = attention_mask[:, 0].sum().item()
            attention_mask = attention_mask.view(1, 1, decoder_seq_len, encoder_cat_seq_len).to(torch.int64)
            cross_attn = cross_attn * attention_mask
            select_mask = select_mask.view(1, 1, 1, encoder_cat_seq_len)
            select_cross_attn = cross_attn.masked_select(select_mask).view(decoder_layer_num, num_heads, decoder_seq_len, -1)
            select_cross_attn = select_cross_attn.sum() / (actual_decoder_seq_len * decoder_layer_num * num_heads)
            select_cross_attn_sum += select_cross_attn.item() 
            select_cross_attn_avg += select_cross_attn.item() / select_mask.sum().item() # each token atten score
            non_select_mask = ((~select_mask) & attention_mask[0, 0, 0, :]).to(torch.bool)
            non_select_cross_attn = cross_attn.masked_select(non_select_mask).view(decoder_layer_num, num_heads, decoder_seq_len, -1)
            non_select_cross_attn = non_select_cross_attn.sum() / (actual_decoder_seq_len * decoder_layer_num * num_heads)
            non_select_cross_attn_sum += non_select_cross_attn.item()
            non_select_cross_attn_avg += non_select_cross_attn.item() / non_select_mask.sum().item()
            total_cnt += 1
        if world_size > 1:
            total_cnt = torch.tensor([total_cnt], dtype=torch.int64).to(rank)
            torch.distributed.all_reduce(total_cnt)
            total_cnt = total_cnt.item()
            cross_attns = torch.tensor([select_cross_attn_sum, select_cross_attn_avg, non_select_cross_attn_sum, non_select_cross_attn_avg], dtype=torch.float64).to(rank)
            torch.distributed.all_reduce(cross_attns)
            cross_attns = cross_attns.tolist()
            select_cross_attn_sum = cross_attns[0]
            select_cross_attn_avg = cross_attns[1]
            non_select_cross_attn_sum = cross_attns[2]
            non_select_cross_attn_avg = cross_attns[3]
        select_cross_attn_sum = select_cross_attn_sum / total_cnt
        select_cross_attn_avg = select_cross_attn_avg / total_cnt
        non_select_cross_attn_sum = non_select_cross_attn_sum / total_cnt
        non_select_cross_attn_avg = non_select_cross_attn_avg / total_cnt
        logger.info("Samples: {}, Select Cross Attention Sum: {:.4f} Each Token: {:.4e}, Not Select Cross Attention Sum: {:.4f} Each Token: {:.4e}".format(total_cnt, select_cross_attn_sum, select_cross_attn_avg, non_select_cross_attn_sum, non_select_cross_attn_avg))
        
        
    def _setup_optimizer(self, args, model, num_training_steps):
        # Build parameter groups (weight decay and non-decay).
        while isinstance(model, DDP):
            model = model.module
        # Divide params into with-weight-decay and without-weight-decay groups.
        # Layernorms and baises will have no weight decay but the rest will.
        no_decay = ["bias", "layer_norm.weight"]
        if args.freeze_extraction:
            freeze_part = ["encoder", 'qa_outputs', 'qa_classifier', 'share']
            for (n, p) in model.named_parameters():
                if any(fp in n for fp in freeze_part):
                    p.requires_grad = False
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay) and p.requires_grad)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay) and p.requires_grad)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.lr)
        num_warmup_steps = int(num_training_steps * args.warmup)
        if args.lr_scheduler == 'linear':
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif args.lr_scheduler == 'constant':
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            raise ValueError(f"No scheduler type named {args.lr_scheduler}.")
        return optimizer, lr_scheduler
    
    def _setup_wandb(self, args):
        if args.wandb_project is not None and args.rank == 0:
            wandb.init(project=args.wandb_project, name=args.wandb_name)
            wandb.config.update(args)
            # define metrics in dev process
            wandb.define_metric("dev/em", summary='max')
            wandb.define_metric("dev/loss", summary='min')
            wandb.define_metric("dev/extractive_loss", summary='min')
            wandb.define_metric("dev/generative_loss", summary='min')

    def log_step(self, log_dict=None, suffix='', wandb_suffix=None, reduce=True, **kwargs):
        new_log_dict = OrderedDict()
        for key, value in kwargs.items():
            new_log_dict[key] = value
        if log_dict is not None:
            # print(f"rank {self.args.rank} {log_dict}")
            # torch.distributed.barrier()
            for key in log_dict:
                key_tensor = torch.tensor(log_dict[key]).cuda()
                if reduce and self.world_size > 1:
                    dist.all_reduce(key_tensor, op=dist.ReduceOp.SUM)
                key_value = (key_tensor / self.world_size).mean().item()
                new_log_dict[key] = key_value
        message = '' + suffix
        for key, value in new_log_dict.items():
            if isinstance(value, float):
                message += ' {:s}: {:.5f}'.format(key, value)
            else:
                message += ' {:s}: {}'.format(key, value)
        logger.info(message)
        if self.args.rank != 0:
            return new_log_dict.get('loss', None)
        # only in main process
        if self.args.wandb_project is not None:
            wandb_log_dict = OrderedDict()
            for key, value in new_log_dict.items():
                if key in ['epoch', 'num_updates']:
                    continue
                tag = f'{wandb_suffix}/{key}' if wandb_suffix is not None else key
                wandb_log_dict[tag] = value
            global_step = kwargs.get('num_updates', None)
            wandb.log(wandb_log_dict, step=global_step)
        return new_log_dict.get('loss', None)
    
    def save_checkpoint(self, args, model, optimizer, lr_scheduler, path):
        # if not args.deepspeed:
        # Build parameter groups (weight decay and non-decay).
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        if self.rank != 0:
            return
        while isinstance(model, DDP):
            model = model.module
        state_dict = OrderedDict()
        state_dict['model'] = model.state_dict()
        if not args.no_save_optim:
            state_dict['optimizer'] = optimizer.state_dict()
            state_dict['lr_scheduler'] = lr_scheduler.state_dict()
        torch.save(state_dict, path + '.pt')
        # _zero3_consolidated_16bit_state_dict
        # else:
        #     tag = os.path.basename(path)
        #     self.wrape_reader.save_checkpoint(args.save, tag)
    
    def load_checkpoint(self, args, model, path, no_load_model=False, optimizer=None, lr_scheduler=None):
        # if not args.deepspeed:
        state_dict = torch.load(path, map_location='cpu')
        while isinstance(model, DDP):
            model = model.module
        if not no_load_model:
            model_dict = model.state_dict()
            pretrained_dict = state_dict['model']
            new_pretrainted_dict = {}
            # compatible for old checkpoint
            for k, v in pretrained_dict.items():
                if k.startswith('t5.'):
                    new_k = "model." + k[3:]
                else:
                    new_k = k
                new_pretrainted_dict[new_k] = v
            pretrained_dict = new_pretrainted_dict
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model = model.load_state_dict(pretrained_dict)
            logger.info(f"Load model checkpoint from {args.load}")
        if optimizer is not None and 'optimizer' in state_dict and not args.no_load_optim:
            optimizer.load_state_dict(state_dict['optimizer'])
            logger.info(f"Load optimizer from {args.load}")
        if lr_scheduler is not None and 'lr_scheduler' in state_dict and not args.no_load_optim:
            lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
            logger.info(f"Load lr scheduler from {args.load}")
        # else:
        #     self.wrape_reader.load_checkpoint(path, load_optimizer_states=False, load_lr_scheduler_states=False, load_module_only=True)
        #     logger.info(f"Load model checkpoint from {path}")
    
    def prepare_input(self, batch, for_generate=False):
        inputs = {}
        inputs['input_ids'] = batch['input_ids'].cuda()
        inputs['attention_mask'] = batch['attention_mask'].cuda()
        if "context_ids" in batch:
            inputs["context_ids"] = batch["context_ids"].cuda()
            inputs["context_attention_mask"] = batch["context_attention_mask"].cuda()
            inputs["prompt_mask"] = batch["prompt_mask"].cuda()
        if "only_labels" in batch:
            inputs["only_labels"] = batch["only_labels"].cuda()
            inputs["only_labels_input"] = batch["only_labels_input"].cuda()
            inputs["labels_attention_mask"] = batch["labels_attention_mask"].cuda()
        if for_generate:
            # replace input ids and attention mask with generate data
            if "generate_input_ids" in batch:
                inputs['input_ids'] = batch['generate_input_ids'].cuda()
                inputs['attention_mask'] = batch['generate_attention_mask'].cuda()
                inputs["prompt_mask"] = batch["generate_prompt_mask"].cuda()
        if self.args.with_generative_loss:
            inputs['labels'] = batch['labels'].cuda()
            if "decoder_attention_mask" in batch:
                inputs['decoder_attention_mask'] = batch['decoder_attention_mask'].cuda()
        if self.args.with_extractive_loss:
            if "local_mask" in batch:
                inputs['local_have_answer'] = batch['local_have_answer'].cuda()
                inputs['local_start_positions'] = batch['local_context_starts'].cuda()
                inputs['local_end_positions'] = batch['local_context_ends'].cuda()
                inputs['local_mask'] = batch['local_mask'].cuda()
            inputs['global_start_positions'] = batch['global_context_starts'].cuda()
            inputs['global_end_positions'] = batch['global_context_ends'].cuda()
            inputs['global_mask'] = batch['global_mask'].cuda()
        if self.args.with_rerank_loss:
            inputs['have_answers'] = batch['have_answers'].cuda()
        if Reader.return_token_type_ids:
            inputs['token_type_ids'] = batch['token_type_ids'].cuda()
        # inputs = {}
        # inputs['input_ids'] = batch['input_ids'].to('cpu')
        # inputs['attention_mask'] = batch['attention_mask'].to('cpu')
        # if self.args.with_generative_loss:
        #     inputs['labels'] = batch['labels'].to('cpu')
        #     inputs['decoder_attention_mask'] = batch['decoder_attention_mask'].to('cpu')
        # if self.args.with_extractive_loss:
        #     inputs['local_have_answer'] = batch['local_have_answer'].to('cpu')
        #     inputs['local_start_positions'] = batch['local_context_starts'].to('cpu')
        #     inputs['local_end_positions'] = batch['local_context_ends'].to('cpu')
        #     inputs['local_mask'] = batch['local_mask'].to('cpu')
        #     inputs['global_start_positions'] = batch['global_context_starts'].to('cpu')
        #     inputs['global_end_positions'] = batch['global_context_ends'].to('cpu')
        #     inputs['global_mask'] = batch['global_mask'].to('cpu')
        # if Reader.return_token_type_ids:
        #     inputs['token_type_ids'] = batch['token_type_ids'].to('cpu')
        return inputs
    
    def optimizer_step(self, optimizer, scheduler):
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

