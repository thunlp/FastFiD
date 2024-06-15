import os
import sys
import torch
import torch.distributed as dist
import logging
import datetime
import time
import torch
import deepspeed
try:
    import bmtrain as bmt
except ModuleNotFoundError:
    bmt = None

def distributed_init(args):
    if args.data_parallel_size > 1:
        if args.deepspeed is not None:
            deepspeed.init_distributed()
        else:
            # create default process group
            dist.init_process_group(args.distributed_backend,
                                    rank=args.rank,
                                    world_size=args.data_parallel_size,
                                    timeout=datetime.timedelta(seconds=7200))
            torch.cuda.set_device(args.local_rank)

def logger_config(args):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper() if args.rank == 0 else "WARNING",
        stream=sys.stdout,
    )

def show_batch_sample(first_batch, tokenizer):
    # DEBUG Function
    for key in first_batch:
        if isinstance(first_batch[key], torch.Tensor):
            print(key, first_batch[key].shape)
        else:
            print(key, type(key))
    print(tokenizer.decode(first_batch['input_ids'][0, 0, :]))
    print("answers:", first_batch['answers'][0])
    print("local_mask:", first_batch['local_mask'][0])
    print("local answers:")
    for k in range(25):
        input_ids = first_batch['input_ids'][0, k, :]
        local_context_starts = first_batch['local_context_starts'][0, k, :].tolist()
        local_context_ends = first_batch['local_context_ends'][0,k,:].tolist()
        local_mask = first_batch['local_mask'][0, k, :].tolist()
        for start, end, mask in zip(local_context_starts, local_context_ends, local_mask):
            if mask > 0:
                print(k, start, end)
                print(tokenizer.decode(input_ids[start:end+1]))
    print("global answers:")
    global_context_starts = first_batch['global_context_starts'][0].tolist()
    global_context_ends = first_batch['global_context_ends'][0].tolist()
    global_mask = first_batch['global_mask'][0].tolist()
    input_ids = first_batch['input_ids'].view(first_batch['input_ids'].shape[0], -1)
    print(input_ids.shape)
    for start, end, mask in zip(global_context_starts, global_context_ends, global_mask):
        if mask > 0:
            print(start, end)
            print(tokenizer.decode(input_ids[0, start:end+1]))
            print(tokenizer.convert_ids_to_tokens(input_ids[0, start:end+1]))

class LossTracker(object):
    def __init__(self):
        self.loss_names = []
        self.loss = {}
        self.step = {}
        self.reset()

    def track(self, **kwargs):
        for name in kwargs:
            if kwargs[name] is not None:
                if name not in self.loss_names:
                    self.loss_names.append(name)
                    self.loss[name] = 0
                    self.step[name] = 0
                self.step[name] += 1
                if isinstance(kwargs[name], torch.Tensor):
                    kwargs[name] = kwargs[name].item()
                self.loss[name] += kwargs[name]
    
    def reset(self):
        for name in self.loss_names:
            self.loss[name] = 0
        for name in self.loss_names:
            self.step[name] = 0
        self.loss_names = []
    
    def all_reduce(self):
        loss_list = []
        step_list = []
        for name in self.loss_names:
            loss_list.append(self.loss[name])
            step_list.append(self.step[name])
        loss_tensor = torch.tensor(loss_list, dtype=torch.float64).cuda()
        step_tensor = torch.tensor(step_list, dtype=torch.int64).cuda()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(step_tensor, op=torch.distributed.ReduceOp.SUM)
        elif bmt is not None and bmt.world_size() > 1:
            loss_tensor = bmt.distributed.all_reduce(loss_tensor, 'sum')
            step_tensor = bmt.distributed.all_reduce(step_tensor, 'sum')
        loss_list = loss_tensor.tolist()
        step_list = step_tensor.tolist()
        for index, name in enumerate(self.loss_names):
            self.step[name] = step_list[index]
            self.loss[name] = loss_list[index]
    
    def get_loss_dict(self):
        loss_dict = {}
        for name in self.loss_names:
            if self.step[name] > 0:
                loss_dict[name] = self.loss[name] / self.step[name]
        return loss_dict
