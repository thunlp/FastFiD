import logging
import torch
import torch.distributed as dist
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    SequentialSampler,
)
from retriever import DPRRetriever, FiDKDRetriever
from data.orqa_wiki_dataset import get_open_retrieval_wiki_dataset
from data.indexer import OpenRetreivalDataStore, detach


logger = logging.getLogger("IndexBuilder")

def get_one_epoch_dataloader(args, dataset):
    world_size = args.world_size
    rank = args.rank
    batch_size = args.indexer_batch_size
    num_workers = args.num_workers
    if world_size > 1:
        sampler = DistributedSampler(dataset, rank=rank, shuffle=False)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            sampler=sampler,
                            pin_memory=True)

    return dataloader

class IndexBuilder(object):
    def __init__(self, args):
        # DPR Retriever
        # self.retriever = Retriever(args)
        # FiDKD Retriever
        self.retriever = FiDKDRetriever(args)
        self.dataset = get_open_retrieval_wiki_dataset(args, self.retriever.context_tokenizer, text_format=self.retriever.passage_format)
        self.dataloader = get_one_epoch_dataloader(args, self.dataset)
        self.log_interval = args.indexer_log_interval
        self.batch_size = args.indexer_batch_size
        self.is_main_builder = (args.rank == 0)
        self.iteration = 0
        self.total_processed = 0
        self.num_total_builders = args.world_size
        self.rank = args.rank
        self.evidence_embedder_obj = OpenRetreivalDataStore(args, load_from_path=False)
        self.no_padding_process_num = ((len(self.dataset) - 1) % self.num_total_builders + 1)
    
    def track_and_report_progress(self, global_batch_size):
        self.iteration += 1
        self.total_processed += global_batch_size
        if self.is_main_builder and self.iteration % self.log_interval == 0:
            logger.info('Batch {:10d} | Total {:10d}'.format(self.iteration, self.total_processed))
    
    @torch.no_grad()
    def build_and_save_index(self):
        self.retriever.context_encoder.eval()
        for local_index, batch in enumerate(self.dataloader):
            input_ids = batch['input_ids'].cuda().squeeze(1)
            attention_mask = batch['attention_mask'].cuda().squeeze(1)
            token_type_ids = batch['token_type_ids'].cuda().squeeze(1)
            row_id = batch['row_id']
            context_logits = self.retriever.embeded_text(input_ids, attention_mask, token_type_ids)
            context_logits = detach(context_logits)
            row_id = row_id.tolist()
            if local_index == (len(self.dataloader) - 1):
                # last batch
                if self.rank >= self.no_padding_process_num:
                    # remove last instance
                    row_id = row_id[:-1]
                    context_logits = context_logits[:-1,:]
                global_batch_size = len(row_id) * self.num_total_builders - (self.num_total_builders - self.no_padding_process_num)
            else:
                global_batch_size = len(row_id) * self.num_total_builders
            self.evidence_embedder_obj.add_block_data(row_id, context_logits, allow_overwrite=False)
            if self.num_total_builders > 1:
                dist.barrier()
            self.track_and_report_progress(global_batch_size=global_batch_size)
        # This process signals to finalize its shard and then synchronize with the other processes
        self.evidence_embedder_obj.save_shard()
        if self.num_total_builders > 1:
            dist.barrier()
        # del self.model

        # rank 0 process builds the final copy
        if self.is_main_builder:
            self.evidence_embedder_obj.merge_shards_and_save()
            # make sure that every single piece of data was embedded
            assert len(self.evidence_embedder_obj.embed_data) == len(self.dataset)
        self.evidence_embedder_obj.clear()

        # complete building the final copy
        # dist.barrier()
            

