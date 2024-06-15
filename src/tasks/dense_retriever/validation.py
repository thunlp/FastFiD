import os
import json
import shutil
import logging
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    SequentialSampler,
    RandomSampler,
)
from data.indexer import OpenRetreivalDataStore, detach, FaissMIPSIndex, DistributedBruteForceIndex
from data.orqa_wiki_dataset import get_open_retrieval_wiki_dataset
from data.retriever_qa_dataset import get_retriever_qa_dataset
from tasks.dense_retriever.metrics import calculate_matches
from tqdm import tqdm


logger = logging.getLogger("retriever-validation")

def build_dataloader(args, dataset, shuffle):
    world_size = args.world_size
    rank = args.rank
    batch_size = args.batch_size
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
    return dataloader

class OpenRetrievalEvaluator(object):
    def __init__(self, args):
        # assert args.query_encoder_path is not None
        # from retriever import Retriever
        from retriever import FiDKDRetriever as Retriever
        self.retriever = Retriever(args)
        self.evidence_embedder_obj = None
        self.mips_index = None
        self.embedding_size = args.embedding_size
        self.get_evidence_dataset(args)
        self.faiss_use_gpu = args.faiss_use_gpu
        # initialize faiss for fast search
        self.faiss_wrapper(args)
        self.world_size = args.world_size
    
    def get_evidence_embedding(self, args):
        # This will load the embedding from the embedding path
        self.evidence_embedder_obj = OpenRetreivalDataStore(args, load_from_path=True)
    
    def get_evidence_dataset(self, args):
        self.evidence_dataset = get_open_retrieval_wiki_dataset(args, None, None)
    
    def faiss_wrapper(self, args):
        # Initialize FAISS wrapper on rank = 0 as the evidence embeddings is distributed over all the GPUs in a node
        if args.rank == 0:
            self.get_evidence_embedding(args)
            assert self.evidence_embedder_obj is not None
            self.mips_index = FaissMIPSIndex(embed_size=self.embedding_size,
                                             embed_data=self.evidence_embedder_obj,
                                             use_gpu=self.faiss_use_gpu)

        # Wait for the FAISS index to be initialized in all the nodes
        if args.world_size > 1:
            dist.barrier()
    
    @torch.no_grad()
    def evaluate(self, args, task_name, qa_file, split):
        # load dataset
        self.eval_dataset = get_retriever_qa_dataset(args, task_name, split,
                            qa_file, self.retriever.query_tokenizer)
        self.eval_dataloader = build_dataloader(args, self.eval_dataset, shuffle=False)
        no_padding_process_num = ((len(self.eval_dataset) - 1) % args.world_size) + 1
        all_query_logits = []
        reference_list = []
        query_list = []
        ids_list = []
        long_answers_list = []
        self.retriever.query_encoder.eval()
        import time
        start_time = time.time()
        for local_index, batch in enumerate(self.eval_dataloader):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            token_type_ids = batch['token_type_ids'].cuda()
            query_logits = self.retriever.encode_question(input_ids, attention_mask, token_type_ids)
            all_query_logits.append(query_logits.detach().cpu())
            reference_list.extend(batch['answers'])
            query_list.extend(batch['question'])
            ids_list.extend(batch['idx'])
            if 'long_answers' in batch:
                long_answers_list.extend(batch['long_answers'])
            if local_index % args.log_interval == 0:
                logger.info(f"Encode {local_index * args.world_size * args.batch_size} questions.")
        logger.info(f"Encode {len(self.eval_dataset)} questions using {time.time() - start_time:.3f}s")
        all_query_logits = torch.cat(all_query_logits, 0).cuda()
        all_query_logits.contiguous()
        if self.world_size > 1:
            query_logits_list = [torch.zeros_like(all_query_logits) for _ in range(self.world_size)]
            dist.all_gather(query_logits_list, all_query_logits)
            all_query_tensor = torch.cat(query_logits_list, 0)
        else:
            all_query_tensor = all_query_logits
        if args.rank == 0:
            distance, topkindex = self.mips_index.search_mips_index(all_query_tensor,
                                                                top_k=args.topk_retrievals,
                                                                reconstruct=False)
            distance = torch.from_numpy(distance).cuda()
            topkindex = torch.LongTensor(topkindex).cuda()
        else:
            distance = torch.empty(len(all_query_tensor), args.topk_retrievals, dtype=torch.float32).cuda()
            topkindex = torch.empty(len(all_query_tensor), args.topk_retrievals, dtype=torch.int64).cuda()
        if args.world_size > 1:
            torch.distributed.broadcast(distance, src=0)
            torch.distributed.broadcast(topkindex, src=0)
            distance = list(torch.chunk(distance, args.world_size, dim=0))[args.rank]
            topkindex = list(torch.chunk(topkindex, args.world_size, dim=0))[args.rank]
            if local_index == (len(self.eval_dataloader) - 1) and args.rank >= no_padding_process_num:
                # remove the padding instance
                distance = distance[:-1,:]
                topkindex = topkindex[:-1, :]
                reference_list = reference_list[:-1]
                query_list = query_list[:-1]
                ids_list = ids_list[:-1]
                long_answers_list = long_answers_list[:-1]
        del all_query_tensor
        del all_query_logits
        print("topk shape in rank {}".format(args.rank), topkindex.shape)
        topk_sim_scores = distance #/ math.sqrt(args.hidden_size)

        top_ids_and_scores = []
        for darray, topkarray in zip(topk_sim_scores, topkindex):
            top_ids_and_scores.append((topkarray.tolist(), darray.tolist()))
        print("rank {} match {}".format(args.rank, args.match))
        if args.match != "none":
            passages = self.evidence_dataset.id2text
            match_stats = calculate_matches(passages,
                                        reference_list,
                                        top_ids_and_scores,
                                        workers_num=args.num_workers,
                                        match_type=args.match)
            doc_hits = match_stats.questions_doc_hits
            question_number = len(doc_hits)
            top_k_hits = torch.FloatTensor(match_stats.top_k_hits).cuda()
            if args.world_size > 1:
                question_number = torch.LongTensor([question_number]).cuda()

                # Accumulating and summing top-k hits scores from all the ranks
                dist.all_reduce(top_k_hits, dist.ReduceOp.SUM)
                dist.all_reduce(question_number, dist.ReduceOp.SUM)
                question_number = question_number.item()

            logger.info("{} SET RESULTS. Total {} questions".format(split, question_number))
            top_k_hits = [v / question_number for v in top_k_hits]
        else:
            top_k_hits = [0] * args.topk_retrievals
            doc_hits = []
            question_number = topk_sim_scores.shape[0]
            for i in range(question_number):
                doc_hits.append([False] * args.topk_retrievals)

        for i in args.report_topk_accuracies:
            logger.info("top-{}: {:.2f}".format(i, top_k_hits[i-1] * 100))
        
        if args.save_topk_outputs_path is not None:
            print("rank {}: saving tmp shards".format(args.rank))
            all_data = []
            for i, (q, d, r, idx) in enumerate(zip(query_list, doc_hits, reference_list, ids_list)):
                ctx_list = []
                for j in range(args.topk_retrievals):
                    row_id = top_ids_and_scores[i][0][j]
                    ctx = {"id": row_id,
                           "score": top_ids_and_scores[i][1][j],
                           "has_answer": d[j]}
                    ctx_list.append(ctx)
                item = {"idx": idx,
                        "question": q,
                        "answers": r}
                if len(long_answers_list) > 0:
                    item['long_answers'] = long_answers_list[i]
                item["ctxs"] = ctx_list
                all_data.append(item)
            temp_dir_name = os.path.join(args.save_topk_outputs_path,
                                        "_tmp_reranker")
            save_shard(args, all_data, temp_dir_name)
            del all_data
            if args.world_size > 1:
                dist.barrier()

            if args.rank == 0:
                print("Merge shards in rank 0")
                file_name = os.path.splitext(os.path.basename(qa_file))[0]
                all_data = merge_shards_and_save(args.save_topk_outputs_path, temp_dir_name, file_name)
                # make sure that every single piece of data was embedded
                assert len(all_data) == len(self.eval_dataset)
                del all_data

        if args.world_size > 1:
            dist.barrier()
        return
            

def save_shard(args, data, temp_dir_name):
    """
    Save the block data that was created this in this process
    """
    if not os.path.isdir(temp_dir_name) and args.rank == 0:
        os.makedirs(temp_dir_name, exist_ok=True)
    if args.world_size > 1:
        dist.barrier()

    outpath = os.path.join(temp_dir_name, "rank{}.json".format(args.rank))
    with open(outpath, "w") as writer:
        writer.write(json.dumps(data, indent=4) + "\n")


def merge_shards_and_save(output_dir_path, temp_dir_name, file_name):
    """Combine all the shards made using self.save_shard()"""
    shard_names = os.listdir(temp_dir_name)
    all_data = []

    for fname in os.listdir(temp_dir_name):
        shard_size = 0
        old_size = len(all_data)
        fpath = '{}/{}'.format(temp_dir_name, fname)
        with open(fpath, 'r') as f:
            data = json.load(f)
            shard_size = len(data)
            all_data.extend(data)

        assert len(all_data) == old_size + shard_size
        os.remove(fpath)
    all_data.sort(key= lambda p: p['idx'])
    # save the consolidated shards
    outpath = os.path.join(output_dir_path, "{}.json".format(file_name))

    with open(outpath, 'w') as writer:
        writer.write(json.dumps(all_data, indent=4) + "\n")
    logger.info("Finished merging {} shards for a total of {} embeds".format(
                len(shard_names), len(all_data)))

    shutil.rmtree(temp_dir_name, ignore_errors=True)

    return all_data
            

        
