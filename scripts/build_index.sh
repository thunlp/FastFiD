#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=2

HOST_NODE_ADDR="localhost:8006"
# We use retriever from the work of Gautier Izacard and Edouard Grave (Distilling Knowledge from Reader to Retriever for Question Answering). You can download their retriever according to https://github.com/facebookresearch/FiD/blob/main/get-model.sh
PLM_DIR="path_to_retriever_model"

INDEX_ARGS="--context-encoder-path ${PLM_DIR} \
            --indexer-batch-size 256 \
            --indexer-log-interval 100 \
            --retriever-seq-length 256 \
            --evidence-data-path data/psgs_w100.tsv \
            --embedding-path wiki_index.pkl"
            # --evidence-number 100009"

DISTRIBUTED_ARGS="-m torch.distributed.run --nnodes=1 --nproc_per_node=4 --rdzv_endpoint=${HOST_NODE_ADDR} --max_restarts=0"


python ${DISTRIBUTED_ARGS} src/create_doc_index.py ${INDEX_ARGS}