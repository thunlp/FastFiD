#!/bin/bash
export OMP_NUM_THREADS=2

HOST_NODE_ADDR="localhost:8002"
PLM_DIR="/data1/private/huangyufei/pretrained_models"

ARGS="--task retrieval \
      --batch-size 64 \
      --context-encoder-path ${PLM_DIR} \
      --retriever-query-seq-length 128 \
      --embedding-path wiki_index.pkl \
      --faiss-use-gpu \
      --evidence-data-path datas/psgs_w100.tsv \
      --qa-file-dev datas/qas/nq-dev.csv \
      --qa-file-test datas/qas/nq-test.csv \
      --qa-file-train datas/qas/nq-train.csv \
      --embedding-size 768 \
      --topk-retrievals 100 \
      --num-workers 2 \
      --match string \
      --report-topk-accuracies 1 5 10 20 25 50 100 \
      --save-topk-outputs-path datas/qaps_fidkd"

export CUDA_VISIBLE_DEVICES=0,1,2,3
DISTRIBUTED_ARGS="-m torch.distributed.run --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=${HOST_NODE_ADDR} --max_restarts=0"


python ${DISTRIBUTED_ARGS} src/tasks/run.py ${ARGS}