#!/bin/bash
export OMP_NUM_THREADS=2
export READER_MODEL_TYPE=t5

HOST_NODE_ADDR="localhost:8001"
PLM_DIR="plm_dir_for_t5"

DATASET_NAME="nq"
if [ ${DATASET_NAME} == "nq" ]; then
    lambda=0.1
    temperature=0.5
    TOPK_RETRIEVALS=100
    batch_size=16
    eval_batch_size=32
    DEC_SEQ_LEN=30
    RUN_NAME="nq_hybrid_top${TOPK_RETRIEVALS}_linearlr_fidkd_l${lambda}_t${temperature}"
    TRAIN_FILE="datas/qaps_fidkd_text_sentence/nq-train.json"
    DEV_FILE="datas/qaps_fidkd_text_sentence/nq-dev.json"
    TEST_FILE="datas/qaps_fidkd_text_sentence/nq-test.json"
elif [ ${DATASET_NAME} == "tqa" ]; then
    lambda=0.05
    temperature=0.5
    TOPK_RETRIEVALS=100
    batch_size=16
    eval_batch_size=32
    DEC_SEQ_LEN=30
    RUN_NAME="tqa_hybrid_top${TOPK_RETRIEVALS}_linearlr_fidkd_l${lambda}_t${temperature}"
    TRAIN_FILE="datas/qaps_fidkd_text_sentence/trivia-train-human-answers.json"
    DEV_FILE="datas/qaps_fidkd_text_sentence/trivia-dev.json"
    TEST_FILE="datas/qaps_fidkd_text_sentence/trivia-test.json"
elif [ ${DATASET_NAME} == "asqa" ]; then
    lambda=0.05
    temperature=0.5
    TOPK_RETRIEVALS=100
    batch_size=16
    eval_batch_size=32
    DEC_SEQ_LEN=128
    RUN_NAME="asqa_hybrid_top${TOPK_RETRIEVALS}_linearlr_fidkd_l${lambda}_t${temperature}"
    TRAIN_FILE="datas/qaps_fidkd_text_sentence/asqa-train.json"
    DEV_FILE="datas/qaps_fidkd_text_sentence/asqa-dev.json"
    TEST_FILE="datas/qaps_fidkd_text_sentence/asqa-dev.json"
fi

ARGS="--task qa \
      --batch-size ${batch_size} \
      --eval-batch-size ${eval_batch_size} \
      --global-batch-size 64 \
      --t5-model-path ${PLM_DIR}/t5-base \
      --bf16 \
      --gradient-checkpointing \
      --encoder-seq-length 384 \
      --decoder-seq-length ${DEC_SEQ_LEN} \
      --qa-file-dev ${DEV_FILE} \
      --qa-file-train ${TRAIN_FILE} \
      --qa-file-test ${TEST_FILE} \
      --iterable-load \
      --with-title \
      --with-extractive-loss \
      --with-generative-loss \
      --extractive-loss-lambda ${lambda} \
      --extractive-loss-temperature ${temperature} \
      --num-beams 4 \
      --train \
      --test \
      --output-prediction-path checkpoints/${RUN_NAME}/results/generative_top${TOPK_RETRIEVALS}_epoch9 \
      --eval-strategy epoch \
      --inference-method generative \
      --topk-retrievals ${TOPK_RETRIEVALS} \
      --num-workers 2 \
      --start-epoch 0 \
      --train-epochs 10 \
      --save checkpoints/${RUN_NAME} \
      --save-strategy epoch \
      --log-interval 10 \
      --lr 1e-4 \
      --lr-scheduler linear \
      --warmup 0.1 \
      --wandb-project fastfid \
      --wandb-name ${RUN_NAME}"

mkdir -p checkpoints/${RUN_NAME}
mkdir -p checkpoints/${RUN_NAME}/results

export CUDA_VISIBLE_DEVICES=0,1,2,3
# DISTRIBUTED_ARGS="-m torch.distributed.run --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=${HOST_NODE_ADDR} --max_restarts=0"
# python ${DISTRIBUTED_ARGS} src/tasks/run.py ${ARGS}
deepspeed src/tasks/run.py ${ARGS} --deepspeed deepspeed_config/ds_config_zero2.json