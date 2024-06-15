#!/bin/bash
export OMP_NUM_THREADS=2
export READER_MODEL_TYPE=t5

HOST_NODE_ADDR="localhost:8003"
PLM_DIR="plm_dir_for_t5"

TOPK_RETRIEVALS=100
lambda=0.1
temperature=0.5
LR_SCHEDULER="constant"

DATASET_NAME="nq"
if [ ${DATASET_NAME} == "nq" ]; then
    EPOCH=7
    batch_size=4
    eval_batch_size=4
    DEC_SEQ_LEN=30
    SELECT=200
    RUN_NAME="select_top${TOPK_RETRIEVALS}_linearlr_fidkd_l${lambda}_t${temperature}_npl"
    TRAIN_FILE="datas/qaps_fidkd_text_sentence/nq-train.json"
    DEV_FILE="datas/qaps_fidkd_text_sentence/nq-dev.json"
    TEST_FILE="datas/qaps_fidkd_text_sentence/nq-test.json"
elif [ ${DATASET_NAME} == "tqa" ]; then
    EPOCH=4
    lambda=0.05
    temperature=0.5
    batch_size=16
    eval_batch_size=32
    DEC_SEQ_LEN=30
    SELECT=400
    RUN_NAME="tqa_hybrid_top${TOPK_RETRIEVALS}_linearlr_fidkd_l${lambda}_t${temperature}_npl"
    TRAIN_FILE="datas/qaps_fidkd_text_sentence/trivia-train-human-answers.json"
    DEV_FILE="datas/qaps_fidkd_text_sentence/trivia-dev.json"
    TEST_FILE="datas/qaps_fidkd_text_sentence/trivia-test.json"
elif [ ${DATASET_NAME} == "asqa" ]; then
    EPOCH=18
    lambda=0.05
    temperature=0.5
    batch_size=16
    eval_batch_size=32
    DEC_SEQ_LEN=128
    SELECT=200
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
      --qa-file-test ${TEST_FILE} \
      --qa-file-train ${TRAIN_FILE} \
      --iterable-load \
      --with-title \
      --with-extractive-loss \
      --with-generative-loss \
      --extractive-loss-lambda ${lambda} \
      --extractive-loss-temperature ${temperature} \
      --inference-method select_generative \
      --support-sentence-length 80 \
      --support-sentence-topk-accuracies ${SELECT} \
      --num-beams 4 \
      --time-analysis \
      --train \
      --test \
      --eval-strategy epoch \
      --eval-step 200 \
      --topk-retrievals ${TOPK_RETRIEVALS} \
      --num-workers 2 \
      --start-epoch 0 \
      --train-epochs 10 \
      --save-strategy epoch \
      --save-step 200 \
      --train \
      --no-load-optim \
      --log-interval 10 \
      --lr 5e-5 \
      --lr-scheduler ${LR_SCHEDULER} \
      --warmup 0.1 \
      --wandb-project fastfid \
      --wandb-name ${RUN_NAME}-s${SELECT} \
      --save checkpoints/${RUN_NAME}/sentence${SELECT}_e${EPOCH}_checkpoints \
      --load checkpoints/${RUN_NAME}/epoch${EPOCH}.pt"
# export CUDA_VISIBLE_DEVICES="3,4,5,6"
# DISTRIBUTED_ARGS="-m torch.distributed.run --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=${HOST_NODE_ADDR} --max_restarts=0"

deepspeed --master_port 8001 src/tasks/run.py ${ARGS} --deepspeed deepspeed_config/ds_config_zero2.json