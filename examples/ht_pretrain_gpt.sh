#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

# CHECKPOINT_PATH=<Specify path>
VOCAB_FILE=/mnt/petrelfs/huangting.p/workspace/HT-Megatron-DeepSpeed/data/gpt2-vocab.json
MERGE_FILE=/mnt/petrelfs/huangting.p/workspace/HT-Megatron-DeepSpeed/data/gpt2-merges.txt
# DATA_PATH=<Specify path and file prefix>_text_document


# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}

head_node_ip=$(cat /etc/hosts | grep -w "$head_node" | awk '{print $1}')
echo $head_node

## distributed env config
GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID
MASTER_ADDR=$head_node_ip
MASTER_PORT=7880
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --lr 0.00015 \
    --train-iters 10 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl


# srun -p llm_s --quotatype=spot --preempt -N 1 -n 1 --ntasks-per-node=1 --gpus-per-task=8 --cpus-per-task=16 sh examples/ht_pretrain_gpt.sh
