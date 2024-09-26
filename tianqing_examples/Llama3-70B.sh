# You can observe that the number of steps for different stage is quite different. They are not magic number. They are set to those numbers simply because I esitimate the time it takes to finish the training, and
# choose the number such that it fits my daily schedule>_<. This is for you to exactly reproduce my results. You many change the steps to other numbers if you want to.
MODEL_DIR=${MODEL_DIR:-"/mnt/zj-gpfs/home/qianhao/models/Meta-Llama-3-8B"}
NGPUS=${NGPUS:-6}
# NGPUS=1
WORLD_SIZE=${WORLD_SIZE:-1}
NUM_PROCESSES=$((${NGPUS} * $((WORLD_SIZE))))
# SEQ_LEN=${SEQ_LEN:-16384}
SEQ_LEN=${SEQ_LEN:-12288}
# SP_SIZE=${SP_SIZE:-1}
SP_SIZE=6
BATCH_SIZE=${BATCH_SIZE:-1}
BATCH_SIZE=1
ALGO=llama3_flash_attn
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# ALGO=lss_transformer
# COMPARED_ALGO=data_parallel
# COMPARED_ALGO=dist_flash_attn
# COMPARED_ALGO=zigzag_ring_attn
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024'
export WANDB_DISABLED=true
export NCCL_DEBUG=WARN
# export CUDA_VISIBLE_DEVICES=2
RANK=0
echo ${RANK}/$((WORLD_SIZE))
MASTER_ADDR=localhost
MASTER_PORT=29500
if [ ${MASTER_ADDR} = 'localhost' ]; then
    export MASTER_ADDR=$(hostname -i)
fi
echo "MASTER_IP:MASTER_PORT"
echo ${MASTER_ADDR}:${MASTER_PORT}
echo "NUM_PROCESSER is ${NUM_PROCESSES}"
echo "NGPUS is ${NGPUS}"
echo "WORLD_SIZE is ${WORLD_SIZE}"
echo "SEQ_LEN is ${SEQ_LEN}"
echo "SP_SIZE is ${SP_SIZE}"
echo "BATCH_SIZE is ${BATCH_SIZE}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ ! -z ${ALGO+x} ]; then
    accelerate launch --config_file examples/accelerate/ds_multi_nodes.yaml \
        --use_deepspeed \
        --num_machines ${WORLD_SIZE} \
        --num_processes ${NUM_PROCESSES} \
        --main_process_ip ${MASTER_ADDR} \
        --main_process_port ${MASTER_PORT} \
        --machine_rank ${RANK} \
        --rdzv_conf "rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT,rdzv_backend=c10d" \
        src/train.py \
        --model_name_or_path ${MODEL_DIR} \
        --stage sft \
        --do_train \
        --finetuning_type full \
        --parallel_mode ${ALGO} \
        --sp_size ${SP_SIZE} \
        --deepspeed examples/deepspeed/ds_z3_offload_config.json \
        --dataset long_sft_32k \
        --template llama3 \
        --cutoff_len ${SEQ_LEN} \
        --max_steps 10 \
        --overwrite_cache \
        --preprocessing_num_workers 16 \
        --output_dir ./output/7B_4K_bs_2_lr_2e-5_${ALGO}_${TIMESTAMP} \
        --logging_steps 1 \
        --save_steps 500 \
        --plot_loss \
        --overwrite_output_dir \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps 1 \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --bf16 \
        --ddp_timeout 180000000 \
        --val_size 0.1 \
        --eval_strategy steps \
        --eval_steps 1000
fi

if [ ! -z ${COMPARED_ALGO+x} ]; then
    accelerate launch \
        --config_file examples/accelerate/ds_multi_nodes.yaml \
        --use_deepspeed \
        --num_machines ${WORLD_SIZE} \
        --num_processes ${NUM_PROCESSES} \
        --main_process_ip ${MASTER_ADDR} \
        --main_process_port ${MASTER_PORT} \
        --machine_rank ${RANK} \
        --rdzv_conf "rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT,rdzv_backend=c10d" \
        src/train.py \
        --model_name_or_path ${MODEL_DIR} \
        --stage sft \
        --do_train \
        --finetuning_type full \
        --parallel_mode ${COMPARED_ALGO} \
        --sp_size ${SP_SIZE} \
        --deepspeed examples/deepspeed/ds_z3_offload_config.json \
        --dataset long_sft_32k \
        --template llama3 \
        --cutoff_len ${SEQ_LEN} \
        --max_steps 10 \
        --overwrite_cache \
        --preprocessing_num_workers 16 \
        --output_dir ./output/7B_4K_bs_2_lr_2e-5_${COMPARED_ALGO}_${TIMESTAMP} \
        --logging_steps 1 \
        --save_steps 500 \
        --plot_loss \
        --overwrite_output_dir \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps 1 \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --bf16 \
        --ddp_timeout 180000000 \
        --val_size 0.1 \
        --eval_strategy steps \
        --eval_steps 1000
fi
