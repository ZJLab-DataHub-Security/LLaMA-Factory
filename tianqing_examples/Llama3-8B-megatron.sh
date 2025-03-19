# You can observe that the number of steps for different stage is quite different. They are not magic number. They are set to those numbers simply because I esitimate the time it takes to finish the training, and 
# choose the number such that it fits my daily schedule>_<. This is for you to exactly reproduce my results. You many change the steps to other numbers if you want to.
MODEL_DIR=${MODEL_DIR:-"/nas/qianhao/models/Meta-Llama-3-8B"}
NGPUS=${NGPUS:-8}
WORLD_SIZE=${WORLD_SIZE:-1}
NUM_PROCESSES=$[${NGPUS}*$[WORLD_SIZE]]
SEQ_LEN=${SEQ_LEN:-1024}
SP_SIZE=${SP_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
ACC=${ACC:-1}
DATASETS=${DATASETS:-"alpaca_en_demo"}
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:1024' 
export WANDB_DISABLED=true
echo ${RANK}/$[WORLD_SIZE]
if [ ${MASTER_ADDR} == 'localhost' ]; then
    export MASTER_ADDR=`hostname -i`
fi
echo ${MASTER_ADDR}:${MASTER_PORT}
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --cudabacktrace=true -x true -o my_profile \
torchrun \
--nproc-per-node=${NGPUS} \
--nnodes=${WORLD_SIZE} \
--node-rank=${RANK} \
--master-addr=${MASTER_ADDR} \
--master-port=${MASTER_PORT} \
--rdzv-backend=c10d \
--rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
src/train.py \
--use_megatron \
--megatron_cfg_path megatron_conf/megatron_gpt_finetuning_config.yaml \
--model_name_or_path ${MODEL_DIR} \
--stage sft \
--do_train \
--dataset ${DATASETS} \
--cutoff_len ${SEQ_LEN} \
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size ${BATCH_SIZE} \
--gradient_accumulation_steps ${ACC} \
--template llama3 \
--preprocessing_num_workers 16 \
--dataloader_pin_memory \
--val_size 0.1 \
--dataloader_drop_last \
--output_dir ./output/megatron_llama3_8b
