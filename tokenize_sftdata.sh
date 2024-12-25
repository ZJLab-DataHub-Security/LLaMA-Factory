# You can observe that the number of steps for different stage is quite different. They are not magic number. They are set to those numbers simply because I esitimate the time it takes to finish the training, and 
# choose the number such that it fits my daily schedule>_<. This is for you to exactly reproduce my results. You many change the steps to other numbers if you want to.


python scripts/build_tokenized_data.py \
    --model_name_or_path ${MODEL_DIR:-"/mnt/zj-gpfs/home/lsy/models/Meta-Llama-3-8B"} \
    --stage sft \
    --do_train \
    --dataset ${DATASET:-long_sft_32k} \
    --template ${TEMPLATE:-qwen} \
    --cutoff_len ${SEQ_LEN:-81920} \
    --max_samples 20000000 \
    --preprocessing_num_workers 16 \
    --tokenized_path ${TOKENIZED_PATH} \
    --output_dir ${UNNECESSARY:-"/mnt/zj-gpfs/home/lsy"} \
