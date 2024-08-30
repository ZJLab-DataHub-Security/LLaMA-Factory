export NCCL_DEBUG=WARN
# export CUDA_LAUNCH_BLOCKING=1
torchrun --nproc_per_node=4 --master_port 29501 test/test_lss_transformer_func.py
