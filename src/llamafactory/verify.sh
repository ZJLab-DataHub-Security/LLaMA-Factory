MASTER_ADDR=localhost
MASTER_PORT=29500
RANK=0
accelerate launch --main_process_ip ${MASTER_ADDR} \
        --main_process_port ${MASTER_PORT} \
        --rdzv_conf "rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT,rdzv_backend=c10d" \
        --num_processes=2 \
        --num_machines=1 \
        --mixed_precision=no \
        --dynamo_backend=no \
        --machine_rank ${RANK} \
        Verify_sp_algorithm.py 