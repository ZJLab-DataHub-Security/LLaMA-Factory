nsys profile -t cuda,osrt,nvtx,cublas,cudnn -w true \
--cudabacktrace=all --force-overwrite true -o test_lss \
bash test.sh