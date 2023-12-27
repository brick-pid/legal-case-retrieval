output_dir="/home/ljb/reorg_lcr/output"
PYDEVD_DISABLE_FILE_VALIDATION=1
CUDA_VISIBLE_DEVICES=1

accelerate launch run.py \
            --output_dir=$output_dir \

