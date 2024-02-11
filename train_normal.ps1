# $env:CUDA_VISIBLE_DEVICES="0,1"
# # $env:CUDA_VISIBLE_DEVICES="0"
# python -m torch.distributed.launch --nproc_per_node=1 --master_addr=localhost train_stego.py

# [W socket.cpp:663] [c10d] The client socket has failed to connect to [LAPTOP-DNEIIFFR]:29500 (system error: 10049 - The requested address is not valid in its context.).

# https://github.com/open-mmlab/mmcv/issues/1969

## new

# export MASTER_ADDR="192.168.9.104"
### add external drive for dataset download.
# greenwich libary eduroam :193.60.79.174 
# $env:MASTER_ADDR="192.168.9.104"
# $env:MASTER_ADDR="172.19.57.238" ## greeniwch 
# $env:CUDA_VISIBLE_DEVICES="0,1"
# $env:CUDA_VISIBLE_DEVICES="0"
$env:CUDA_VISIBLE_DEVICES="" # use for CPU only run
$env:CUDA_LAUNCH_BLOCKING="1"

# torchrun --standalone --nnodes=1 --nproc-per-node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 train_stego.py
# torchrun --standalone --nnodes=1 --nproc-per-node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 train_stego.py --gpu_ids="0"
torchrun --standalone --nnodes=1 --nproc-per-node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 train_stego_CPU_ONLY.py --gpu_ids="0"