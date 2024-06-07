# $env:MASTER_ADDR="192.168.9.104"
# $env:MASTER_ADDR="172.19.57.238" ## greeniwch 
$env:CUDA_VISIBLE_DEVICES="0,1"
# $env:CUDA_VISIBLE_DEVICES="0" # use for CPU only run
$env:CUDA_LAUNCH_BLOCKING="1"

conda activate ML

# torchrun --standalone --nnodes=1 --nproc-per-node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 train_stego.py
# torchrun --standalone --nnodes=1 --nproc-per-node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 train_stego.py --gpu_ids="0"
# torchrun --standalone --nnodes=1 --nproc-per-node=1 --rdzv-backend=c10d --rdzv-endpoint=localhost:0 train_stego.py --gpu_ids="0"
python train_stego.py --gpu_ids="0"