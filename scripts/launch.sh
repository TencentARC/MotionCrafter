#!/bin/sh
# shellcheck disable=SC2230
# shellcheck disable=SC2086
set -x
# Exit script when a command returns nonzero state
set -e
# set -o pipefail


CACHE_DIR="cache"
LOG_DIR="log"
ACC_CFG=configs/accelerate_config.yaml
TASK_CFG=${1}

export NCCL_DEBUG=INFO
export XDG_CACHE=${CACHE_DIR}/xdg
export TORCH_HOME=${CACHE_DIR}/torch
export HF_HOME=${CACHE_DIR}/hf
export TOKENIZERS_PARALLELISM=false
export TORCH_DISTRIBUTED_DEBUG=INFO
export WANDB_MODE="offline"

gpu_model=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits | head -n 1)

if [[ "$gpu_model" == *"V100"* ]]; then
  echo "GPU is V100: $gpu_model"
  export NCCL_IB_DISABLE=1
  export NCCL_P2P_DISABLE=1
  export NCCL_SOCKET_IFNAME=eth1
elif [[ "$gpu_model" == *"A100"* ]]; then
  echo "GPU is A100: $gpu_model"
  export NCCL_SOCKET_IFNAME=eth1
  # export NCCL_IB_GID_INDEX=3
  # export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
  # export NCCL_IB_SL=3
  export NCCL_CHECKS_DISABLE=1
  # export NCCL_P2P_DISABLE=0
  # export NCCL_LL_THRESHOLD=16384
  # export NCCL_IB_CUDA_SUPPORT=1
elif [[ "$gpu_model" == *"H100"* ]]; then
  echo "GPU is H100: $gpu_model"
  export NCCL_IB_PCI_RELAXED_ORDERING=2
  export NCCL_IB_QPS_PER_CONNECTION=16
  export NCCL_IB_ADAPTIVE_ROUTING=1
  export NCCL_IB_TIMEOUT=22
  export NCCL_IB_RETRY_CNT=10
  export NCCL_CHECKS_DISABLE=1
  export NCCL_CHECK_POINTERS=0
  export NCCL_CROSS_NIC=2
  export NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export NCCL_SOCKET_NTHREADS=4
  export NCCL_NSOCKS_PERTHREAD=4
  export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
elif [[ "$gpu_model" == *"H20"* ]]; then
  echo "GPU is H20: $gpu_model"
  export NCCL_P2P_LEVEL=NVLINK
  export NCCL_ALGO=Ring
  export NCCL_MAX_NRINGS=8
  export NCCL_PROTO=LL128
  export NCCL_SOCKET_NTHREADS=4
  export NCCL_NSOCKS_PERTHREAD=4  
  export NCCL_IB_PCI_RELAXED_ORDERING=2
  export NCCL_IB_TIMEOUT=22
  export NCCL_IB_RETRY_CNT=10
  export NCCL_CHECKS_DISABLE=1
  export NCCL_CHECK_POINTERS=0
  export NCCL_CROSS_NIC=0
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export NCCL_SOCKET_IFNAME=eth1
fi


HOST_NUM=1
INDEX=0
CHIEF_IP=127.0.0.1
LOCAL_GPU_NUM=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits | wc -l)
NODE_NUM=$((LOCAL_GPU_NUM * HOST_NUM))

export CHIEF_IP="${CHIEF_IP:-127.0.0.1}"

start_cmd="\
  accelerate launch \
    --config_file ${ACC_CFG} \
    --num_machines ${HOST_NUM} \
    --num_processes ${NODE_NUM} \
    --machine_rank ${INDEX}  \
    --main_process_ip ${CHIEF_IP} \
  apps/train.py \
    --ginc ${TASK_CFG} \
    --ginb TrainerConfig.output_dir=\"'${LOG_DIR}'\" \
  2>&1"

echo $start_cmd
eval ${start_cmd}