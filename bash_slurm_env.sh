#!/bin/bash

# Get the current hostname
hostname=$(hostname)

if [[ "$hostname" == "ecbab01e-"* ]]; then
    echo "On ecbab01e cluster"
    slurm_launcher=slurm_launcher_ecbab01e.py
    cpus_per_gpu=8

    export NCCL_ALGO=RING
    export NCCL_IB_AR_THRESHOLD=0
    export NCCL_IB_PCI_RELAXED_ORDERING=1
    export NCCL_IB_SPLIT_DATA_ON_QPS=0
    export NCCL_IB_QPS_PER_CONNECTION=2
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_11:1
    export NCCL_SOCKET_IFNAME=enp217s0f0np0
    export NCCL_IGNORE_CPU_AFFINITY=1
elif [[ "$hostname" == "pika-cpu-ash-ad2-"* || "$hostname" == "pika-h100-ash-ad2-"* ]]; then
    echo "On ash_ad2 cluster"
    slurm_launcher=slurm_launcher_ash_ad2.py
    cpus_per_gpu=14

    export NCCL_CROSS_NIC=0
    export NCCL_CUMEM_ENABLE=0
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_IB_SPLIT_DATA_ON_QPS=0
    export NCCL_IB_QPS_PER_CONNECTION=16
    export NCCL_IB_SL=0
    export NCCL_IB_TC=41
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_TIMEOUT=22
    export RX_QUEUE_LEN=8192
    export IB_RX_QUEUE_LEN=8192
    export UCX_TLS=tcp
    export HCOLL_ENABLE_MCAST_ALL=0
    export coll_hcoll_enable=0
    export UCX_NET_DEVICES=eth0
    export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17
    export NCCL_ALGO=Auto
    export NCCL_IGNORE_CPU_AFFINITY=1
    export NCCL_TOPO_FILE=/opt/nccl_files/H100-topology.xml
elif [[ "$hostname" == "denvrbm-"* ]]; then
    echo "On denvrbm cluster"
    slurm_launcher=slurm_launcher.py
    cpus_per_gpu=26

    export NCCL_SOCKET_IFNAME=bond0
    export NCCL_IB_HCA=mlx5
    export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
    export MELLANOX_VISIBLE_DEVICES=all
    export NCCL_IB_TIMEOUT=22
else
    echo "Error: failed to identify cluster name."
    exit 1
fi

export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO

if [ $2 -lt 8 ]
then
    NUM_NODES=1
    GRES=$2
    CPUS=$(( $2 * $cpus_per_gpu ))
else
    if [ $(( $2 % 8 )) -eq 0 ]
    then
        NUM_NODES=$(( $2 / 8 ))
        GRES=8
        CPUS=$(( $cpus_per_gpu * 8 ))
    else
        echo "Error: number of GPUs not divisible by 8."
        exit 1
    fi
fi
echo "Launching with ${NUM_NODES} node(s), ${GRES} GPU(s) and ${CPUS} CPUs per node"

timestamp=$(date +%Y-%m-%dT%H-%M-%S%Z)
