#!/usr/bin/env bash
# TOOLS="srun --partition=Test --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1"
LOGPRE=$1

now=$(date +"%Y%m%d_%H%M%S")
# GLOG_logtostderr=1 LD_LIBRARY_PATH=/mnt/lustre/share/cuda-9.0/lib64 $TOOLS --job-name=exp_1252 \
python train.py --config ./configs/exp_0001.yaml
