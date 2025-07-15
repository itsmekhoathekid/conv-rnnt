#!/bin/bash
#SBATCH --job-name=conv
#SBATCH -o /data/npl/Speech2Text/rna/conv-rnnt/log_train_slurm/conv_%j.out
#SBATCH --gres=gpu:1   # Yêu cầu 1 GPU bất kỳ
#SBATCH -N 1           # Số lượng node để chạy
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G


python /data/npl/Speech2Text/rna/conv-rnnt/train.py --config /data/npl/Speech2Text/rna/conv-rnnt/configs/conv_rnnt.yaml
