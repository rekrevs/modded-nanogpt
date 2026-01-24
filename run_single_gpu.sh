#!/bin/bash
# Single-GPU test runner for ICE cluster
# Reduced batch sizes and iterations for quick testing

export DATA_PATH="${DATA_PATH:-.}"

# Single GPU with torchrun
torchrun --standalone --nproc_per_node=1 train_gpt_single.py
