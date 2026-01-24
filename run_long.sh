#!/bin/bash
# Long training (10x steps) for better generation quality
# Checkpoints saved to DATA_PATH/checkpoints/ every 2000 steps
torchrun --standalone --nproc_per_node=8 train_gpt_long.py
