"""
train_partner_randomized.py — Entry point for partner-randomized training.

Place this file in the DouZero root directory (next to train.py).

Usage:
    # Basic partner-randomized training
    python train_partner_randomized.py \
        --partner_random \
        --savedir douzero_checkpoints \
        --xpid partner_random

    # Seed the pool from a previous baseline run
    python train_partner_randomized.py \
        --partner_random \
        --seed_pool_from douzero_checkpoints/douzero \
        --pool_dir partner_pool \
        --savedir douzero_checkpoints \
        --xpid partner_random

    # Control randomization probability and strategy
    python train_partner_randomized.py \
        --partner_random \
        --partner_random_prob 0.5 \
        --pool_sample_strategy recent_biased \
        --pool_save_interval 500000 \
        --savedir douzero_checkpoints \
        --xpid partner_random_biased

    # Run WITHOUT partner randomization (identical to baseline)
    python train_partner_randomized.py \
        --savedir douzero_checkpoints \
        --xpid baseline_repro
"""

import os
from douzero.dmc.dmc_partner_random import parser, train

if __name__ == '__main__':
    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    train(flags)
