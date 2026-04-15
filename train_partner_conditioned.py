"""
train_partner_conditioned.py — Entry point for partner-conditioned training.

Place this file in the DouZero root directory (next to train.py).

Usage:
    # Partner-conditioned + partner-randomized training (the main experiment)
    python train_partner_conditioned.py \
        --partner_conditioned \
        --partner_random \
        --partner_random_prob 0.5 \
        --pool_save_interval 500000 \
        --pool_sample_strategy uniform \
        --total_frames 50000000 \
        --savedir douzero_checkpoints \
        --xpid partner_conditioned_50

    # With GPU specification
    python train_partner_conditioned.py \
        --partner_conditioned \
        --partner_random \
        --partner_random_prob 0.5 \
        --total_frames 50000000 \
        --gpu_devices 0 \
        --savedir douzero_checkpoints \
        --xpid partner_conditioned_50

    # Resume from checkpoint
    python train_partner_conditioned.py \
        --partner_conditioned \
        --partner_random \
        --partner_random_prob 0.5 \
        --total_frames 50000000 \
        --load_model \
        --savedir douzero_checkpoints \
        --xpid partner_conditioned_50
"""

import os
from douzero.dmc.dmc_partner_conditioned import parser, train

if __name__ == '__main__':
    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    train(flags)
