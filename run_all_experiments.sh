#!/bin/bash
# ================================================================
# run_all_experiments.sh — Launch all training configurations
#
# Place in DouZero root. Each experiment can be run in a separate
# terminal, SLURM job, or screen session on PSC.
#
# Usage:
#   bash run_all_experiments.sh [experiment_name]
#   
# Examples:
#   bash run_all_experiments.sh pr50          # just partner_random_50
#   bash run_all_experiments.sh all           # all experiments
#   bash run_all_experiments.sh pr30          # just partner_random_30
# ================================================================

SAVEDIR="douzero_checkpoints"
POOL_DIR="partner_pool"
POOL_INTERVAL=500000

run_pr50() {
    echo "=== Experiment: Partner Random 50% (uniform) ==="
    python train_partner_randomized.py \
        --partner_random \
        --pool_dir ${POOL_DIR} \
        --partner_random_prob 0.5 \
        --pool_save_interval ${POOL_INTERVAL} \
        --pool_sample_strategy uniform \
        --savedir ${SAVEDIR} \
        --xpid partner_random_50
}

run_pr30() {
    echo "=== Experiment: Partner Random 30% (uniform) ==="
    python train_partner_randomized.py \
        --partner_random \
        --pool_dir ${POOL_DIR} \
        --partner_random_prob 0.3 \
        --pool_save_interval ${POOL_INTERVAL} \
        --pool_sample_strategy uniform \
        --savedir ${SAVEDIR} \
        --xpid partner_random_30
}

run_pr50_biased() {
    echo "=== Experiment: Partner Random 50% (recent biased) ==="
    python train_partner_randomized.py \
        --partner_random \
        --pool_dir ${POOL_DIR} \
        --partner_random_prob 0.5 \
        --pool_save_interval ${POOL_INTERVAL} \
        --pool_sample_strategy recent_biased \
        --savedir ${SAVEDIR} \
        --xpid partner_random_50_biased
}

run_pr50_both() {
    echo "=== Experiment: Partner Random 50% (both peasants) ==="
    python train_partner_randomized.py \
        --partner_random \
        --pool_dir ${POOL_DIR} \
        --partner_random_prob 0.5 \
        --pool_save_interval ${POOL_INTERVAL} \
        --pool_sample_strategy uniform \
        --partner_position both \
        --savedir ${SAVEDIR} \
        --xpid partner_random_50_both
}

run_pr50_up() {
    echo "=== Experiment: Partner Random 50% (randomize landlord_up only) ==="
    python train_partner_randomized.py \
        --partner_random \
        --pool_dir ${POOL_DIR} \
        --partner_random_prob 0.5 \
        --pool_save_interval ${POOL_INTERVAL} \
        --pool_sample_strategy uniform \
        --partner_position landlord_up \
        --savedir ${SAVEDIR} \
        --xpid partner_random_50_up
}

run_pc50() {
    echo "=== Experiment: Partner-Conditioned + Partner Random 50% ==="
    python train_partner_conditioned.py \
        --partner_conditioned \
        --partner_random \
        --pool_dir ${POOL_DIR} \
        --partner_random_prob 0.5 \
        --pool_save_interval ${POOL_INTERVAL} \
        --pool_sample_strategy uniform \
        --total_frames 50000000 \
        --savedir ${SAVEDIR} \
        --xpid partner_conditioned_50
}

# ================================================================
# Dispatch
# ================================================================

case "${1:-all}" in
    pr50)       run_pr50 ;;
    pr30)       run_pr30 ;;
    pr50b|biased) run_pr50_biased ;;
    pr50both|both) run_pr50_both ;;
    pr50up|up)  run_pr50_up ;;
    pc50|conditioned) run_pc50 ;;
    all)
        echo "Running all experiments sequentially."
        echo "For parallel runs, launch each separately."
        echo ""
        run_pr50
        run_pr30
        run_pr50_biased
        run_pr50_both
        run_pc50
        ;;
    *)
        echo "Unknown experiment: $1"
        echo "Options: pr50, pr30, pr50b/biased, pr50both/both, pr50up/up, pc50/conditioned, all"
        exit 1
        ;;
esac
