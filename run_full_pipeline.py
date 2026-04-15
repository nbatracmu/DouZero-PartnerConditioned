#!/usr/bin/env python3
"""
run_full_pipeline.py — Unified training + evaluation + plotting pipeline.

This script is designed to be run headlessly on EC2 via tmux.
It handles training, evaluation, cross-play analysis, behavioral metrics,
and automatic plot generation — all in one command.

Usage:
    # Train only (the main use case for tmux on EC2)
    python run_full_pipeline.py --mode train --total_frames 50000000

    # Evaluate + generate plots (after training finishes)
    python run_full_pipeline.py --mode evaluate

    # Full pipeline: train then evaluate
    python run_full_pipeline.py --mode all --total_frames 50000000

    # Train with specific GPU
    python run_full_pipeline.py --mode train --total_frames 50000000 --gpu 0

All outputs (checkpoints, logs, plots, JSON results) are saved to disk.
Plots use matplotlib Agg backend — no display needed.
"""

import os
import sys
import json
import glob
import pickle
import argparse
import subprocess
import time
from datetime import datetime

# Use Agg backend for headless rendering (MUST be before pyplot import)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# Configuration
# ================================================================

SAVEDIR = 'douzero_checkpoints'
POOL_DIR = 'partner_pool'
OUTPUT_DIR = 'results'

# Default experiment configs for evaluation
DEFAULT_CONFIGS = {
    'baseline': f'{SAVEDIR}/douzero',
    'pr50': f'{SAVEDIR}/partner_random_50',
    'pc50': f'{SAVEDIR}/partner_conditioned_50',
}


def get_parser():
    parser = argparse.ArgumentParser(
        description='DouZero Full Pipeline: Train + Evaluate + Plot')

    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'evaluate', 'all'],
                        help='Pipeline mode: train, evaluate, or all')

    # Training args
    parser.add_argument('--total_frames', type=int, default=50000000,
                        help='Total training frames (default: 50M)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID')
    parser.add_argument('--xpid', type=str,
                        default='partner_conditioned_50',
                        help='Experiment ID for this run')
    parser.add_argument('--partner_random_prob', type=float, default=0.5,
                        help='Partner randomization probability')
    parser.add_argument('--pool_save_interval', type=int, default=500000,
                        help='Save pool snapshot every N frames')
    parser.add_argument('--save_interval', type=float, default=30,
                        help='Checkpoint save interval in minutes')
    parser.add_argument('--num_actors', type=int, default=5,
                        help='Number of actor processes per GPU')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Learner batch size')
    parser.add_argument('--load_model', action='store_true',
                        help='Resume from existing checkpoint')

    # Evaluation args
    parser.add_argument('--eval_games', type=int, default=1000,
                        help='Number of games for behavioral evaluation')
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                        help='Config names to evaluate (default: auto-detect)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Directory for output plots and results')

    return parser


# ================================================================
# TRAINING
# ================================================================

def run_training(args):
    """Launch partner-conditioned training as a subprocess."""
    print("\n" + "=" * 70)
    print(f"STARTING TRAINING: {args.xpid}")
    print(f"  Total frames: {args.total_frames:,}")
    print(f"  GPU: {args.gpu}")
    print(f"  Partner random prob: {args.partner_random_prob}")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    cmd = [
        sys.executable, 'train_partner_conditioned.py',
        '--partner_conditioned',
        '--partner_random',
        '--partner_random_prob', str(args.partner_random_prob),
        '--pool_dir', POOL_DIR,
        '--pool_save_interval', str(args.pool_save_interval),
        '--pool_sample_strategy', 'uniform',
        '--total_frames', str(args.total_frames),
        '--gpu_devices', args.gpu,
        '--training_device', '0',
        '--num_actors', str(args.num_actors),
        '--batch_size', str(args.batch_size),
        '--save_interval', str(args.save_interval),
        '--savedir', SAVEDIR,
        '--xpid', args.xpid,
    ]

    if args.load_model:
        cmd.append('--load_model')

    print(f"Command: {' '.join(cmd)}\n")

    # Run training as subprocess (prints to stdout in real-time)
    result = subprocess.run(cmd, cwd=os.getcwd())

    if result.returncode != 0:
        print(f"\n[ERROR] Training exited with code {result.returncode}")
        sys.exit(1)

    print(f"\n[DONE] Training completed at "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ================================================================
# EVALUATION
# ================================================================

def find_latest_weights(checkpoint_dir):
    """Find the most recent weight files for each position."""
    weights = {}
    for position in ['landlord', 'landlord_up', 'landlord_down']:
        pattern = os.path.join(checkpoint_dir, f'{position}_weights_*.ckpt')
        files = sorted(glob.glob(pattern))
        if files:
            weights[position] = files[-1]
        else:
            weights[position] = None
    return weights


def detect_available_configs():
    """Auto-detect which training configurations have completed."""
    available = {}
    for name, path in DEFAULT_CONFIGS.items():
        if os.path.isdir(path):
            weights = find_latest_weights(path)
            if all(v is not None for v in weights.values()):
                available[name] = path
                print(f"  [OK] {name}: {path}")
            else:
                print(f"  [PARTIAL] {name}: {path} (missing some weights)")
        else:
            print(f"  [SKIP] {name}: {path} not found")
    return available


def generate_eval_data():
    """Generate evaluation data if not present."""
    if not os.path.exists('eval_data.pkl'):
        print("Generating evaluation data...")
        subprocess.run([
            sys.executable, 'generate_eval_data.py',
            '--num_games', '10000'
        ], check=True)
        print("Evaluation data generated: eval_data.pkl")
    else:
        print("Evaluation data already exists: eval_data.pkl")


def run_cross_play(configs, output_dir):
    """Run cross-play evaluation between all available configs."""
    from douzero.evaluation.simulation import evaluate

    all_weights = {}
    for name, ckpt_dir in configs.items():
        all_weights[name] = find_latest_weights(ckpt_dir)

    # Use first config's landlord as the fixed landlord
    landlord_ckpt = all_weights[list(configs.keys())[0]]['landlord']
    print(f"\nUsing landlord from: {landlord_ckpt}")

    if not os.path.exists('eval_data.pkl'):
        print("eval_data.pkl not found. Generating...")
        generate_eval_data()

    results = {}
    config_names = list(configs.keys())

    for up_name in config_names:
        for down_name in config_names:
            up_ckpt = all_weights[up_name]['landlord_up']
            down_ckpt = all_weights[down_name]['landlord_down']

            if up_ckpt is None or down_ckpt is None:
                print(f"  SKIP Up={up_name}, Down={down_name}")
                continue

            print(f"\n--- Cross-play: Up={up_name}, Down={down_name} ---")

            try:
                wp, adp = evaluate(
                    landlord_ckpt, up_ckpt, down_ckpt,
                    'eval_data.pkl', 5
                )
                results[(up_name, down_name)] = {'wp': wp, 'adp': adp}
                print(f"  WP={wp:.4f}, ADP={adp:.4f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                results[(up_name, down_name)] = {'wp': 0, 'adp': 0}

    return results


def run_behavioral_analysis(configs, num_games=500, output_dir='results'):
    """Run behavioral metric analysis for each config (self-play)."""

    # Import here to avoid circular imports
    from douzero.dmc.models import model_dict
    from douzero.env import Env
    from douzero.env.env import get_obs
    import torch

    all_metrics = {}

    for name, ckpt_dir in configs.items():
        print(f"\n{'=' * 50}")
        print(f"Behavioral Analysis: {name}")
        print(f"{'=' * 50}")

        weights = find_latest_weights(ckpt_dir)
        if any(v is None for v in weights.values()):
            print(f"  SKIP: missing weights")
            continue

        # Load models
        agents = {}
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            model_cls = model_dict[position]
            net = model_cls()
            state_dict = torch.load(weights[position], map_location='cpu')

            # Handle partner-conditioned model (different architecture)
            if 'partner_encoder' in str(state_dict.keys()):
                from douzero.dmc.models_partner_conditioned import \
                    FarmerPartnerConditionedModel
                net = FarmerPartnerConditionedModel()

            # Load what we can (handles mismatched keys gracefully)
            model_state = net.state_dict()
            compatible_dict = {
                k: v for k, v in state_dict.items()
                if k in model_state and model_state[k].shape == v.shape
            }
            model_state.update(compatible_dict)
            net.load_state_dict(model_state)
            net.eval()
            agents[position] = net

        # Run games
        env = Env('adp')
        peasant_positions = ['landlord_up', 'landlord_down']

        counters = {
            'peasant_wins': 0,
            'peasant_turns': 0,
            'peasant_passes': 0,
            'passes_partner_control': 0,
            'plays_after_landlord': 0,
            'peasant_non_pass': 0,
            'total_cards': 0,
        }

        for game_idx in range(num_games):
            obs_dict = env.reset()
            position = obs_dict['position']
            who_controls = 'landlord'
            done = False
            prev_position = None

            while not done:
                agent = agents[position]
                z = torch.from_numpy(obs_dict['z_batch']).float()
                x = torch.from_numpy(obs_dict['x_batch']).float()

                with torch.no_grad():
                    out = agent.forward(z, x, return_value=False, flags=None)

                action_idx = int(out['action'].cpu().numpy())
                action = obs_dict['legal_actions'][action_idx]
                is_pass = len(action) == 0
                is_peasant = position in peasant_positions
                partner = ('landlord_down' if position == 'landlord_up'
                           else 'landlord_up')

                if is_peasant:
                    counters['peasant_turns'] += 1
                    if is_pass:
                        counters['peasant_passes'] += 1
                        if who_controls == partner:
                            counters['passes_partner_control'] += 1
                    else:
                        counters['peasant_non_pass'] += 1
                        counters['total_cards'] += len(action)
                        if (prev_position == 'landlord'):
                            counters['plays_after_landlord'] += 1

                if not is_pass:
                    who_controls = position

                prev_position = position
                obs_dict, reward, done, _ = env.step(action)

                if done:
                    if reward < 0:  # Landlord lost
                        counters['peasant_wins'] += 1
                else:
                    position = obs_dict['position']

            if (game_idx + 1) % 100 == 0:
                wr = counters['peasant_wins'] / (game_idx + 1)
                print(f"  Game {game_idx + 1}/{num_games}, WR: {wr:.3f}")

        # Compute metrics
        c = counters
        metrics = {
            'peasant_win_rate': c['peasant_wins'] / num_games,
            'peasant_pass_rate': (c['peasant_passes'] / c['peasant_turns']
                                 if c['peasant_turns'] > 0 else 0),
            'partner_enabling_pass_rate': (
                c['passes_partner_control'] / c['peasant_passes']
                if c['peasant_passes'] > 0 else 0),
            'landlord_block_rate': (
                c['plays_after_landlord'] / c['peasant_non_pass']
                if c['peasant_non_pass'] > 0 else 0),
            'mean_cards_per_play': (
                c['total_cards'] / c['peasant_non_pass']
                if c['peasant_non_pass'] > 0 else 0),
            'total_games': num_games,
        }
        all_metrics[name] = metrics

        print(f"\n  Results for {name}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    return all_metrics


# ================================================================
# PLOTTING
# ================================================================

def plot_training_curves(output_dir):
    """Compare training dynamics across all available conditions."""
    import pandas as pd

    configs = {}
    for name, path in DEFAULT_CONFIGS.items():
        log_path = os.path.join(path, 'logs.csv')
        if os.path.exists(log_path):
            configs[name] = log_path

    if not configs:
        print("No training logs found. Skipping training curves.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {
        'baseline': '#2196F3',
        'pr50': '#F44336',
        'pc50': '#4CAF50',
        'frozen': '#FF9800',
        'pr30': '#9C27B0',
    }

    for name, path in configs.items():
        try:
            df = pd.read_csv(path, comment='#')
        except Exception as e:
            print(f"  Error reading {path}: {e}")
            continue

        color = colors.get(name, '#666666')

        # Returns
        ax = axes[0]
        if 'mean_episode_return_landlord_up' in df.columns:
            ax.plot(df['frames'],
                    df['mean_episode_return_landlord_up'].rolling(50).mean(),
                    label=f'{name} Peasant Up', color=color, alpha=0.9)
        if 'mean_episode_return_landlord' in df.columns:
            ax.plot(df['frames'],
                    df['mean_episode_return_landlord'].rolling(50).mean(),
                    label=f'{name} Landlord', color=color,
                    linestyle='--', alpha=0.5)

        # Loss
        ax = axes[1]
        if 'loss_landlord_up' in df.columns:
            ax.plot(df['frames'],
                    df['loss_landlord_up'].rolling(50).mean(),
                    label=f'{name}', color=color, alpha=0.9)

    axes[0].set_xlabel('Training Frames')
    axes[0].set_ylabel('Mean Episode Return')
    axes[0].set_title('Training Performance by Condition')
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Training Frames')
    axes[1].set_ylabel('Loss (smoothed)')
    axes[1].set_title('Peasant Up Training Loss')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_training_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_cross_play_heatmap(results, config_names, output_dir):
    """Generate cross-play heatmap."""
    n = len(config_names)
    matrix = np.zeros((n, n))

    for i, up in enumerate(config_names):
        for j, dn in enumerate(config_names):
            if (up, dn) in results:
                matrix[i, j] = results[(up, dn)]['wp']

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.set_yticklabels(config_names)
    ax.set_xlabel('Landlord Down Source')
    ax.set_ylabel('Landlord Up Source')
    ax.set_title('Cross-Play Peasant Win Rate')

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = 'white' if val < 0.3 or val > 0.7 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    color=color, fontsize=14, fontweight='bold')

    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor='black',
                                    linewidth=2))

    plt.colorbar(im, label='Peasant Win Rate')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_cross_play_heatmap.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # Print robustness
    print("\nRobustness (std across partners):")
    for i, name in enumerate(config_names):
        row_std = np.std(matrix[i, :])
        print(f"  {name}: row std = {row_std:.4f}")


def plot_behavioral_metrics(all_metrics, output_dir):
    """Bar chart of behavioral cooperation metrics."""
    if not all_metrics:
        return

    names = list(all_metrics.keys())
    metrics_to_plot = [
        ('peasant_win_rate', 'Peasant Win Rate'),
        ('peasant_pass_rate', 'Pass Rate'),
        ('partner_enabling_pass_rate', 'Partner-Enabling\nPass %'),
        ('landlord_block_rate', 'Landlord\nBlock Rate'),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 5))
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

    for ax_idx, (key, label) in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        values = [all_metrics[name].get(key, 0) for name in names]
        bars = ax.bar(range(len(names)), values,
                      color=colors[:len(names)])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Behavioral Cooperation Metrics by Training Condition',
                 fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_behavioral_metrics.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_robustness_scatter(results, config_names, output_dir):
    """Scatter: self-play WR vs cross-play WR."""
    if not results:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

    for idx, name in enumerate(config_names):
        self_key = (name, name)
        self_wp = results.get(self_key, {}).get('wp', 0)

        cross_wps = []
        for other in config_names:
            if other != name:
                for key in [(name, other), (other, name)]:
                    if key in results:
                        cross_wps.append(results[key]['wp'])

        cross_mean = np.mean(cross_wps) if cross_wps else 0
        cross_std = np.std(cross_wps) if cross_wps else 0

        ax.scatter(self_wp, cross_mean, s=200,
                   c=colors[idx % len(colors)],
                   label=name, zorder=5, edgecolors='black')
        ax.errorbar(self_wp, cross_mean, yerr=cross_std,
                    color=colors[idx % len(colors)], capsize=5, zorder=4)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No gap')
    ax.set_xlabel('Self-Play Win Rate')
    ax.set_ylabel('Mean Cross-Play Win Rate')
    ax.set_title('Self-Play vs Cross-Play Robustness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_robustness_scatter.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ================================================================
# MAIN PIPELINE
# ================================================================

def main():
    args = get_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("DouZero Partner-Conditioned Pipeline")
    print(f"Mode: {args.mode}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ========================================
    # PHASE 1: Training
    # ========================================
    if args.mode in ('train', 'all'):
        run_training(args)

    # ========================================
    # PHASE 2: Evaluation + Plotting
    # ========================================
    if args.mode in ('evaluate', 'all'):
        print("\n" + "=" * 70)
        print("EVALUATION PHASE")
        print("=" * 70)

        # Detect available configs
        print("\nDetecting available configurations...")
        configs = detect_available_configs()

        if not configs:
            print("\nNo complete configurations found!")
            print("Make sure training has saved weight files.")
            return

        config_names = list(configs.keys())

        # Generate eval data
        generate_eval_data()

        # Cross-play evaluation
        print("\n--- Cross-Play Evaluation ---")
        cross_results = run_cross_play(configs, args.output_dir)
        if cross_results:
            # Print table
            print("\n" + "=" * 60)
            print("CROSS-PLAY RESULTS")
            print("=" * 60)
            col_label = 'Up\\Down'
            header = f"{col_label:<15}"
            for name in config_names:
                header += f"{name:<15}"
            print(header)
            for up in config_names:
                row = f"{up:<15}"
                for dn in config_names:
                    wp = cross_results.get((up, dn), {}).get('wp', 0)
                    row += f"{wp:<15.4f}"
                print(row)

        # Behavioral metrics
        print("\n--- Behavioral Metrics ---")
        behavioral_results = run_behavioral_analysis(
            configs, num_games=args.eval_games, output_dir=args.output_dir)

        # Generate plots
        print("\n--- Generating Plots ---")
        plot_training_curves(args.output_dir)

        if cross_results:
            plot_cross_play_heatmap(cross_results, config_names,
                                    args.output_dir)
            plot_robustness_scatter(cross_results, config_names,
                                    args.output_dir)

        if behavioral_results:
            plot_behavioral_metrics(behavioral_results, args.output_dir)

        # Save all results to JSON
        results_out = {
            'timestamp': datetime.now().isoformat(),
            'configs': {k: v for k, v in configs.items()},
            'cross_play': {
                str(k): v for k, v in cross_results.items()
            } if cross_results else {},
            'behavioral': behavioral_results,
        }

        results_path = os.path.join(args.output_dir,
                                     'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_out, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")

    print("\n" + "=" * 70)
    print(f"Pipeline finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
