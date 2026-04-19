#!/usr/bin/env python3
"""
plot_pc50_results.py — Generate training analysis plots from pc50 logs.

Reads the training CSV log directly (handles the '# ' header prefix)
and generates meaningful, publication-quality figures showing the
partner-conditioned model's learning trajectory.

Usage:
    python plot_pc50_results.py [--log_path PATH] [--output_dir DIR]

Designed to run headlessly (matplotlib Agg backend).
"""

import os
import sys
import json
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def parse_log_csv(path):
    """Parse the training log CSV, handling the '# ' header prefix."""
    with open(path, 'r') as f:
        lines = f.readlines()

    # Fix header: remove leading '# ' if present
    if lines[0].startswith('# '):
        lines[0] = lines[0][2:]

    reader = csv.DictReader(lines)
    data = {key: [] for key in reader.fieldnames}
    for row in reader:
        for key in reader.fieldnames:
            try:
                data[key].append(float(row[key]))
            except (ValueError, TypeError):
                data[key].append(0.0)

    # Convert to numpy
    for key in data:
        data[key] = np.array(data[key])
    return data


def smooth(arr, window=200):
    """Simple moving average smoothing."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


def plot_training_curves(data, output_dir):
    """
    Figure 1: Training curves — 3 subplots showing
    (a) Mean episode returns for all 3 positions
    (b) Training loss for all 3 positions
    (c) Zoomed-in final convergence
    """
    frames = data['frames']
    window = 200

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    colors = {
        'landlord': '#E53935',
        'landlord_up': '#1E88E5',
        'landlord_down': '#43A047',
    }
    labels = {
        'landlord': 'Landlord',
        'landlord_up': 'Peasant Up (PC)',
        'landlord_down': 'Peasant Down',
    }

    # --- (a) Mean Episode Returns ---
    ax = axes[0]
    for pos in ['landlord', 'landlord_up', 'landlord_down']:
        col = f'mean_episode_return_{pos}'
        if col in data:
            y = smooth(data[col], window)
            x = frames[:len(y)]
            ax.plot(x, y, color=colors[pos], label=labels[pos],
                    linewidth=1.5, alpha=0.9)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_xlabel('Training Frames', fontsize=11)
    ax.set_ylabel('Mean Episode Return', fontsize=11)
    ax.set_title('(a) Training Returns', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.0f}M'))

    # --- (b) Training Loss ---
    ax = axes[1]
    for pos in ['landlord', 'landlord_up', 'landlord_down']:
        col = f'loss_{pos}'
        if col in data:
            y = smooth(data[col], window)
            x = frames[:len(y)]
            ax.plot(x, y, color=colors[pos], label=labels[pos],
                    linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Training Frames', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('(b) Training Loss', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.0f}M'))

    # --- (c) Final Convergence (last 25%) ---
    ax = axes[2]
    cutoff = int(len(frames) * 0.75)
    for pos in ['landlord_up', 'landlord_down']:
        col = f'mean_episode_return_{pos}'
        if col in data:
            y_full = data[col][cutoff:]
            y = smooth(y_full, min(100, len(y_full) // 4 + 1))
            x = frames[cutoff:cutoff + len(y)]
            ax.plot(x, y, color=colors[pos], label=labels[pos],
                    linewidth=2, alpha=0.9)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_xlabel('Training Frames', fontsize=11)
    ax.set_ylabel('Mean Episode Return', fontsize=11)
    ax.set_title('(c) Final Convergence (Peasants)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.0f}M'))

    plt.suptitle('Partner-Conditioned Training (pc50): 50M Frames',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_training_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_return_trajectory(data, output_dir):
    """
    Figure 2: Detailed trajectory of peasant returns,
    showing the transition from losing (-1.35) to winning (+0.08).
    Also shows the landlord-peasant return gap over time.
    """
    frames = data['frames']
    window = 300

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- (a) Full trajectory with phase annotations ---
    ax = axes[0]

    ret_up = smooth(data['mean_episode_return_landlord_up'], window)
    ret_dn = smooth(data['mean_episode_return_landlord_down'], window)
    ret_ll = smooth(data['mean_episode_return_landlord'], window)
    x = frames[:len(ret_up)]

    ax.fill_between(x, ret_up, ret_dn, alpha=0.15, color='#1E88E5',
                    label='Peasant spread')
    ax.plot(x, ret_up, color='#1E88E5', linewidth=2,
            label='Peasant Up (PC)', alpha=0.9)
    ax.plot(x, ret_dn, color='#43A047', linewidth=2,
            label='Peasant Down', alpha=0.9)
    ax.plot(x, ret_ll, color='#E53935', linewidth=1.5,
            label='Landlord', alpha=0.7, linestyle='--')

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)

    # Phase annotations
    ax.axvspan(0, 2e6, alpha=0.08, color='red', label='Exploration phase')
    cross_idx = np.where(ret_up > 0)[0]
    if len(cross_idx) > 0:
        cross_frame = x[cross_idx[0]]
        ax.axvline(x=cross_frame, color='green', linestyle=':', alpha=0.6)
        ax.annotate(f'Zero crossing\n({cross_frame/1e6:.1f}M)',
                    xy=(cross_frame, 0), fontsize=8,
                    xytext=(cross_frame + 2e6, 0.3),
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))

    ax.set_xlabel('Training Frames', fontsize=11)
    ax.set_ylabel('Mean Episode Return', fontsize=11)
    ax.set_title('(a) Learning Trajectory', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.0f}M'))

    # --- (b) Peasant cooperation gap ---
    ax = axes[1]
    min_len = min(len(ret_up), len(ret_dn))
    gap = np.abs(ret_up[:min_len] - ret_dn[:min_len])
    x_gap = frames[:min_len]

    ax.plot(x_gap, gap, color='#7B1FA2', linewidth=1.5, alpha=0.9)
    ax.fill_between(x_gap, 0, gap, alpha=0.2, color='#7B1FA2')
    ax.set_xlabel('Training Frames', fontsize=11)
    ax.set_ylabel('|Return_Up - Return_Down|', fontsize=11)
    ax.set_title('(b) Peasant Asymmetry Over Training',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.0f}M'))

    # Add text with final values
    final_gap = gap[-1] if len(gap) > 0 else 0
    ax.text(0.95, 0.95, f'Final gap: {final_gap:.4f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender',
                      alpha=0.8))

    plt.suptitle('Partner-Conditioned Policy: Return Analysis',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_return_trajectory.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_behavioral_comparison(output_dir):
    """
    Figure 3: Behavioral metrics bar chart.
    Reads from evaluation_results.json if available.
    Shows cooperation metrics with reference lines.
    """
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    if not os.path.exists(results_path):
        print(f'No evaluation_results.json found at {results_path}, skipping.')
        return

    with open(results_path) as f:
        results = json.load(f)

    behavioral = results.get('behavioral', {})
    if not behavioral:
        print('No behavioral data found, skipping.')
        return

    metrics = {
        'Peasant\nWin Rate': ('peasant_win_rate', 0.5, 'Win rate > 50% means peasants beat landlord'),
        'Pass\nRate': ('peasant_pass_rate', None, 'Fraction of turns where peasant passes'),
        'Partner-Enabling\nPass %': ('partner_enabling_pass_rate', 0.5, 'Higher = more cooperative passing'),
        'Landlord\nBlock Rate': ('landlord_block_rate', 0.5, 'How often peasants respond to landlord plays'),
        'Cards per\nPlay': ('mean_cards_per_play', None, 'Average number of cards played per non-pass'),
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
    bar_colors = ['#1E88E5', '#43A047', '#E53935', '#FF9800', '#9C27B0']

    for idx, (name, vals) in enumerate(behavioral.items()):
        for ax_idx, (label, (key, ref, desc)) in enumerate(metrics.items()):
            ax = axes[ax_idx]
            val = vals.get(key, 0)

            if idx == 0:  # Only plot once per metric
                bar = ax.bar([name], [val], color=bar_colors[ax_idx],
                             width=0.5, edgecolor='white', linewidth=1.5)
                ax.text(0, val + 0.015, f'{val:.3f}',
                        ha='center', va='bottom', fontsize=11,
                        fontweight='bold')

                if ref is not None:
                    ax.axhline(y=ref, color='gray', linestyle='--',
                               alpha=0.5, linewidth=1)
                    ax.text(0.5, ref + 0.01, f'Baseline: {ref}',
                            ha='center', fontsize=7, color='gray',
                            transform=ax.get_yaxis_transform())

                ax.set_title(label, fontsize=11, fontweight='bold')
                ax.set_ylim(0, max(val * 1.3, 0.6))
                ax.grid(True, alpha=0.2, axis='y')
                ax.tick_params(axis='x', labelsize=10)

    plt.suptitle('Behavioral Cooperation Metrics: pc50 (Partner-Conditioned)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_behavioral_metrics.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def plot_loss_analysis(data, output_dir):
    """
    Figure 4: Loss analysis showing convergence quality.
    """
    frames = data['frames']
    window = 200

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = {
        'landlord': '#E53935',
        'landlord_up': '#1E88E5',
        'landlord_down': '#43A047',
    }
    labels = {
        'landlord': 'Landlord',
        'landlord_up': 'Peasant Up (PC)',
        'landlord_down': 'Peasant Down',
    }

    # --- (a) Raw loss comparison ---
    ax = axes[0]
    for pos in ['landlord', 'landlord_up', 'landlord_down']:
        col = f'loss_{pos}'
        if col in data:
            y = smooth(data[col], window)
            x = frames[:len(y)]
            ax.plot(x, y, color=colors[pos], label=labels[pos],
                    linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Training Frames', fontsize=11)
    ax.set_ylabel('Loss (smoothed)', fontsize=11)
    ax.set_title('(a) Training Loss Over Time', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.0f}M'))

    # --- (b) Loss ratio (Up/Down) ---
    ax = axes[1]
    loss_up = smooth(data.get('loss_landlord_up', np.zeros(1)), window)
    loss_dn = smooth(data.get('loss_landlord_down', np.zeros(1)), window)
    min_len = min(len(loss_up), len(loss_dn))
    if min_len > 0:
        ratio = loss_up[:min_len] / (loss_dn[:min_len] + 1e-8)
        x = frames[:min_len]
        ax.plot(x, ratio, color='#7B1FA2', linewidth=1.5, alpha=0.9)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.text(0.95, 0.95,
                f'Final ratio: {ratio[-1]:.3f}\n(1.0 = equal complexity)',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender',
                          alpha=0.8))

    ax.set_xlabel('Training Frames', fontsize=11)
    ax.set_ylabel('Loss_Up / Loss_Down', fontsize=11)
    ax.set_title('(b) Relative Loss: PC Model vs Standard',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f'{x/1e6:.0f}M'))

    plt.suptitle('Loss Analysis: Partner-Conditioned Training',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig4_loss_analysis.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def print_summary(data, output_dir):
    """Print a text summary of training results."""
    frames = data['frames']
    total = frames[-1]
    window = 500

    print('\n' + '=' * 60)
    print('TRAINING SUMMARY: Partner-Conditioned (pc50)')
    print('=' * 60)
    print(f'Total frames trained: {total:,.0f}')
    print(f'Total log entries: {len(frames):,}')

    for pos in ['landlord', 'landlord_up', 'landlord_down']:
        col = f'mean_episode_return_{pos}'
        if col in data:
            final = np.mean(data[col][-window:])
            early = np.mean(data[col][:window])
            print(f'\n  {pos}:')
            print(f'    Early return (first {window}): {early:.4f}')
            print(f'    Final return (last {window}):  {final:.4f}')
            print(f'    Improvement: {final - early:+.4f}')

    # Save summary JSON
    summary = {
        'total_frames': float(total),
        'total_entries': len(frames),
    }
    for pos in ['landlord', 'landlord_up', 'landlord_down']:
        col = f'mean_episode_return_{pos}'
        if col in data:
            summary[f'final_return_{pos}'] = float(np.mean(data[col][-500:]))
            summary[f'final_loss_{pos}'] = float(
                np.mean(data.get(f'loss_{pos}', [0])[-500:]))

    results_path = os.path.join(output_dir, 'training_summary.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved: {results_path}')
    print('=' * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Generate training plots from pc50 logs')
    parser.add_argument('--log_path', type=str,
                        default='results/training_log.csv',
                        help='Path to training_log.csv or logs.csv')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory for output plots')
    args = parser.parse_args()

    # Also check the checkpoint dir if the results copy doesn't exist
    if not os.path.exists(args.log_path):
        alt = 'douzero_checkpoints/partner_conditioned_50/logs.csv'
        if os.path.exists(alt):
            args.log_path = alt
            print(f'Using log from: {alt}')
        else:
            print(f'ERROR: Log file not found at {args.log_path} or {alt}')
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Reading training log: {args.log_path}')
    data = parse_log_csv(args.log_path)
    print(f'Loaded {len(data["frames"]):,} entries, '
          f'{data["frames"][-1]:,.0f} total frames')

    print('\n--- Generating Plots ---')
    plot_training_curves(data, args.output_dir)
    plot_return_trajectory(data, args.output_dir)
    plot_behavioral_comparison(args.output_dir)
    plot_loss_analysis(data, args.output_dir)
    print_summary(data, args.output_dir)

    print('\nAll plots generated! Check the results/ directory.')


if __name__ == '__main__':
    main()
