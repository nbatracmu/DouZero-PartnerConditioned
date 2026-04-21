#!/usr/bin/env python3
"""
plot_pc50_results.py — Generate training analysis plots from pc logs.
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
    with open(path, 'r') as f:
        lines = f.readlines()

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

    for key in data:
        data[key] = np.array(data[key])
    return data


def smooth(arr, window=200):
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


def plot_training_curves(data, output_dir):
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

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_training_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str,
                        default='results/training_log.csv',
                        help='Path to log')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = parse_log_csv(args.log_path)
    plot_training_curves(data, args.output_dir)

if __name__ == '__main__':
    main()
