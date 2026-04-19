#!/usr/bin/env python3
"""
plot_comparison.py — Generate comparative training curves across models.

Loads Baseline, Frozen, and Partner-Conditioned logs, smooths them,
and plots them in a single structure to directly compare performance.
"""

import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def parse_log_csv(path):
    """Parse the training log CSV, handling the '# ' header prefix."""
    if not os.path.exists(path):
        return None
    
    with open(path, 'r') as f:
        lines = f.readlines()
        
    if not lines:
        return None

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

def main():
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths to the logs
    configs = {
        'Baseline': 'baselines/baseline_logs.csv',
        'Frozen': 'baselines/frozen_logs.csv',
        'PC50': 'douzero_checkpoints/partner_conditioned_50/logs.csv'
    }
    
    # Load data
    data = {}
    for name, path in configs.items():
        print(f"Loading {name} from {path}...")
        df = parse_log_csv(path)
        if df is not None:
            data[name] = df
        else:
            print(f"  -> WARNING: Could not load {name}")
            
    if not data:
        print("Error: No data loaded.")
        return

    # Set up the plot: Mean returns and Loss for Peasant Up
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'Baseline': '#2196F3', 'Frozen': '#FF9800', 'PC50': '#4CAF50'}
    
    window = 200
    
    # Plot 1: Mean Episode Returns (Peasant Up)
    ax = axes[0]
    for name, df in data.items():
        if 'mean_episode_return_landlord_up' in df:
            y = smooth(df['mean_episode_return_landlord_up'], window)
            x = df['frames'][:len(y)]
            ax.plot(x, y, label=name, color=colors[name], linewidth=2, alpha=0.9)
            
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_xlabel('Training Frames', fontsize=11)
    ax.set_ylabel('Mean Episode Return (Peasant Up)', fontsize=11)
    ax.set_title('Comparative Training Performance', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
    
    # Plot 2: Loss (Peasant Up)
    ax = axes[1]
    for name, df in data.items():
        if 'loss_landlord_up' in df:
            y = smooth(df['loss_landlord_up'], window)
            x = df['frames'][:len(y)]
            ax.plot(x, y, label=name, color=colors[name], linewidth=2, alpha=0.9)
            
    ax.set_xlabel('Training Frames', fontsize=11)
    ax.set_ylabel('Loss (Smoothed)', fontsize=11)
    ax.set_title('Comparative Training Loss', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
    
    plt.suptitle('Baseline vs. Frozen vs. Partner-Conditioned (PC50)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'fig_comparison_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparative plot to: {path}")

if __name__ == '__main__':
    main()
