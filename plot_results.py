"""
plot_results.py — Generate paper-ready figures from evaluation results.

Run after cross_play_and_metrics.py finishes.

Usage:
    python plot_results.py

Or copy cells into your Colab notebook.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12

# ================================================================
# 1. TRAINING CURVES COMPARISON (3-way)
# ================================================================

def plot_training_curves():
    """Compare training dynamics across baseline, frozen, and partner-randomized."""
    import pandas as pd
    
    configs = {
        'Baseline': 'douzero_checkpoints/douzero/logs.csv',
        'Frozen': 'douzero_checkpoints/asym_freeze_down/logs.csv',
        'PartnerRand': 'douzero_checkpoints/partner_random/logs.csv',
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'Baseline': '#2196F3', 'Frozen': '#F44336', 'PartnerRand': '#4CAF50'}
    
    for name, path in configs.items():
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"  Skipping {name}: {path} not found")
            continue
        
        color = colors[name]
        
        # Mean episode returns
        ax = axes[0]
        ax.plot(df['frames'], df['mean_episode_return_landlord'],
                label=f'{name} Landlord', color=color, linestyle='--', alpha=0.7)
        ax.plot(df['frames'], df['mean_episode_return_landlord_up'],
                label=f'{name} Peasant Up', color=color, alpha=0.9)
        
        # Loss
        ax = axes[1]
        window = 50
        ax.plot(df['frames'], df['loss_landlord_up'].rolling(window).mean(),
                label=f'{name} Peasant Up', color=color, alpha=0.9)
    
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
    plt.savefig('fig_training_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: fig_training_curves_comparison.png")


# ================================================================
# 2. CROSS-PLAY HEATMAP
# ================================================================

def plot_cross_play_heatmap(results_path='evaluation_results.json'):
    """Generate a heatmap of cross-play win rates."""
    with open(results_path) as f:
        data = json.load(f)
    
    cross_play = data.get('cross_play', {})
    if not cross_play:
        print("No cross-play results found.")
        return
    
    # Extract unique names
    names = sorted(set(
        k.split("'")[1] for k in cross_play.keys()
    ))
    
    # Build matrix
    n = len(names)
    matrix = np.zeros((n, n))
    for i, up_name in enumerate(names):
        for j, down_name in enumerate(names):
            key = f"('{up_name}', '{down_name}')"
            if key in cross_play:
                matrix[i, j] = cross_play[key]['wp']
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)
    ax.set_xlabel('Landlord Down Source')
    ax.set_ylabel('Landlord Up Source')
    ax.set_title('Cross-Play Peasant Win Rate')
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = 'white' if val < 0.3 or val > 0.7 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    color=color, fontsize=14, fontweight='bold')
    
    # Highlight diagonal (self-play)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor='black', linewidth=2))
    
    plt.colorbar(im, label='Peasant Win Rate')
    plt.tight_layout()
    plt.savefig('fig_cross_play_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: fig_cross_play_heatmap.png")
    
    # Print robustness summary
    print("\nRobustness (std of win rate across partners):")
    for i, name in enumerate(names):
        row_std = np.std(matrix[i, :])
        col_std = np.std(matrix[:, i])
        print(f"  {name}: as Up std={row_std:.4f}, as Down std={col_std:.4f}")
    print("Lower std = more robust to partner changes")


# ================================================================
# 3. BEHAVIORAL METRICS BAR CHART
# ================================================================

def plot_behavioral_metrics(results_path='evaluation_results.json'):
    """Bar chart comparing behavioral metrics across conditions."""
    with open(results_path) as f:
        data = json.load(f)
    
    behavioral = data.get('behavioral', {})
    if not behavioral:
        print("No behavioral results found.")
        return
    
    names = list(behavioral.keys())
    
    metrics_to_plot = [
        ('peasant_win_rate', 'Peasant Win Rate'),
        ('peasant_pass_rate', 'Pass Rate'),
        ('partner_enabling_pass_rate', 'Partner-Enabling\nPass %'),
        ('landlord_block_rate', 'Landlord\nBlock Rate'),
    ]
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 5))
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
    
    for ax_idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        values = [behavioral[name].get(metric_key, 0) for name in names]
        bars = ax.bar(range(len(names)), values, color=colors[:len(names)])
        
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Behavioral Cooperation Metrics by Training Condition', fontsize=14)
    plt.tight_layout()
    plt.savefig('fig_behavioral_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: fig_behavioral_metrics.png")


# ================================================================
# 4. ROBUSTNESS SCATTER PLOT
# ================================================================

def plot_robustness_scatter(results_path='evaluation_results.json'):
    """
    Plot self-play performance vs cross-play robustness.
    X-axis: self-play win rate (diagonal of cross-play matrix)
    Y-axis: mean cross-play win rate (off-diagonal)
    
    This is the money figure: it shows whether partner-randomized
    agents sacrifice self-play performance for cross-play robustness.
    """
    with open(results_path) as f:
        data = json.load(f)
    
    cross_play = data.get('cross_play', {})
    if not cross_play:
        print("No cross-play results found.")
        return
    
    names = sorted(set(k.split("'")[1] for k in cross_play.keys()))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
    
    for idx, name in enumerate(names):
        # Self-play win rate
        self_key = f"('{name}', '{name}')"
        self_wp = cross_play.get(self_key, {}).get('wp', 0)
        
        # Cross-play win rates (off-diagonal)
        cross_wps = []
        for other in names:
            if other != name:
                key = f"('{name}', '{other}')"
                if key in cross_play:
                    cross_wps.append(cross_play[key]['wp'])
                key2 = f"('{other}', '{name}')"
                if key2 in cross_play:
                    cross_wps.append(cross_play[key2]['wp'])
        
        cross_mean = np.mean(cross_wps) if cross_wps else 0
        cross_std = np.std(cross_wps) if cross_wps else 0
        
        ax.scatter(self_wp, cross_mean, s=200, c=colors[idx],
                   label=name, zorder=5, edgecolors='black')
        ax.errorbar(self_wp, cross_mean, yerr=cross_std,
                    color=colors[idx], capsize=5, zorder=4)
    
    # Reference line (x=y means no robustness gap)
    lims = [0, 1]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='No gap (x=y)')
    
    ax.set_xlabel('Self-Play Win Rate (with own partner)', fontsize=12)
    ax.set_ylabel('Mean Cross-Play Win Rate (with other partners)', fontsize=12)
    ax.set_title('Self-Play Performance vs Cross-Play Robustness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('fig_robustness_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: fig_robustness_scatter.png")


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    print("Generating figures...\n")
    
    # 1. Training curves
    print("--- Training Curves ---")
    plot_training_curves()
    
    # 2-4 require evaluation_results.json from cross_play_and_metrics.py
    if os.path.exists('evaluation_results.json'):
        print("\n--- Cross-Play Heatmap ---")
        plot_cross_play_heatmap()
        
        print("\n--- Behavioral Metrics ---")
        plot_behavioral_metrics()
        
        print("\n--- Robustness Scatter ---")
        plot_robustness_scatter()
    else:
        print("\nevaluation_results.json not found.")
        print("Run cross_play_and_metrics.py first.")
    
    print("\nDone! Figures saved as PNG files.")
