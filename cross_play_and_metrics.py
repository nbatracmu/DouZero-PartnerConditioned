"""
cross_play_and_metrics.py — Cross-play evaluation & behavioral metrics

Place in DouZero root directory. Run as a script or copy cells into Colab.

This script does two things:
1. Cross-play evaluation: pairs agents from different training runs and
   measures peasant win rates to test generalization
2. Behavioral metrics: runs games while recording action-level data to
   compute cooperation proxies (pass rates, card reduction, etc.)

Usage:
    python cross_play_and_metrics.py
"""

import os
import sys
import glob
import pickle
import itertools
import numpy as np
from collections import defaultdict, Counter

import torch

# ================================================================
# PART 1: CROSS-PLAY EVALUATION
# ================================================================

def find_latest_weights(checkpoint_dir):
    """Find the most recent weight files for each position in a checkpoint dir."""
    weights = {}
    for position in ['landlord', 'landlord_up', 'landlord_down']:
        pattern = os.path.join(checkpoint_dir, f'{position}_weights_*.ckpt')
        files = sorted(glob.glob(pattern))
        if files:
            weights[position] = files[-1]
        else:
            weights[position] = None
    return weights


def run_cross_play_evaluation(configs, eval_data_path='eval_data.pkl',
                              num_workers=5, gpu_device='0'):
    """
    Run cross-play evaluation: pair landlord_up from one config with
    landlord_down from another, against a fixed landlord.
    
    Args:
        configs: dict of {name: checkpoint_dir} for each training run
        eval_data_path: path to eval_data.pkl (generate with generate_eval_data.py)
        num_workers: number of parallel workers
        gpu_device: GPU device string
    
    Returns:
        results: dict of {(up_name, down_name): win_rate}
    """
    from douzero.evaluation.simulation import evaluate
    
    # Find latest weights for each config
    all_weights = {}
    for name, ckpt_dir in configs.items():
        w = find_latest_weights(ckpt_dir)
        all_weights[name] = w
        print(f"{name}: L={w['landlord']}, U={w['landlord_up']}, D={w['landlord_down']}")
    
    # Use the baseline landlord for all evaluations (keeps it controlled)
    # You can change this to use each config's own landlord if preferred
    landlord_ckpt = all_weights[list(configs.keys())[0]]['landlord']
    print(f"\nUsing landlord from: {landlord_ckpt}")
    
    # Check eval data exists
    if not os.path.exists(eval_data_path):
        print(f"\neval_data.pkl not found at {eval_data_path}")
        print("Generate it with: python generate_eval_data.py")
        print("This creates random card deals for consistent evaluation.\n")
        return None
    
    results = {}
    for up_name in configs:
        for down_name in configs:
            up_ckpt = all_weights[up_name]['landlord_up']
            down_ckpt = all_weights[down_name]['landlord_down']
            
            if up_ckpt is None or down_ckpt is None:
                print(f"  SKIP Up={up_name}, Down={down_name} (missing weights)")
                continue
            
            print(f"\n--- Evaluating: Up={up_name}, Down={down_name} ---")
            
            # DouZero's evaluate() prints results and returns win rates
            wp, adp = evaluate(
                landlord_ckpt,
                up_ckpt,
                down_ckpt,
                eval_data_path,
                num_workers
            )
            
            results[(up_name, down_name)] = {
                'wp': wp,   # win percentage
                'adp': adp  # average difference points
            }
            print(f"  Result: WP={wp:.4f}, ADP={adp:.4f}")
    
    return results


def print_cross_play_table(results, config_names):
    """Print cross-play results as a formatted table."""
    print("\n" + "=" * 60)
    print("CROSS-PLAY RESULTS: Peasant Win Percentage")
    print("Rows = landlord_up source, Columns = landlord_down source")
    print("=" * 60)
    
    # Header
    header = f"{'Up \\ Down':<20}"
    for name in config_names:
        header += f"{name:<15}"
    print(header)
    print("-" * (20 + 15 * len(config_names)))
    
    for up_name in config_names:
        row = f"{up_name:<20}"
        for down_name in config_names:
            key = (up_name, down_name)
            if key in results:
                row += f"{results[key]['wp']:.4f}         "
            else:
                row += f"{'N/A':<15}"
        print(row)
    
    # Compute and print robustness metrics
    print("\n--- Robustness Analysis ---")
    for name in config_names:
        # How well does this agent's landlord_up do with different partners?
        up_scores = [results[(name, dn)]['wp'] for dn in config_names 
                     if (name, dn) in results]
        if up_scores:
            mean = np.mean(up_scores)
            std = np.std(up_scores)
            print(f"  {name} (as Up):   mean WP = {mean:.4f}, std = {std:.4f}")
        
        # How well does this agent's landlord_down do with different partners?
        down_scores = [results[(up, name)]['wp'] for up in config_names 
                       if (up, name) in results]
        if down_scores:
            mean = np.mean(down_scores)
            std = np.std(down_scores)
            print(f"  {name} (as Down): mean WP = {mean:.4f}, std = {std:.4f}")


# ================================================================
# PART 2: BEHAVIORAL METRICS
# ================================================================

def load_agent(ckpt_path, position, device='cpu'):
    """Load a trained agent's network."""
    from douzero.dmc.models import model_dict
    
    model_cls = model_dict[position]
    net = model_cls().to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def run_games_with_logging(landlord_ckpt, up_ckpt, down_ckpt,
                           num_games=1000, device='cpu'):
    """
    Run evaluation games while recording detailed action-level data
    for behavioral metric computation.
    
    Records for each turn:
    - position: who played
    - action: what they played (list of card ints)
    - is_pass: whether the action was a pass (empty list)
    - hand_sizes: dict of hand sizes for all players after the action
    - who_has_control: which player last played a non-pass action
    - game_result: 'landlord' or 'peasant' (filled in at end of game)
    
    Returns:
        list of game records, each containing turn-by-turn data
    """
    from douzero.env import Env
    
    # Load models
    agents = {
        'landlord': load_agent(landlord_ckpt, 'landlord', device),
        'landlord_up': load_agent(up_ckpt, 'landlord_up', device),
        'landlord_down': load_agent(down_ckpt, 'landlord_down', device),
    }
    
    all_games = []
    env = Env('adp')
    
    for game_idx in range(num_games):
        # Initialize game
        position, obs_dict, overall = env.reset()
        
        game_record = {
            'turns': [],
            'result': None,
            'game_idx': game_idx,
        }
        
        who_has_control = 'landlord'  # Landlord starts
        
        done = False
        while not done:
            # Get observation for current position
            obs = obs_dict
            
            # Get action from agent
            agent = agents[position]
            
            # Build input tensors
            z_batch = torch.from_numpy(obs['z_batch']).float().to(device)
            x_batch = torch.from_numpy(obs['x_batch']).float().to(device)
            
            with torch.no_grad():
                output = agent.forward(z_batch, x_batch, return_value=False, flags=None)
            
            action_idx = int(output['action'].cpu().numpy())
            action = obs['legal_actions'][action_idx]
            
            is_pass = len(action) == 0
            
            if not is_pass:
                who_has_control = position
            
            # Record this turn
            turn_record = {
                'position': position,
                'action': action,
                'is_pass': is_pass,
                'num_cards_played': len(action),
                'who_has_control': who_has_control,
            }
            game_record['turns'].append(turn_record)
            
            # Step environment
            position, obs_dict, overall = env.step(action)
            done = overall.get('done', False) if isinstance(overall, dict) else False
        
        # Record game result
        game_record['result'] = 'landlord' if overall.get('episode_return', 0) > 0 else 'peasant'
        all_games.append(game_record)
        
        if (game_idx + 1) % 100 == 0:
            peasant_wins = sum(1 for g in all_games if g['result'] == 'peasant')
            print(f"  Game {game_idx + 1}/{num_games}, "
                  f"Peasant WR: {peasant_wins / len(all_games):.3f}")
    
    return all_games


def compute_behavioral_metrics(game_records):
    """
    Compute behavioral proxy metrics for peasant cooperation.
    
    Metrics:
    1. Strategic Pass Rate: fraction of peasant turns that are passes
       when the peasant had at least one non-pass legal action AND
       the partner or other peasant has control
    2. Partner-Enabling Pass Rate: fraction of peasant passes that
       occur when the partner has control (helping partner keep initiative)
    3. Overall Peasant Pass Rate: simple fraction of peasant turns that pass
    4. Peasant Win Rate: fraction of games won by peasant team
    5. Mean Peasant Cards Per Turn: average cards played per peasant turn
    6. Landlord Block Rate: fraction of peasant non-pass plays that
       immediately follow a landlord play (blocking the landlord)
    
    Args:
        game_records: list of game dicts from run_games_with_logging
    
    Returns:
        dict of metric_name -> value
    """
    metrics = {}
    
    # Counters
    peasant_turns = 0
    peasant_passes = 0
    peasant_passes_partner_control = 0
    peasant_passes_landlord_control = 0
    peasant_plays_after_landlord = 0
    peasant_non_pass_turns = 0
    total_peasant_cards_played = 0
    peasant_positions = ['landlord_up', 'landlord_down']
    
    for game in game_records:
        turns = game['turns']
        for t_idx, turn in enumerate(turns):
            if turn['position'] not in peasant_positions:
                continue
            
            peasant_turns += 1
            partner = 'landlord_down' if turn['position'] == 'landlord_up' else 'landlord_up'
            
            if turn['is_pass']:
                peasant_passes += 1
                if turn['who_has_control'] == partner:
                    peasant_passes_partner_control += 1
                elif turn['who_has_control'] == 'landlord':
                    peasant_passes_landlord_control += 1
            else:
                peasant_non_pass_turns += 1
                total_peasant_cards_played += turn['num_cards_played']
                
                # Check if this play follows a landlord play (blocking)
                if t_idx > 0 and turns[t_idx - 1]['position'] == 'landlord' \
                        and not turns[t_idx - 1]['is_pass']:
                    peasant_plays_after_landlord += 1
    
    # Compute rates
    metrics['peasant_win_rate'] = (
        sum(1 for g in game_records if g['result'] == 'peasant') / len(game_records)
        if game_records else 0
    )
    metrics['peasant_pass_rate'] = (
        peasant_passes / peasant_turns if peasant_turns > 0 else 0
    )
    metrics['partner_enabling_pass_rate'] = (
        peasant_passes_partner_control / peasant_passes if peasant_passes > 0 else 0
    )
    metrics['landlord_control_pass_rate'] = (
        peasant_passes_landlord_control / peasant_passes if peasant_passes > 0 else 0
    )
    metrics['landlord_block_rate'] = (
        peasant_plays_after_landlord / peasant_non_pass_turns 
        if peasant_non_pass_turns > 0 else 0
    )
    metrics['mean_cards_per_peasant_play'] = (
        total_peasant_cards_played / peasant_non_pass_turns 
        if peasant_non_pass_turns > 0 else 0
    )
    metrics['total_games'] = len(game_records)
    metrics['total_peasant_turns'] = peasant_turns
    
    return metrics


def run_behavioral_analysis(configs, num_games=500, device='cpu'):
    """
    Run behavioral metric analysis for each training configuration.
    
    For each config, runs games using that config's own agents (self-play),
    then computes behavioral metrics.
    """
    all_metrics = {}
    
    for name, ckpt_dir in configs.items():
        print(f"\n{'='*50}")
        print(f"Behavioral Analysis: {name}")
        print(f"{'='*50}")
        
        weights = find_latest_weights(ckpt_dir)
        
        if any(v is None for v in weights.values()):
            print(f"  SKIP {name}: missing weight files")
            continue
        
        print(f"  Running {num_games} evaluation games...")
        games = run_games_with_logging(
            weights['landlord'],
            weights['landlord_up'],
            weights['landlord_down'],
            num_games=num_games,
            device=device
        )
        
        metrics = compute_behavioral_metrics(games)
        all_metrics[name] = metrics
        
        print(f"\n  Results for {name}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
    
    return all_metrics


def print_behavioral_comparison(all_metrics):
    """Print a comparison table of behavioral metrics across configs."""
    if not all_metrics:
        return
    
    metric_keys = [
        'peasant_win_rate',
        'peasant_pass_rate',
        'partner_enabling_pass_rate',
        'landlord_control_pass_rate',
        'landlord_block_rate',
        'mean_cards_per_peasant_play',
    ]
    
    metric_labels = {
        'peasant_win_rate': 'Peasant Win Rate',
        'peasant_pass_rate': 'Pass Rate (all)',
        'partner_enabling_pass_rate': 'Partner-Enabling Pass %',
        'landlord_control_pass_rate': 'Pass When LL Controls %',
        'landlord_block_rate': 'Landlord Block Rate',
        'mean_cards_per_peasant_play': 'Avg Cards/Play',
    }
    
    names = list(all_metrics.keys())
    
    print("\n" + "=" * 70)
    print("BEHAVIORAL METRICS COMPARISON")
    print("=" * 70)
    
    header = f"{'Metric':<30}"
    for name in names:
        header += f"{name:<15}"
    print(header)
    print("-" * (30 + 15 * len(names)))
    
    for key in metric_keys:
        row = f"{metric_labels.get(key, key):<30}"
        for name in names:
            val = all_metrics[name].get(key, 0)
            row += f"{val:<15.4f}"
        print(row)


# ================================================================
# PART 3: MAIN — NOTEBOOK-READY CELLS
# ================================================================

if __name__ == '__main__':
    
    # ==============================================
    # CONFIGURE YOUR CHECKPOINT DIRECTORIES HERE
    # ==============================================
    configs = {
        'baseline': 'douzero_checkpoints/douzero',
        'frozen': 'douzero_checkpoints/asym_freeze_down',
        'partner_rand': 'douzero_checkpoints/partner_random',
    }
    
    # Check which configs actually have weight files
    available = {}
    for name, path in configs.items():
        weights = find_latest_weights(path)
        if all(v is not None for v in weights.values()):
            available[name] = path
            print(f"[OK] {name}: {path}")
        else:
            print(f"[SKIP] {name}: missing weights in {path}")
    
    if not available:
        print("\nNo complete checkpoint directories found!")
        print("Make sure training has run long enough to save weight files.")
        sys.exit(1)
    
    # ==============================================
    # STEP 1: Generate eval data if needed
    # ==============================================
    if not os.path.exists('eval_data.pkl'):
        print("\nGenerating evaluation data...")
        os.system('python generate_eval_data.py')
    
    # ==============================================
    # STEP 2: Cross-play evaluation
    # ==============================================
    print("\n" + "=" * 60)
    print("RUNNING CROSS-PLAY EVALUATION")
    print("=" * 60)
    
    results = run_cross_play_evaluation(available)
    if results:
        print_cross_play_table(results, list(available.keys()))
    
    # ==============================================
    # STEP 3: Behavioral metrics
    # ==============================================
    print("\n" + "=" * 60)
    print("RUNNING BEHAVIORAL METRICS ANALYSIS")
    print("=" * 60)
    
    all_metrics = run_behavioral_analysis(available, num_games=500, device='cpu')
    print_behavioral_comparison(all_metrics)
    
    # ==============================================
    # STEP 4: Save results
    # ==============================================
    import json
    
    results_out = {
        'cross_play': {str(k): v for k, v in results.items()} if results else {},
        'behavioral': all_metrics,
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results_out, f, indent=2, default=str)
    
    print("\n\nResults saved to evaluation_results.json")
