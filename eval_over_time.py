"""
eval_over_time.py — Track cooperation metrics at multiple training checkpoints.

This evaluates saved weight files at different frame counts to show HOW
cooperation develops over training, not just the final state.

Place in DouZero root directory.

Usage:
    python eval_over_time.py --config_dir douzero_checkpoints/partner_random_50
    python eval_over_time.py --config_dir douzero_checkpoints/douzero
"""

import os
import sys
import glob
import argparse
import json
import numpy as np
from collections import defaultdict

import torch

def get_parser():
    parser = argparse.ArgumentParser(description='Evaluate cooperation over training time')
    parser.add_argument('--config_dir', type=str, required=True,
                        help='Checkpoint directory to evaluate')
    parser.add_argument('--num_games', type=int, default=200,
                        help='Games per checkpoint evaluation')
    parser.add_argument('--max_checkpoints', type=int, default=10,
                        help='Max number of checkpoints to evaluate (evenly spaced)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for evaluation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: config_dir/eval_over_time.json)')
    return parser


def find_checkpoint_pairs(config_dir):
    """Find all checkpoint frame numbers that have weights for all 3 positions."""
    frames_available = defaultdict(set)
    
    for position in ['landlord', 'landlord_up', 'landlord_down']:
        pattern = os.path.join(config_dir, f'{position}_weights_*.ckpt')
        for f in glob.glob(pattern):
            basename = os.path.basename(f)
            # Extract frame number: "landlord_weights_12345.ckpt" -> 12345
            frame_str = basename.replace(f'{position}_weights_', '').replace('.ckpt', '')
            try:
                frame = int(frame_str)
                frames_available[frame].add(position)
            except ValueError:
                continue
    
    # Only keep frames where all 3 positions are available
    complete_frames = sorted([
        f for f, positions in frames_available.items()
        if len(positions) == 3
    ])
    
    return complete_frames


def load_agent(ckpt_path, position, device='cpu'):
    """Load a trained agent's network."""
    from douzero.dmc.models import model_dict
    model_cls = model_dict[position]
    net = model_cls().to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def evaluate_checkpoint(config_dir, frame, num_games, device='cpu'):
    """
    Run games with a specific checkpoint and compute cooperation metrics.
    
    Uses DouZero's built-in environment to simulate games and tracks
    action-level data for behavioral analysis.
    """
    from douzero.env import Env
    from douzero.env.env import _cards2array
    
    # Load agents for this checkpoint
    agents = {}
    for position in ['landlord', 'landlord_up', 'landlord_down']:
        ckpt_path = os.path.join(config_dir, f'{position}_weights_{frame}.ckpt')
        agents[position] = load_agent(ckpt_path, position, device)
    
    # Metrics accumulators
    peasant_wins = 0
    total_peasant_turns = 0
    total_peasant_passes = 0
    passes_when_partner_controls = 0
    passes_when_landlord_controls = 0
    total_peasant_cards_played = 0
    total_peasant_non_pass = 0
    total_game_lengths = []
    
    env = Env('adp')
    
    for game_idx in range(num_games):
        # Reset environment
        # DouZero's Env.reset() returns card deal info
        # We need to step through the game manually
        
        # Use the env's internal game setup
        env.reset()
        
        # Get initial state
        position = env.get_acting_player_position()
        obs = env.get_obs(position)
        
        who_controls = 'landlord'  # Landlord starts
        game_length = 0
        done = False
        
        while not done:
            # Get action from agent
            agent = agents[position]
            z = torch.from_numpy(obs['z_batch']).float().to(device)
            x = torch.from_numpy(obs['x_batch']).float().to(device)
            
            with torch.no_grad():
                output = agent.forward(z, x, return_value=False, flags=None)
            
            action_idx = int(output['action'].cpu().numpy())
            action = obs['legal_actions'][action_idx]
            
            is_pass = (len(action) == 0)
            is_peasant = position in ['landlord_up', 'landlord_down']
            partner = 'landlord_down' if position == 'landlord_up' else 'landlord_up'
            
            if is_peasant:
                total_peasant_turns += 1
                if is_pass:
                    total_peasant_passes += 1
                    if who_controls == partner:
                        passes_when_partner_controls += 1
                    elif who_controls == 'landlord':
                        passes_when_landlord_controls += 1
                else:
                    total_peasant_non_pass += 1
                    total_peasant_cards_played += len(action)
            
            if not is_pass:
                who_controls = position
            
            game_length += 1
            
            # Step environment
            env.step(action)
            
            # Check if game is over
            if env.game_over:
                done = True
                # In DouZero, if landlord empties hand first, landlord wins
                # Otherwise peasants win
                if env.get_acting_player_position() != 'landlord':
                    # The player who just played was a peasant who emptied hand
                    peasant_wins += 1
                # Actually check the reward
                if hasattr(env, 'get_reward'):
                    reward = env.get_reward()
                    if reward < 0:  # Negative reward = landlord lost
                        peasant_wins += 1
            else:
                position = env.get_acting_player_position()
                obs = env.get_obs(position)
        
        total_game_lengths.append(game_length)
    
    # Compute metrics
    metrics = {
        'frame': frame,
        'num_games': num_games,
        'peasant_win_rate': peasant_wins / num_games if num_games > 0 else 0,
        'peasant_pass_rate': total_peasant_passes / total_peasant_turns if total_peasant_turns > 0 else 0,
        'partner_enabling_pass_rate': passes_when_partner_controls / total_peasant_passes if total_peasant_passes > 0 else 0,
        'landlord_control_pass_rate': passes_when_landlord_controls / total_peasant_passes if total_peasant_passes > 0 else 0,
        'mean_cards_per_play': total_peasant_cards_played / total_peasant_non_pass if total_peasant_non_pass > 0 else 0,
        'mean_game_length': np.mean(total_game_lengths) if total_game_lengths else 0,
    }
    
    return metrics


def main():
    flags = get_parser().parse_args()
    config_dir = flags.config_dir
    
    if not os.path.isdir(config_dir):
        print(f"Error: {config_dir} not found")
        sys.exit(1)
    
    # Find available checkpoints
    all_frames = find_checkpoint_pairs(config_dir)
    print(f"Found {len(all_frames)} complete checkpoints in {config_dir}")
    
    if not all_frames:
        print("No complete checkpoints found (need landlord, landlord_up, landlord_down weights)")
        sys.exit(1)
    
    # Sample evenly across training
    if len(all_frames) > flags.max_checkpoints:
        indices = np.linspace(0, len(all_frames) - 1, flags.max_checkpoints, dtype=int)
        selected_frames = [all_frames[i] for i in indices]
    else:
        selected_frames = all_frames
    
    print(f"Evaluating {len(selected_frames)} checkpoints: {selected_frames}")
    
    # Evaluate each checkpoint
    results = []
    for frame in selected_frames:
        print(f"\n--- Evaluating frame {frame} ---")
        try:
            metrics = evaluate_checkpoint(
                config_dir, frame, flags.num_games, flags.device
            )
            results.append(metrics)
            print(f"  Win rate: {metrics['peasant_win_rate']:.3f}, "
                  f"Pass rate: {metrics['peasant_pass_rate']:.3f}, "
                  f"Partner-enabling: {metrics['partner_enabling_pass_rate']:.3f}")
        except Exception as e:
            print(f"  Error evaluating frame {frame}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    output_path = flags.output or os.path.join(config_dir, 'eval_over_time.json')
    with open(output_path, 'w') as f:
        json.dump({
            'config_dir': config_dir,
            'results': results,
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print summary table
    print(f"\n{'Frame':>12} {'WinRate':>10} {'PassRate':>10} {'PartnerPass':>12} {'GameLen':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['frame']:>12} {r['peasant_win_rate']:>10.3f} "
              f"{r['peasant_pass_rate']:>10.3f} {r['partner_enabling_pass_rate']:>12.3f} "
              f"{r['mean_game_length']:>10.1f}")


if __name__ == '__main__':
    main()
