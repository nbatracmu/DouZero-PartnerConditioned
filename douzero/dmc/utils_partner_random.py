"""
utils_partner_random.py — Drop into douzero/dmc/ alongside utils.py

Contains act_partner_random(), a modified version of act() that
randomizes peasant weights each episode. Supports randomizing
one peasant position or both simultaneously.
"""

import os
import copy
import glob
import random
import traceback
import time as time_module
import numpy as np

import torch

from .env_utils import Environment
from .utils import create_env, _cards2tensor, log


def _get_map_location(device):
    """Convert DouZero's device (int or 'cpu') to a proper torch map_location."""
    if isinstance(device, int):
        return 'cuda:' + str(device)
    elif isinstance(device, str) and device != 'cpu':
        return 'cuda:' + device
    else:
        return 'cpu'


def load_random_partner_weights(pool_dir, position, device, sample_strategy='uniform'):
    """
    Load a random checkpoint from the partner pool.
    
    Returns:
        state_dict or None if pool is empty or load fails
    """
    pattern = os.path.join(pool_dir, 'pool_*.tar')
    pool_files = sorted(glob.glob(pattern))

    if not pool_files:
        return None

    if sample_strategy == 'uniform':
        chosen = random.choice(pool_files)
    elif sample_strategy == 'recent_biased':
        n = len(pool_files)
        weights = np.array([np.exp(i / max(n * 0.3, 1)) for i in range(n)])
        weights = weights / weights.sum()
        chosen = np.random.choice(pool_files, p=weights)
    elif sample_strategy == 'recent_only':
        start = max(0, int(len(pool_files) * 0.7))
        chosen = random.choice(pool_files[start:])
    else:
        chosen = random.choice(pool_files)

    map_loc = _get_map_location(device)

    for attempt in range(3):
        try:
            checkpoint = torch.load(chosen, map_location=map_loc)
            if position in checkpoint:
                return checkpoint[position]
            key = f'model_state_dict_{position}'
            if key in checkpoint:
                return checkpoint[key]
            if 'model_state_dict' in checkpoint and position in checkpoint['model_state_dict']:
                return checkpoint['model_state_dict'][position]
            log.warning('Position %s not found in %s, keys: %s',
                        position, chosen, list(checkpoint.keys()))
            return None
        except Exception as e:
            if attempt < 2:
                time_module.sleep(0.5)
                continue
            log.warning('Failed to load pool checkpoint %s after 3 attempts: %s',
                        chosen, e)

    return None


def act_partner_random(i, device, free_queue, full_queue, model, buffers, flags):
    """
    Modified actor that randomizes peasant model weights during self-play.
    
    Supports three modes via flags.partner_position:
    - 'landlord_down': only randomize landlord_down (default)
    - 'landlord_up': only randomize landlord_up  
    - 'both': randomize BOTH peasants independently each episode
    
    When 'both' is used, each peasant independently gets randomized
    with probability partner_random_prob. They may get weights from
    DIFFERENT checkpoints, simulating a situation where both peasants
    are unfamiliar with each other.
    """
    positions = ['landlord', 'landlord_up', 'landlord_down']
    partner_position = getattr(flags, 'partner_position', 'landlord_down')
    partner_random_prob = getattr(flags, 'partner_random_prob', 0.5)
    pool_dir = getattr(flags, 'pool_dir', 'partner_pool')
    sample_strategy = getattr(flags, 'pool_sample_strategy', 'uniform')

    # Determine which positions to randomize
    if partner_position == 'both':
        randomized_positions = ['landlord_up', 'landlord_down']
    else:
        randomized_positions = [partner_position]

    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started (partner-randomized, pos=%s, p=%.2f, strategy=%s).',
                 str(device), i, partner_position, partner_random_prob, sample_strategy)

        env = create_env(flags)
        env = Environment(env, device)

        # Create LOCAL copies of each randomized position's sub-network.
        # Each gets its own independent deepcopy so weights can be swapped
        # without affecting shared memory or each other.
        local_nets = {}
        for pos in randomized_positions:
            local_nets[pos] = copy.deepcopy(model.get_model(pos))
            local_nets[pos].eval()

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}

        position, obs, env_output = env.initial()

        while True:
            # ============================================================
            # PARTNER RANDOMIZATION: at the start of each episode,
            # independently decide for each randomized position whether
            # to use pool weights or current weights.
            #
            # When mode is 'both', each peasant rolls independently,
            # so they may get weights from different checkpoints —
            # maximum partner diversity.
            # ============================================================
            for pos in randomized_positions:
                if random.random() < partner_random_prob:
                    pool_weights = load_random_partner_weights(
                        pool_dir, pos, device, sample_strategy
                    )
                    if pool_weights is not None:
                        local_nets[pos].load_state_dict(pool_weights)
                    else:
                        local_nets[pos].load_state_dict(
                            model.get_model(pos).state_dict()
                        )
                else:
                    local_nets[pos].load_state_dict(
                        model.get_model(pos).state_dict()
                    )
            # ============================================================

            # Play one episode
            while True:
                obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                obs_z_buf[position].append(env_output['obs_z'])

                with torch.no_grad():
                    if position in local_nets:
                        # Use local randomized network
                        agent_output = local_nets[position].forward(
                            obs['z_batch'], obs['x_batch'],
                            return_value=False, flags=flags
                        )
                    else:
                        # Use shared model (landlord, or non-randomized peasant)
                        agent_output = model.forward(
                            position, obs['z_batch'], obs['x_batch'], flags=flags
                        )

                _action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs['legal_actions'][_action_idx]
                obs_action_buf[position].append(_cards2tensor(action))
                size[position] += 1
                position, obs, env_output = env.step(action)

                if env_output['done']:
                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff - 1)])
                            done_buf[p].append(True)

                            episode_return = env_output['episode_return'] \
                                if p == 'landlord' else -env_output['episode_return']
                            episode_return_buf[p].extend([0.0 for _ in range(diff - 1)])
                            episode_return_buf[p].append(episode_return)
                            target_buf[p].extend([episode_return for _ in range(diff)])
                    break

            # Write completed trajectory segments to buffers
            for p in positions:
                while size[p] > T:
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
                        buffers[p]['obs_action'][index][t, ...] = obs_action_buf[p][t]
                        buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                    obs_action_buf[p] = obs_action_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    size[p] -= T

            # Re-initialize for the next episode
            position, obs, env_output = env.initial()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e
