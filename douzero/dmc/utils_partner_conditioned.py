"""
utils_partner_conditioned.py — Actor function for partner-conditioned training.

Contains act_partner_conditioned(), which extends act_partner_random() to:
1. Track partner behavioral features during each episode
2. Store partner features in the replay buffer alongside observations
3. Pass partner features to the partner-conditioned model during action selection

Drop this file into douzero/dmc/ alongside utils.py and utils_partner_random.py.
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
from .partner_features import PartnerFeatureTracker
from .utils_partner_random import load_random_partner_weights


def act_partner_conditioned(i, device, free_queue, full_queue,
                            model, buffers, flags):
    """
    Modified actor that:
    1. Randomizes peasant partner weights (like act_partner_random)
    2. Tracks partner behavioral features during each episode
    3. Records partner features in buffers for training
    4. Passes partner features to the model for landlord_up's actions

    The partner feature tracking works as follows:
    - A PartnerFeatureTracker is maintained per episode
    - Every action by any player updates the tracker
    - When landlord_up acts, the tracker provides a 6-dim feature vector
      summarizing landlord_down's behavior so far in this episode
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
        log.info('Device %s Actor %i started (partner-CONDITIONED, pos=%s, '
                 'p=%.2f, strategy=%s).',
                 str(device), i, partner_position,
                 partner_random_prob, sample_strategy)

        env = create_env(flags)
        env = Environment(env, device)

        # Create LOCAL copies of randomized position sub-networks
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
        # NEW: partner features buffer (only for landlord_up)
        obs_partner_features_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}

        position, obs, env_output = env.initial()

        # Initialize partner feature tracker
        tracker = PartnerFeatureTracker()

        while True:
            # ============================================================
            # PARTNER RANDOMIZATION (same as act_partner_random)
            # ============================================================
            tracker.reset()

            for pos in randomized_positions:
                current_pool = pool_dir
                # NEW: Check for pro/noob sub-pools
                pro_dir = os.path.join(pool_dir, 'pro')
                noob_dir = os.path.join(pool_dir, 'noob')
                if os.path.exists(pro_dir) and os.path.exists(noob_dir):
                    current_pool = pro_dir if random.random() < 0.5 else noob_dir

                if random.random() < partner_random_prob:
                    pool_weights = load_random_partner_weights(
                        current_pool, pos, device, sample_strategy
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

            # Track who has control during this episode
            who_has_control = 'landlord'  # Landlord starts

            # ============================================================
            # Play one episode
            # ============================================================
            while True:
                obs_x_no_action_buf[position].append(
                    env_output['obs_x_no_action'])
                obs_z_buf[position].append(env_output['obs_z'])

                # Compute partner features for landlord_up
                if position == 'landlord_up':
                    partner_feats = tracker.get_features('landlord_down')
                    partner_feats_tensor = torch.from_numpy(
                        partner_feats).float()
                    obs_partner_features_buf[position].append(
                        partner_feats_tensor)
                else:
                    # For other positions, store zeros (won't be used in training)
                    obs_partner_features_buf[position].append(
                        torch.zeros(6, dtype=torch.float32))

                with torch.no_grad():
                    if position in local_nets:
                        # Use local randomized network
                        agent_output = local_nets[position].forward(
                            obs['z_batch'], obs['x_batch'],
                            return_value=False, flags=flags
                        )
                    elif position == 'landlord_up':
                        # Use partner-conditioned model with features
                        if not isinstance(device, str) or device != 'cpu':
                            dev = torch.device('cuda:' + str(device))
                        else:
                            dev = torch.device('cpu')
                        pf = torch.from_numpy(partner_feats).float().to(dev)
                        # Expand to match batch dim of x_batch
                        pf_batch = pf.unsqueeze(0).expand(
                            obs['x_batch'].shape[0], -1)
                        agent_output = model.forward(
                            position, obs['z_batch'], obs['x_batch'],
                            flags=flags, partner_features=pf_batch
                        )
                    else:
                        # Use shared model (landlord)
                        agent_output = model.forward(
                            position, obs['z_batch'], obs['x_batch'],
                            flags=flags
                        )

                _action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs['legal_actions'][_action_idx]
                obs_action_buf[position].append(_cards2tensor(action))

                # Update partner feature tracker
                is_pass = len(action) == 0
                if not is_pass:
                    who_has_control = position
                tracker.update(position, action, who_has_control)

                size[position] += 1
                position, obs, env_output = env.step(action)

                if env_output['done']:
                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            done_buf[p].extend(
                                [False for _ in range(diff - 1)])
                            done_buf[p].append(True)

                            episode_return = env_output['episode_return'] \
                                if p == 'landlord' \
                                else -env_output['episode_return']
                            episode_return_buf[p].extend(
                                [0.0 for _ in range(diff - 1)])
                            episode_return_buf[p].append(episode_return)
                            target_buf[p].extend(
                                [episode_return for _ in range(diff)])
                    break

            # Write completed trajectory segments to buffers
            for p in positions:
                while size[p] > T:
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = \
                            episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['obs_x_no_action'][index][t, ...] = \
                            obs_x_no_action_buf[p][t]
                        buffers[p]['obs_action'][index][t, ...] = \
                            obs_action_buf[p][t]
                        buffers[p]['obs_z'][index][t, ...] = \
                            obs_z_buf[p][t]
                        buffers[p]['obs_partner_features'][index][t, ...] = \
                            obs_partner_features_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                    obs_action_buf[p] = obs_action_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    obs_partner_features_buf[p] = \
                        obs_partner_features_buf[p][T:]
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
