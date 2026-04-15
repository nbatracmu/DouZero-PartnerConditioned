"""
dmc_partner_conditioned.py — Training loop for partner-conditioned peasant policy.

Extends dmc_partner_random.py with:
1. Uses ModelPartnerConditioned instead of Model (landlord_up gets partner encoder)
2. Extends buffers with obs_partner_features (shape [T, 6]) for all positions
3. Modified learn() passes partner features to landlord_up during training
4. Uses act_partner_conditioned() which tracks partner behavior during episodes

Everything else (checkpoint logic, pool management, logging) is identical
to dmc_partner_random.py.
"""

import os
import threading
import time
import timeit
import pprint
from collections import deque
import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .models_partner_conditioned import ModelPartnerConditioned, PARTNER_FEATURE_DIM
from .utils import get_batch, log, create_env, create_optimizers, act
from .utils_partner_conditioned import act_partner_conditioned
from .dmc_partner_random import save_pool_snapshot, seed_pool_from_baseline

import argparse

# ---- Argument parser (extends partner-random parser) ----

parser = argparse.ArgumentParser(
    description='DouZero: Partner-Conditioned + Partner-Randomized Training')

# Original DouZero arguments
parser.add_argument('--xpid', default='douzero',
                    help='Experiment id (default: douzero)')
parser.add_argument('--save_interval', default=30, type=float,
                    help='Time interval (in minutes) at which to save the model')
parser.add_argument('--objective', default='adp', type=str,
                    choices=['adp', 'wp'],
                    help='Use ADP or WP as reward (default: ADP)')
parser.add_argument('--actor_device_cpu', action='store_true',
                    help='Use CPU as actor device')
parser.add_argument('--gpu_devices', default='0', type=str,
                    help='Which GPUs to be used for training')
parser.add_argument('--num_actor_devices', default=1, type=int,
                    help='The number of devices used for simulation')
parser.add_argument('--num_actors', default=5, type=int,
                    help='The number of actors for each simulation device')
parser.add_argument('--training_device', default='0', type=str,
                    help='The index of the GPU used for training models')
parser.add_argument('--load_model', action='store_true',
                    help='Load an existing model')
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint')
parser.add_argument('--savedir', default='douzero_checkpoints',
                    help='Root dir where experiment data will be saved')
parser.add_argument('--total_frames', default=100000000000, type=int,
                    help='Total environment frames to train for')
parser.add_argument('--exp_epsilon', default=0.01, type=float,
                    help='The probability for exploration')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Learner batch size')
parser.add_argument('--unroll_length', default=100, type=int,
                    help='The unroll length (time dimension)')
parser.add_argument('--num_buffers', default=50, type=int,
                    help='Number of shared-memory buffers')
parser.add_argument('--num_threads', default=4, type=int,
                    help='Number of learner threads')
parser.add_argument('--max_grad_norm', default=40., type=float,
                    help='Max gradient norm')
parser.add_argument('--learning_rate', default=0.0001, type=float,
                    help='Learning rate')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum')
parser.add_argument('--epsilon', default=0.00001, type=float,
                    help='RMSProp epsilon')

# ---- Partner randomization arguments ----
parser.add_argument('--partner_random', action='store_true',
                    help='Enable partner-randomized peasant training')
parser.add_argument('--pool_dir', default='partner_pool', type=str,
                    help='Directory to store/read partner checkpoint pool')
parser.add_argument('--pool_save_interval', default=500000, type=int,
                    help='Save a snapshot to the pool every N frames')
parser.add_argument('--partner_random_prob', default=0.5, type=float,
                    help='Probability of using a random partner each episode')
parser.add_argument('--partner_position', default='landlord_down', type=str,
                    choices=['landlord_up', 'landlord_down', 'both'],
                    help='Which peasant position(s) to randomize')
parser.add_argument('--pool_sample_strategy', default='uniform', type=str,
                    choices=['uniform', 'recent_biased', 'recent_only'],
                    help='How to sample from the checkpoint pool')
parser.add_argument('--seed_pool_from', default=None, type=str,
                    help='Path to a baseline checkpoint dir to seed the pool from')

# ---- NEW: Partner-conditioned flag ----
parser.add_argument('--partner_conditioned', action='store_true',
                    help='Use partner-conditioned architecture for landlord_up')


# ===========================================================================
# Replay buffer return tracking
# ===========================================================================

mean_episode_return_buf = {
    p: deque(maxlen=100)
    for p in ['landlord', 'landlord_up', 'landlord_down']
}


def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets) ** 2).mean()
    return loss


def learn(position, actor_models, model, batch, optimizer, flags, lock):
    """
    Performs a learning step. Modified to pass partner features
    to landlord_up's partner-conditioned model.
    """
    if flags.training_device != "cpu":
        device = torch.device('cuda:' + str(flags.training_device))
    else:
        device = torch.device('cpu')

    obs_x_no_action = batch['obs_x_no_action'].to(device)
    obs_action = batch['obs_action'].to(device)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf[position].append(
        torch.mean(episode_returns).to(device))

    # NEW: Extract partner features for landlord_up
    partner_features = None
    if position == 'landlord_up' and flags.partner_conditioned:
        partner_features = torch.flatten(
            batch['obs_partner_features'].to(device), 0, 1).float()

    with lock:
        # Call model with partner features if applicable
        if position == 'landlord_up' and flags.partner_conditioned:
            learner_outputs = model(
                obs_z, obs_x,
                partner_features=partner_features,
                return_value=True
            )
        else:
            learner_outputs = model(obs_z, obs_x, return_value=True)

        loss = compute_loss(learner_outputs['values'], target)
        stats = {
            'mean_episode_return_' + position: torch.mean(
                torch.stack(
                    [_r for _r in mean_episode_return_buf[position]]
                )).item(),
            'loss_' + position: loss.item(),
        }

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()

        for actor_model in actor_models.values():
            actor_model.get_model(position).load_state_dict(
                model.state_dict())
        return stats


def create_buffers_with_partner_features(flags, device_iterator):
    """
    Create buffers identical to the original, plus obs_partner_features.

    The partner features buffer has shape [T, 6] for ALL positions,
    but is only meaningfully filled for landlord_up.
    """
    T = flags.unroll_length
    positions = ['landlord', 'landlord_up', 'landlord_down']
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            x_dim = 319 if position == 'landlord' else 430
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
                obs_action=dict(size=(T, 54), dtype=torch.int8),
                obs_z=dict(size=(T, 5, 162), dtype=torch.int8),
                # NEW: partner behavioral features
                obs_partner_features=dict(
                    size=(T, PARTNER_FEATURE_DIM), dtype=torch.float32),
            )
            _buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    if not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(
                            torch.device('cuda:' + str(device))
                        ).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(
                            torch.device('cpu')
                        ).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers


# ===========================================================================
# Modified train() function
# ===========================================================================

def train(flags):
    """
    Main training function for partner-conditioned + partner-randomized training.

    Changes from dmc_partner_random.py:
    1. Uses ModelPartnerConditioned (landlord_up has partner encoder)
    2. Uses create_buffers_with_partner_features (adds obs_partner_features)
    3. Uses act_partner_conditioned (tracks + records partner features)
    4. learn() passes partner features to landlord_up during training
    """
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError(
                "CUDA not available. If you have GPUs, please specify the ID "
                "after `--gpu_devices`. Otherwise, please train with CPU with "
                "`python3 train.py --actor_device_cpu --training_device cpu`")

    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser(
            '%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), \
            'The number of actor devices can not exceed available devices'

    # ---- Seed the partner pool if requested ----
    if flags.partner_random and flags.seed_pool_from:
        seed_pool_from_baseline(flags.seed_pool_from, flags.pool_dir)

    # ---- Create the pool directory ----
    if flags.partner_random:
        os.makedirs(flags.pool_dir, exist_ok=True)
        log.info('Partner-randomized training ENABLED:')
        log.info('  Pool dir: %s', flags.pool_dir)
        log.info('  Randomization prob: %.2f', flags.partner_random_prob)
        log.info('  Partner position: %s', flags.partner_position)
        log.info('  Sample strategy: %s', flags.pool_sample_strategy)
        log.info('  Pool save interval: %d frames', flags.pool_save_interval)

    if flags.partner_conditioned:
        log.info('Partner-CONDITIONED architecture ENABLED for landlord_up')

    # Initialize actor models — use partner-conditioned model
    models = {}
    for device in device_iterator:
        model = ModelPartnerConditioned(device=device)
        model.share_memory()
        model.eval()
        models[device] = model

    # Initialize buffers WITH partner features
    buffers = create_buffers_with_partner_features(flags, device_iterator)

    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = {}
    full_queue = {}

    for device in device_iterator:
        _free_queue = {
            'landlord': ctx.SimpleQueue(),
            'landlord_up': ctx.SimpleQueue(),
            'landlord_down': ctx.SimpleQueue()
        }
        _full_queue = {
            'landlord': ctx.SimpleQueue(),
            'landlord_up': ctx.SimpleQueue(),
            'landlord_down': ctx.SimpleQueue()
        }
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue

    # Learner model for training — partner-conditioned
    learner_model = ModelPartnerConditioned(device=flags.training_device)

    # Create optimizers
    optimizers = create_optimizers(flags, learner_model)

    # Stat Keys
    stat_keys = [
        'mean_episode_return_landlord',
        'loss_landlord',
        'mean_episode_return_landlord_up',
        'loss_landlord_up',
        'mean_episode_return_landlord_down',
        'loss_landlord_down',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'landlord': 0, 'landlord_up': 0, 'landlord_down': 0}

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath,
            map_location=("cuda:" + str(flags.training_device)
                          if flags.training_device != "cpu" else "cpu")
        )
        for k in ['landlord', 'landlord_up', 'landlord_down']:
            learner_model.get_model(k).load_state_dict(
                checkpoint_states["model_state_dict"][k])
            optimizers[k].load_state_dict(
                checkpoint_states["optimizer_state_dict"][k])
            for device in device_iterator:
                models[device].get_model(k).load_state_dict(
                    learner_model.get_model(k).state_dict())
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")

    # ============================================================
    # START ACTOR PROCESSES
    # Use act_partner_conditioned when both flags are set
    # ============================================================
    if flags.partner_conditioned and flags.partner_random:
        act_fn = act_partner_conditioned
    elif flags.partner_random:
        # Fallback: partner-random but not conditioned
        # (shouldn't happen in our usage, but handles edge case)
        from .utils_partner_random import act_partner_random
        act_fn = act_partner_random
    else:
        act_fn = act

    for device in device_iterator:
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act_fn,
                args=(i, device, free_queue[device], full_queue[device],
                      models[device], buffers[device], flags))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(i, device, position, local_lock, position_lock,
                        lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device][position],
                              full_queue[device][position],
                              buffers[device][position], flags, local_lock)
            _stats = learn(position, models,
                           learner_model.get_model(position), batch,
                           optimizers[position], flags, position_lock)

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                position_frames[position] += T * B

    for device in device_iterator:
        for m in range(flags.num_buffers):
            free_queue[device]['landlord'].put(m)
            free_queue[device]['landlord_up'].put(m)
            free_queue[device]['landlord_down'].put(m)

    threads = []
    locks = {}
    for device in device_iterator:
        locks[device] = {
            'landlord': threading.Lock(),
            'landlord_up': threading.Lock(),
            'landlord_down': threading.Lock()
        }
    position_locks = {
        'landlord': threading.Lock(),
        'landlord_up': threading.Lock(),
        'landlord_down': threading.Lock()
    }

    for device in device_iterator:
        for i in range(flags.num_threads):
            for position in ['landlord', 'landlord_up', 'landlord_down']:
                thread = threading.Thread(
                    target=batch_and_learn,
                    name='batch-and-learn-%d' % i,
                    args=(i, device, position,
                          locks[device][position],
                          position_locks[position]))
                thread.start()
                threads.append(thread)

    # ============================================================
    # Checkpoint + pool snapshot saving
    # ============================================================
    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)
        _models = learner_model.get_models()
        torch.save({
            'model_state_dict': {
                k: _models[k].state_dict() for k in _models},
            'optimizer_state_dict': {
                k: optimizers[k].state_dict() for k in optimizers},
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames
        }, checkpointpath)

        # Save per-position weights
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid,
                              position + '_weights_' + str(frames) + '.ckpt')))
            torch.save(learner_model.get_model(position).state_dict(),
                        model_weights_dir)

    fps_log = []
    timer = timeit.default_timer
    last_pool_save_frames = 0

    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {
                k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(5)

            # ---- Save pool snapshots periodically ----
            if flags.partner_random:
                if frames - last_pool_save_frames >= flags.pool_save_interval:
                    save_pool_snapshot(flags.pool_dir, learner_model, frames)
                    last_pool_save_frames = frames

            if timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = timer()
            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = {
                k: (position_frames[k] - position_start_frames[k]) /
                   (end_time - start_time)
                for k in position_frames
            }
            log.info(
                'After %i (L:%i U:%i D:%i) frames: @ %.1f fps '
                '(avg@ %.1f fps) (L:%.1f U:%.1f D:%.1f) Stats:\n%s',
                frames,
                position_frames['landlord'],
                position_frames['landlord_up'],
                position_frames['landlord_down'],
                fps, fps_avg,
                position_fps['landlord'],
                position_fps['landlord_up'],
                position_fps['landlord_down'],
                pprint.pformat(stats))

    except KeyboardInterrupt:
        return
    else:
        for thread in threads:
            thread.join()
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
    plogger.close()
