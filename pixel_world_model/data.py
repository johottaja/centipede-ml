"""Sharded frame storage. Stacks are reconstructed without crossing episode boundaries."""
import inspect
import json
import time
import uuid
from bisect import bisect_right
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset
from core.env import CentipedeEnv
from pixel_world_model.pixels import make_rollout_env, capture_frame
from pixel_world_model.runtime import Cancellation, device_for, emit, progress, VERSION


class FrameWriter:
    def __init__(self, root, shard_size=512, prefix="frames"):
        self.prefix = prefix
        self.root, self.size = Path(root), shard_size
        self.pending, self.shards, self.count = [], [], 0

    def add(self, frame):
        index = self.count
        self.pending.append(frame)
        self.count += 1
        if len(self.pending) == self.size:
            self.flush()
        return index

    def flush(self):
        if not self.pending:
            return
        name = f'{self.prefix}_{len(self.shards):05d}.npy'
        np.save(self.root / name, np.stack(self.pending))
        self.shards.append({'file': name, 'start': self.count-len(self.pending), 'count': len(self.pending)})
        self.pending.clear()


def stratified_indices(start, end, limit, rng):
    """One uniformly chosen transition per equal-duration bin, without duplicates."""
    count = min(limit, end-start)
    if count <= 0:
        return []
    edges = np.linspace(start, end, count+1, dtype=np.int64)
    return [int(rng.integers(left, right)) for left, right in zip(edges[:-1], edges[1:], strict=True)]


def training_indices(start, end, c, rng):
    """Choose diverse VAE examples or every consecutive dynamics transition."""
    if c.get('collection_mode', 'sampled') == 'consecutive':
        return list(range(start, end))
    return stratified_indices(start, end, c['samples_per_episode'], rng)


def episode_splits(episodes, validation_episodes, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    validation = set(int(i) for i in rng.permutation(episodes)[:validation_episodes])
    return ['validation' if i in validation else 'train' for i in range(episodes)]


def _rollout_worker(pipe, target, resolution, limit):
    """CPU-only spawned worker; frame files never travel over IPC."""
    import traceback
    torch.set_num_threads(1)
    env = None
    try:
        env = make_rollout_env()
        while True:
            command, value = pipe.recv()
            if command == 'close':
                break
            if command == 'reset':
                eid, seed = value
                writer = FrameWriter(target, prefix=f'episode_{eid:06d}_frames')
                actions, terminals = [], []
                obs, _ = env.reset(seed=seed)
                writer.add(capture_frame(env, resolution))
                pipe.send(('ok', (obs, None)))
            elif command in ('step', 'finish'):
                done = command == 'finish'
                reason = 'cancelled'
                obs = None
                if command == 'step':
                    obs, _, terminated, truncated, _ = env.step(value)
                    writer.add(capture_frame(env, resolution))
                    actions.append(value)
                    terminals.append(terminated or truncated)
                    done = terminated or truncated or len(actions) >= limit
                    reason = 'terminal' if terminated else 'truncated' if truncated else 'safety_cap'
                result = None
                if done:
                    writer.flush()
                    result = (eid, writer.shards, actions, terminals, reason)
                pipe.send(('ok', (obs, result)))
    except BaseException:
        try:
            pipe.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if env is not None:
            env.close()
        pipe.close()


def parallel_episodes(c, target, policy, cancel, report):
    """Batch inference centrally, and simulate/render/write in spawned processes."""
    import multiprocessing as mp
    from pixel_world_model.policy import greedy_actions
    context = mp.get_context('spawn')
    workers, active = [], {}
    next_eid = 0
    played = 0
    last_report = time.monotonic()

    def receive(pipe):
        status, value = pipe.recv()
        if status == 'error':
            raise RuntimeError('Collection worker failed:\n' + value)
        return value

    def assign(slot):
        nonlocal next_eid
        eid = next_eid
        next_eid += 1
        pipe = workers[slot][0]
        pipe.send(('reset', (eid, c['seed'] + eid)))
        obs, _ = receive(pipe)
        active[slot] = [eid, obs, np.random.default_rng(np.random.SeedSequence([c['seed'], eid, 3]))]

    try:
        for slot in range(min(c['collection_workers'], c['episodes'])):
            parent, child = context.Pipe()
            process = context.Process(target=_rollout_worker,
                                      args=(child, str(target), c['resolution'], c['episode_limit']))
            process.start()
            child.close()
            workers.append((parent, process))
            assign(slot)
        while active:
            slots = list(active)
            if cancel.cancelled:
                for slot in slots:
                    workers[slot][0].send(('finish', None))
            else:
                chosen, greedy_slots = {}, []
                for slot in slots:
                    rng = active[slot][2]
                    if rng.random() < c['exploration']:
                        chosen[slot] = int(rng.integers(10))
                    else:
                        greedy_slots.append(slot)
                if greedy_slots:
                    predicted = greedy_actions(policy, np.stack([active[slot][1] for slot in greedy_slots]))
                    chosen.update(zip(greedy_slots, map(int, predicted), strict=True))
                for slot in slots:
                    workers[slot][0].send(('step', chosen[slot]))
            if not cancel.cancelled:
                played += len(slots)
            if time.monotonic() - last_report >= 1:
                report(played)
                last_report = time.monotonic()
            for slot in slots:
                obs, result = receive(workers[slot][0])
                if result is None:
                    active[slot][1] = obs
                else:
                    del active[slot]
                    yield result
                    if not cancel.cancelled and next_eid < c['episodes']:
                        assign(slot)
    finally:
        for pipe, process in workers:
            try:
                pipe.send(('close', None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for pipe, process in workers:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            pipe.close()


def collect(c):
    from pixel_world_model.policy import load_c51_policy, greedy_action, resolve_policy_model
    target = Path(c['dataset'])
    if target.exists() and any(target.iterdir()):
        raise ValueError('Dataset directory is not empty. Select a new directory to preserve existing data.')
    target.mkdir(parents=True, exist_ok=True)
    cancel = Cancellation()
    rng = np.random.default_rng(c['seed'])
    sampling_rng = np.random.default_rng(np.random.SeedSequence([c['seed'], 1]))
    splits = episode_splits(c['episodes'], c['validation_episodes'], c['seed'])
    device = device_for(c['device'])
    emit('device', device=str(device))
    policy_path = resolve_policy_model(c['policy_model'].removesuffix('.zip'))
    policy = load_c51_policy(policy_path, device)
    writer = FrameWriter(target)
    env = make_rollout_env() if c.get('collection_workers', 1) == 1 else None
    actions, states, nexts, episode_ids, terminals = [], [], [], [], []
    episodes, sampled_indices = [], []
    start = time.monotonic()
    seed = c['seed']
    emit('start', phase='collection', total=c['episodes'],
         training_episodes=c['episodes']-c['validation_episodes'], validation_episodes=c['validation_episodes'])

    reported_played = 0

    def report(played=None):
        nonlocal reported_played
        reported_played = max(reported_played, len(actions) if played is None else played)
        progress(start, len(episodes), c['episodes'], phase='collection',
                 played_transitions=reported_played, sampled_transitions=len(sampled_indices),
                 episode_seed=seed)

    try:
        if env is None:
            results = parallel_episodes(c, target, policy, cancel, report)
            try:
                for eid, shards, episode_actions, episode_terminals, reason in results:
                    if not episode_actions:
                        for shard in shards:
                            (target / shard['file']).unlink()
                        continue
                    seed = c['seed'] + eid
                    first, begin = writer.count, len(actions)
                    for shard in shards:
                        writer.shards.append(shard | {'start': shard['start'] + first})
                        writer.count += shard['count']
                    count = len(episode_actions)
                    actions.extend(episode_actions)
                    states.extend(range(first, first + count))
                    nexts.extend(range(first + 1, first + count + 1))
                    episode_ids.extend([len(episodes)] * count)
                    terminals.extend(episode_terminals)
                    sample_rng = np.random.default_rng(np.random.SeedSequence([c['seed'], eid, 1]))
                    selected = training_indices(begin, len(actions), c, sample_rng)
                    sampled_indices.extend(selected)
                    episodes.append({'first_frame': first, 'start': begin, 'end': len(actions),
                                     'seed': seed, 'split': splits[eid], 'sample_count': len(selected),
                                     'ended_by': reason})
                    report()
            finally:
                results.close()
        else:
            for eid in range(c['episodes']):
                if cancel.cancelled:
                    break
                seed = c['seed'] + eid
                obs, _ = env.reset(seed=seed)
                first = writer.add(capture_frame(env, c['resolution']))
                current = first
                begin = len(actions)
                terminated = truncated = False
                for _ in range(c['episode_limit']):
                    if cancel.cancelled:
                        break
                    action = int(rng.integers(10)) if rng.random() < c['exploration'] else greedy_action(policy, obs)
                    obs, _, terminated, truncated, _ = env.step(action)
                    nxt = writer.add(capture_frame(env, c['resolution']))
                    actions.append(action)
                    states.append(current)
                    nexts.append(nxt)
                    episode_ids.append(eid)
                    terminals.append(terminated or truncated)
                    current = nxt
                    if len(actions) % 100 == 0:
                        report()
                    if terminated or truncated:
                        break
                if len(actions) > begin:
                    selected = training_indices(begin, len(actions), c, sampling_rng)
                    sampled_indices.extend(selected)
                    episodes.append({'first_frame': first, 'start': begin, 'end': len(actions),
                                     'seed': seed, 'split': splits[eid], 'sample_count': len(selected),
                                     'ended_by': 'terminal' if terminated else 'truncated' if truncated else
                                                 'cancelled' if cancel.cancelled else 'safety_cap'})
                    report()
                else:
                    break
    finally:
        writer.flush()
        if env is not None:
            env.close()
    if len(episodes) < 2:
        raise ValueError('Need at least two episodes for validation. Collect again into a new directory.')
    # A cancelled prefix may contain only one of the preassigned splits.
    if cancel.cancelled and len({ep['split'] for ep in episodes}) == 1:
        replacement = 'validation' if episodes[0]['split'] == 'train' else 'train'
        episodes[-1]['split'] = replacement
        emit('log', text='Partial collection: reassigned the last episode to retain both splits.')
    np.savez(target / 'transitions.npz', actions=np.asarray(actions, dtype=np.int64),
             states=np.asarray(states), nexts=np.asarray(nexts), episodes=np.asarray(episode_ids),
             terminals=np.asarray(terminals, dtype=bool), sampled_indices=np.asarray(sampled_indices, dtype=np.int64))
    signature = inspect.signature(CentipedeEnv.__init__)
    rewards = {k: p.default for k, p in signature.parameters.items() if k.startswith('reward_') or k == 'proximity_distance_tiles'}
    meta = {'version': VERSION, 'id': str(uuid.uuid4()), 'resolution': c['resolution'],
            'frame_stack': 4, 'frame_gap': 4, 'shards': writer.shards, 'episodes': episodes,
            'policy_model': policy_path, 'environment_defaults': rewards, 'config': c,
            'count': len(actions), 'sampled_count': len(sampled_indices),
            'collection_mode': c.get('collection_mode', 'sampled'),
            'sampling': 'consecutive_v1' if c.get('collection_mode') == 'consecutive' else 'stratified_episode_v1', 'cancelled': cancel.cancelled}
    (target / 'manifest.json').write_text(json.dumps(meta, indent=2))
    report()
    emit('done', dataset=str(target), cancelled=cancel.cancelled, played_transitions=len(actions),
         sampled_transitions=len(sampled_indices),
         training_episodes=sum(ep['split'] == 'train' for ep in episodes),
         validation_episodes=sum(ep['split'] == 'validation' for ep in episodes))


class TransitionDataset(Dataset):
    def __init__(self, root, split=None):
        self.root = Path(root)
        self.meta = json.loads((self.root / 'manifest.json').read_text())
        if self.meta['version'] != VERSION or self.meta['frame_stack'] != 4 or self.meta['frame_gap'] != 4:
            raise ValueError('Incompatible dataset; recollect with this experiment.')
        with np.load(self.root / 'transitions.npz') as data:
            self.transitions = {key: data[key] for key in data.files}
        eligible = self.transitions.get('sampled_indices', np.arange(len(self.transitions['actions'])))
        self.sample_indices = np.asarray(eligible, dtype=np.int64)
        if (len(self.sample_indices) == 0 or np.any(np.diff(self.sample_indices) <= 0) or
                self.sample_indices[0] < 0 or self.sample_indices[-1] >= len(self.transitions['actions'])):
            raise ValueError('Dataset sampled indices are invalid; recollect into a new directory.')
        self.indices = np.asarray([i for i in self.sample_indices
                                   if split is None or self.meta['episodes'][int(self.transitions['episodes'][i])]['split'] == split], dtype=np.int64)
        self.starts = [shard['start'] for shard in self.meta['shards']]
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.indices)

    def frame(self, index):
        shard = bisect_right(self.starts, index)-1
        if shard not in self.cache:
            self.cache[shard] = np.load(self.root / self.meta['shards'][shard]['file'], mmap_mode='r')
            if len(self.cache) > 4:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        return self.cache[shard][index-self.starts[shard]]

    def stack(self, frame, episode):
        first = self.meta['episodes'][episode]['first_frame']
        return np.stack([self.frame(max(first, frame-offset)) for offset in (3, 2, 1, 0)])

    def raw(self, index):
        eid = int(self.transitions['episodes'][index])
        return (self.stack(int(self.transitions['states'][index]), eid),
                int(self.transitions['actions'][index]),
                self.stack(int(self.transitions['nexts'][index]), eid))

    def __getitem__(self, index):
        s, a, ns = self.raw(int(self.indices[index]))
        return torch.from_numpy(s.astype(np.float32)/255), a, torch.from_numpy(ns.astype(np.float32)/255)

    def trajectory(self, start, horizon):
        eid = int(self.transitions['episodes'][start])
        end = min(start+horizon, self.meta['episodes'][eid]['end'])
        return [self.raw(i) for i in range(start, end)]
