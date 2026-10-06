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
    def __init__(self, root, shard_size=512):
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
        name = f'frames_{len(self.shards):05d}.npy'
        np.save(self.root / name, np.stack(self.pending))
        self.shards.append({'file': name, 'start': self.count-len(self.pending), 'count': len(self.pending)})
        self.pending.clear()


def collect(c):
    from pixel_world_model.policy import load_c51_policy, greedy_action, resolve_policy_model
    target = Path(c['dataset'])
    if target.exists() and any(target.iterdir()):
        raise ValueError('Dataset directory is not empty. Select a new directory to preserve existing data.')
    target.mkdir(parents=True, exist_ok=True)
    cancel = Cancellation()
    rng = np.random.default_rng(c['seed'])
    device = device_for(c['device'])
    emit('device', device=str(device))
    policy_path = resolve_policy_model(c['policy_model'].removesuffix('.zip'))
    policy = load_c51_policy(policy_path, device)
    writer = FrameWriter(target)
    env = make_rollout_env()
    actions, states, nexts, episode_ids, terminals = [], [], [], [], []
    episodes = []
    start = time.monotonic()
    try:
        while len(actions) < c['transitions'] and not cancel.cancelled:
            eid = len(episodes)
            obs, _ = env.reset(seed=c['seed'] + eid)
            first = writer.add(capture_frame(env, c['resolution']))
            current = first
            begin = len(actions)
            for _ in range(min(c['episode_limit'], c['transitions']-begin)):
                if cancel.cancelled:
                    break
                action = int(rng.integers(10)) if rng.random() < c['exploration'] else greedy_action(policy, obs)
                obs, _, terminated, truncated, _ = env.step(action)
                nxt = writer.add(capture_frame(env, c['resolution']))
                actions.append(action); states.append(current); nexts.append(nxt)
                episode_ids.append(eid); terminals.append(terminated or truncated)
                current = nxt
                if len(actions) % 100 == 0:
                    progress(start, len(actions), c['transitions'], phase='collection')
                if terminated or truncated:
                    break
            if len(actions) > begin:
                episodes.append({'first_frame': first, 'start': begin, 'end': len(actions)})
            else:
                break
    finally:
        writer.flush()
        env.close()
    if len(episodes) < 2:
        raise ValueError('Need at least two trajectories for an episode-separated split. Collect more transitions or reduce episode_limit; select a new output directory.')
    order = rng.permutation(len(episodes))
    nval = min(len(episodes)-1, max(1, round(len(episodes)*c['validation_fraction'])))
    val = set(int(i) for i in order[:nval])
    for eid, ep in enumerate(episodes):
        ep['split'] = 'validation' if eid in val else 'train'
    np.savez(target / 'transitions.npz', actions=np.asarray(actions, dtype=np.int64),
             states=np.asarray(states), nexts=np.asarray(nexts), episodes=np.asarray(episode_ids),
             terminals=np.asarray(terminals, dtype=bool))
    signature = inspect.signature(CentipedeEnv.__init__)
    rewards = {k: p.default for k, p in signature.parameters.items() if k.startswith('reward_') or k == 'proximity_distance_tiles'}
    meta = {'version': VERSION, 'id': str(uuid.uuid4()), 'resolution': c['resolution'],
            'frame_stack': 4, 'frame_gap': 4, 'shards': writer.shards, 'episodes': episodes,
            'policy_model': policy_path, 'environment_defaults': rewards, 'config': c,
            'count': len(actions), 'cancelled': cancel.cancelled}
    (target / 'manifest.json').write_text(json.dumps(meta, indent=2))
    progress(start, len(actions), c['transitions'], phase='collection')
    emit('done', dataset=str(target), cancelled=cancel.cancelled)


class TransitionDataset(Dataset):
    def __init__(self, root, split=None):
        self.root = Path(root)
        self.meta = json.loads((self.root / 'manifest.json').read_text())
        if self.meta['version'] != VERSION or self.meta['frame_stack'] != 4 or self.meta['frame_gap'] != 4:
            raise ValueError('Incompatible dataset; recollect with this experiment.')
        with np.load(self.root / 'transitions.npz') as data:
            self.transitions = {key: data[key] for key in data.files}
        self.indices = np.asarray([i for i, eid in enumerate(self.transitions['episodes'])
                                   if split is None or self.meta['episodes'][int(eid)]['split'] == split], dtype=np.int64)
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
