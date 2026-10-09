"""Collection sampling and compact latent-cache regression tests."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from pixel_world_model.config import DEFAULTS, validate
from pixel_world_model.data import (collect, TransitionDataset, stratified_indices, episode_splits)
from pixel_world_model.train import train, load_checkpoint, load_vae
from pixel_world_model.visualize import Inspection


class FakeEnv:
    def __init__(self, length=43):
        self.length = length
        self.seeds = []
        self.closed = False

    def reset(self, seed):
        self.seeds.append(seed)
        self.time = 0
        return np.zeros(1), {}

    def step(self, action):
        self.time += 1
        return np.zeros(1), 0, self.time == self.length, False, {}

    def close(self):
        self.closed = True


class SamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.c = DEFAULTS | dict(dataset=str(self.root/'data'), episodes=5,
                                validation_episodes=1, samples_per_episode=4, collection_workers=1,
                                episode_limit=100, resolution=64, device='cpu', seed=1000)

    def tearDown(self):
        self.temp.cleanup()

    def collect_fake(self, length=43, c=None):
        env = FakeEnv(length)
        def capture(env, resolution):
            return np.full((resolution, resolution), env.time, dtype=np.uint8)
        with patch('pixel_world_model.data.make_rollout_env', return_value=env), \
             patch('pixel_world_model.data.capture_frame', side_effect=capture), \
             patch('pixel_world_model.policy.resolve_policy_model', return_value='fake'), \
             patch('pixel_world_model.policy.load_c51_policy', return_value=object()), \
             patch('pixel_world_model.policy.greedy_action', return_value=9):
            collect(c or self.c)
        return env

    def test_bins_reproducible_and_short_episodes_no_duplicates(self):
        first = stratified_indices(100, 113, 5, np.random.default_rng(4))
        self.assertEqual(first, stratified_indices(100, 113, 5, np.random.default_rng(4)))
        edges = np.linspace(100, 113, 6, dtype=int)
        for sample, low, high in zip(first, edges[:-1], edges[1:]):
            self.assertTrue(low <= sample < high)
        self.assertEqual(len(set(first)), 5)
        self.assertEqual(stratified_indices(10, 13, 100, np.random.default_rng(0)), [10, 11, 12])
        self.assertEqual(stratified_indices(0, 0, 100, np.random.default_rng(0)), [])

    def test_split_exact_and_independent(self):
        split = episode_splits(500, 50, 0)
        self.assertEqual(split.count('train'), 450)
        self.assertEqual(split.count('validation'), 50)
        self.assertEqual(split, episode_splits(500, 50, 0))
        self.assertNotEqual(split, episode_splits(500, 50, 1))

    def test_collection_seeds_sample_counts_and_consecutive_context(self):
        env = self.collect_fake()
        self.assertEqual(env.seeds, list(range(1000, 1005)))
        self.assertTrue(env.closed)
        data = TransitionDataset(self.c['dataset'])
        self.assertEqual(data.meta['count'], 5*43)
        self.assertEqual(len(data), 5*4)
        self.assertEqual(len(TransitionDataset(self.c['dataset'], 'train')), 16)
        self.assertEqual(len(TransitionDataset(self.c['dataset'], 'validation')), 4)
        for ep in data.meta['episodes']:
            selected = data.indices[(data.indices >= ep['start']) & (data.indices < ep['end'])]
            self.assertEqual(len(selected), 4)
            self.assertEqual(ep['ended_by'], 'terminal')
            self.assertTrue(data.transitions['terminals'][ep['end']-1])
            edges = np.linspace(ep['start'], ep['end'], 5, dtype=int)
            for index, low, high in zip(selected, edges[:-1], edges[1:]):
                self.assertTrue(low <= index < high)
                state, _, nxt = data.raw(int(index))
                local = index-ep['start']
                self.assertEqual(state[:, 0, 0].tolist(), [max(0, int(local)-i) for i in (3, 2, 1, 0)])
                self.assertEqual(nxt[:, 0, 0].tolist(), [max(0, int(local)+1-i) for i in (3, 2, 1, 0)])
            sequence = data.trajectory(int(selected[0]), 15)
            self.assertEqual(len(sequence), 15)
            self.assertTrue(np.array_equal(sequence[0][2], sequence[1][0]))
            self.assertEqual(len(data.trajectory(ep['end']-1, 15)), 1)
        # Altering exploration consumption does not change the preassigned split.
        self.assertEqual([ep['split'] for ep in data.meta['episodes']], episode_splits(5, 1, 1000))

    def test_short_and_capped_episodes(self):
        self.collect_fake(length=3)
        data = TransitionDataset(self.c['dataset'])
        self.assertEqual(len(data), 15)
        self.assertEqual(data.meta['sampled_count'], 15)
        other = self.c | dict(dataset=str(self.root/'capped'), episode_limit=7)
        self.collect_fake(c=other)
        capped = TransitionDataset(other['dataset'])
        self.assertTrue(all(ep['ended_by'] == 'safety_cap' for ep in capped.meta['episodes']))
        self.assertFalse(capped.transitions['terminals'].any())

    def test_sparse_dataset_both_trainers_and_compact_cache(self):
        self.collect_fake()
        c = self.c | dict(latent_dim=8, hidden_width=16, vae_epochs=1, dynamics_epochs=1,
                          vae_batch=4, dynamics_batch=4,
                          vae_output=str(self.root/'vae'), dynamics_output=str(self.root/'dynamics'))
        train(c, 'vae')
        ck = load_checkpoint(self.root/'vae/final.pt', 'vae')
        self.assertEqual(ck['updates'], 4)  # 16 selected training examples, not 172 played transitions
        c['vae_checkpoint'] = str(self.root/'vae/final.pt')
        train(c, 'dynamics')
        dyn = load_checkpoint(self.root/'dynamics/final.pt', 'dynamics')
        self.assertEqual(dyn['updates'], 4)
        cache = next((self.root/'data/latent_cache').glob('*/pairs.npy'))
        pairs = np.load(cache, mmap_mode='r')
        self.assertEqual(pairs.shape, (20, 2, 8))
        data = TransitionDataset(c['dataset'])
        vae, _ = load_vae(c['vae_checkpoint'])
        position = 7
        with torch.no_grad():
            raw = data.raw(int(data.sample_indices[position]))
            target = torch.tensor(raw[2], dtype=torch.float32)[None]/255
            expected = vae.encode(target)[0][0].numpy()
        np.testing.assert_allclose(pairs[position, 1], expected, atol=1e-6)
        c['dynamics_checkpoint'] = str(self.root/'dynamics/final.pt')
        view = Inspection(c, 'dynamics')
        self.assertEqual(len(view.data), 4)
        self.assertEqual(len(view.trajectory), min(15, data.meta['episodes'][int(data.transitions['episodes'][view.index])]['end']-view.index))

    def test_spawned_collection_preserves_context_and_worker_independent_seeds(self):
        datasets = []
        for workers in (2, 3):
            c = self.c | dict(collection_workers=workers, exploration=1.0,
                              episode_limit=4, dataset=str(self.root/f'parallel{workers}'))
            with patch('pixel_world_model.policy.resolve_policy_model', return_value='fake'), \
                 patch('pixel_world_model.policy.load_c51_policy', return_value=object()):
                collect(c)
            data = TransitionDataset(c['dataset'])
            self.assertEqual(data.meta['count'], 20)
            self.assertEqual(len(data), 20)
            self.assertEqual(len(TransitionDataset(c['dataset'], 'validation')), 4)
            by_seed = {}
            for eid, ep in enumerate(data.meta['episodes']):
                self.assertEqual(ep['ended_by'], 'safety_cap')
                self.assertEqual(ep['split'], episode_splits(5, 1, 1000)[ep['seed']-1000])
                self.assertTrue(np.all(data.transitions['episodes'][ep['start']:ep['end']] == eid))
                first, _, _ = data.raw(ep['start'])
                self.assertTrue(all(np.array_equal(first[0], frame) for frame in first))
                for i in range(ep['start'], ep['end']-1):
                    self.assertTrue(np.array_equal(data.raw(i)[2], data.raw(i+1)[0]))
                by_seed[ep['seed']] = [data.raw(i) for i in range(ep['start'], ep['end'])]
            datasets.append(by_seed)
        for seed in datasets[0]:
            for left, right in zip(datasets[0][seed], datasets[1][seed], strict=True):
                self.assertEqual(left[1], right[1])
                np.testing.assert_array_equal(left[0], right[0])
                np.testing.assert_array_equal(left[2], right[2])

    def test_parallel_cancellation_flushes_active_episodes(self):
        from pixel_world_model.data import parallel_episodes
        from types import SimpleNamespace
        target = self.root/'cancelled'
        target.mkdir()
        cancel = SimpleNamespace(cancelled=False)
        c = self.c | dict(collection_workers=2, exploration=1.0, episode_limit=10000)
        results = list(parallel_episodes(c, target, object(), cancel,
                                        lambda *_: setattr(cancel, 'cancelled', True)))
        self.assertEqual(len(results), 2)
        self.assertTrue(all(ep[-1] == 'cancelled' for ep in results))
        for _, shards, actions, terminals, _ in results:
            self.assertEqual(sum(shard['count'] for shard in shards), len(actions)+1)
            self.assertEqual(len(actions), len(terminals))
            for shard in shards:
                self.assertTrue((target/shard['file']).is_file())

    def test_worker_errors_are_propagated(self):
        from pixel_world_model.data import parallel_episodes
        from types import SimpleNamespace
        target = self.root/'failed'
        c = self.c | dict(collection_workers=2, exploration=1.0, episode_limit=1)
        with self.assertRaisesRegex(RuntimeError, 'Collection worker failed'):
            list(parallel_episodes(c, target, object(), SimpleNamespace(cancelled=False), lambda *_: None))

    def test_batched_policy_inference(self):
        from pixel_world_model.policy import greedy_actions
        from types import SimpleNamespace
        model = SimpleNamespace(policy=SimpleNamespace(obs_to_tensor=lambda obs: (torch.tensor(obs), True)),
                                q_net=lambda obs: obs)
        np.testing.assert_array_equal(greedy_actions(model, np.array([[0, 3, 1], [4, 2, 0]])), [1, 0])

    def test_consecutive_mode_selects_every_transition(self):
        c = self.c | dict(collection_mode='consecutive')
        self.collect_fake(length=7, c=c)
        data = TransitionDataset(c['dataset'])
        self.assertEqual(len(data), 35)
        np.testing.assert_array_equal(data.sample_indices, np.arange(35))
        self.assertEqual(data.meta['collection_mode'], 'consecutive')
        self.assertEqual(len(TransitionDataset(c['dataset'], 'train')), 28)
        self.assertEqual(len(TransitionDataset(c['dataset'], 'validation')), 7)

    def test_configuration_split_bounds(self):
        for key, value in [('collection_mode', 'unknown'), ('collection_workers', 0), ('episodes', 0), ('validation_episodes', 0),
                           ('validation_episodes', 500), ('samples_per_episode', 0)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(DEFAULTS | {key: value})


if __name__ == '__main__':
    unittest.main()
