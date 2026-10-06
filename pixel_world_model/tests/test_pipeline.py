"""Run with: python -m unittest discover -s pixel_world_model/tests -v"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from pixel_world_model.config import DEFAULTS, save, load, validate
from pixel_world_model.data import FrameWriter, TransitionDataset
from pixel_world_model.nets import VisualVAE, ActionLatentDynamics, reconstruction_loss
from pixel_world_model.runtime import VERSION, load_checkpoint, fingerprint
from pixel_world_model.train import train, load_vae
from pixel_world_model.visualize import predict_sequence, Inspection, action, draw


def fixture(root):
    root.mkdir()
    writer = FrameWriter(root, shard_size=3)
    episodes = []
    states, nexts, actions, ids, done = [], [], [], [], []
    for eid in range(3):
        first = writer.count
        for t in range(5):
            writer.add(np.full((64, 64), eid*50+t, dtype=np.uint8))
        begin = len(states)
        for t in range(4):
            states.append(first+t); nexts.append(first+t+1); actions.append(t)
            ids.append(eid); done.append(t == 3)
        episodes.append(dict(first_frame=first, start=begin, end=len(states), split='validation' if eid == 2 else 'train'))
    writer.flush()
    np.savez(root/'transitions.npz', states=states, nexts=nexts, actions=actions, episodes=ids, terminals=done)
    meta = dict(version=VERSION, id='fixture', resolution=64, frame_stack=4, frame_gap=4,
                shards=writer.shards, episodes=episodes, count=len(states))
    (root/'manifest.json').write_text(json.dumps(meta))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        fixture(self.root/'data')
        self.c = DEFAULTS | dict(dataset=str(self.root/'data'), device='cpu', latent_dim=8,
                                vae_epochs=1, dynamics_epochs=1, vae_batch=4, dynamics_batch=4,
                                hidden_width=16, checkpoint_frequency=1,
                                vae_output=str(self.root/'vae'), dynamics_output=str(self.root/'dynamics'))

    def tearDown(self):
        self.temp.cleanup()

    def test_four_game_frames_per_transition(self):
        from pixel_world_model.pixels import make_rollout_env
        env = make_rollout_env()
        env.reset(seed=0)
        try:
            with patch.object(env._engine, 'step', wraps=env._engine.step) as steps:
                env.step(0)
                self.assertEqual(steps.call_count, 4)
        finally:
            env.close()

    def test_stack_shards_boundaries_and_terminal(self):
        data = TransitionDataset(self.root/'data')
        self.assertEqual(data.raw(3)[0][:, 0, 0].tolist(), [0, 1, 2, 3])
        self.assertEqual(data.raw(4)[0][:, 0, 0].tolist(), [50]*4)
        self.assertEqual(data.raw(4)[2][:, 0, 0].tolist(), [50, 50, 50, 51])
        self.assertEqual(len(data.trajectory(3, 15)), 1)
        self.assertTrue(data.transitions['terminals'][3])
        train_ids = set(TransitionDataset(self.root/'data', 'train').transitions['episodes'][TransitionDataset(self.root/'data', 'train').indices])
        val_ids = set(data.transitions['episodes'][TransitionDataset(self.root/'data', 'validation').indices])
        self.assertFalse(train_ids & val_ids)

    def test_shapes_gradients_and_losses(self):
        for size in (64, 84, 128, 192, 256):
            model = VisualVAE(8, size)
            x = torch.rand(1, 4, size, size)
            recon, _, _ = model(x)
            self.assertEqual(recon.shape, x.shape)
            self.assertTrue(bool(((recon >= 0) & (recon <= 1)).all()))
            loss, metrics = model.loss(x)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(model.logvar.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.mu.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(np.isfinite(v) for v in metrics.values()))

    def test_reconstruction_is_bce_with_logits(self):
        target = torch.rand(2, 4, 16, 16)
        logits = torch.randn_like(target, requires_grad=True)
        loss, metrics = reconstruction_loss(logits, target)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        self.assertTrue(torch.equal(loss, expected))
        self.assertEqual(metrics['reconstruction'], loss.item())
        self.assertAlmostEqual(metrics['pixel_mse'], (logits.sigmoid()-target).square().mean().item())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_bce_saturated_missed_objects_have_gradients(self):
        for value, logit in ((0.0, 100.0), (1.0, -100.0), (0.0, -100.0), (1.0, 100.0)):
            target = torch.full((1, 4, 8, 8), value)
            logits = torch.full_like(target, logit, requires_grad=True)
            loss, _ = reconstruction_loss(logits, target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            if (value == 1 and logit < 0) or (value == 0 and logit > 0):
                self.assertAlmostEqual(logits.grad.abs().sum().item(), 1)

    def test_decode_and_forward_return_sigmoid_images(self):
        model = VisualVAE(8, 84)
        x = torch.rand(2, 4, 84, 84)
        with torch.no_grad():
            logits, mu, _ = model.forward_logits(x, sample=False)
            images, _, _ = model(x, sample=False)
            self.assertEqual(logits.shape, x.shape)
            self.assertTrue(torch.equal(images, logits.sigmoid()))
            self.assertTrue(torch.equal(model.decode(mu), model.decode_logits(mu).sigmoid()))

    def test_resume_previous_loss_resets_best(self):
        train(self.c, 'vae')
        checkpoint = self.root/'vae/final.pt'
        ck = load_checkpoint(checkpoint, 'vae')
        for old_loss in (None, 'pixel_mse_v1', 'region_balanced_v1'):
            with self.subTest(old_loss=old_loss):
                ck['reconstruction_loss'] = old_loss
                ck['best'] = -1
                torch.save(ck, checkpoint)
                output = self.root/str(old_loss)
                train(self.c | dict(vae_resume=str(checkpoint), vae_epochs=2,
                                   vae_output=str(output)), 'vae')
                resumed = load_checkpoint(output/'best.pt', 'vae')
                self.assertEqual(resumed['epoch'], 2)
                self.assertEqual(resumed['reconstruction_loss'], 'bce_logits_v1')
                self.assertGreater(resumed['best'], 0)

    def test_delta_and_one_hot(self):
        model = ActionLatentDynamics(8, 16, 2)
        z = torch.randn(10, 8)
        captured = []
        handle = model.input.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        with torch.no_grad():
            model.delta.weight.zero_(); model.delta.bias.zero_()
        self.assertTrue(torch.equal(model(z, torch.arange(10)), z))
        self.assertTrue(torch.equal(captured[0][:, 8:], torch.eye(10)))
        handle.remove()

    def test_stages_freezing_cache_and_inspection(self):
        train(self.c, 'vae')
        path = self.root/'vae/final.pt'
        before = fingerprint(path)
        self.c['vae_checkpoint'] = str(path)
        captured = []
        def capture_vae(*args):
            model, ckpt = load_vae(*args)
            captured.append((model, {k: v.clone() for k, v in model.state_dict().items()}))
            return model, ckpt
        with patch('pixel_world_model.train.load_vae', side_effect=capture_vae):
            train(self.c, 'dynamics')
        frozen, initial_weights = captured[0]
        for key, value in frozen.state_dict().items():
            self.assertTrue(torch.equal(value, initial_weights[key]), key)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in frozen.parameters()))
        self.assertEqual(before, fingerprint(path))
        vae, _ = load_vae(path)
        self.assertFalse(any(p.requires_grad or p.grad is not None for p in vae.parameters()))
        ck = load_checkpoint(self.root/'dynamics/final.pt', 'dynamics')
        torch.manual_seed(self.c['seed'])
        initial = ActionLatentDynamics(**ck['architecture'])
        self.assertTrue(any(not torch.equal(v, initial.state_dict()[k]) for k, v in ck['model'].items()))
        self.assertTrue((self.root/'data/latent_cache'/before/'pairs.npy').exists())
        self.c['dynamics_checkpoint'] = str(self.root/'dynamics/final.pt')
        view = Inspection(self.c, 'dynamics')
        count = 0
        def hook(*args):
            nonlocal count
            count += 1
        handle = vae.encoder.register_forward_hook(hook)
        raw = TransitionDataset(self.root/'data').raw(0)[0]
        latents, decoded = predict_sequence(vae, initial, raw, [0, 1, 2], 'cpu')
        handle.remove()
        self.assertEqual(count, 1)
        self.assertEqual(len(decoded), 4)
        action(view, 'next'); self.assertEqual(view.step, 1)
        self.assertEqual(view.actions, [0, 1, 2, 3])
        action(view, 'reset'); self.assertEqual(view.step, 0)
        action(view, 'frame'); self.assertEqual(view.channel, 0)
        action(view, 'more'); self.assertEqual(view.horizon, 16)
        self.assertEqual(len(view.panels()[0]), 4)
        self.assertEqual(len(Inspection(self.c, 'vae').panels()[0]), 3)
        associated = self.root/'dynamics/associated_vae.pt'
        associated.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Associated VAE'):
            Inspection(self.c, 'dynamics')

    def test_resume_matches_uninterrupted(self):
        original = self.c | dict(vae_epochs=2)
        train(original, 'vae')
        expected = load_checkpoint(self.root/'vae/final.pt', 'vae')
        train(self.c | dict(vae_resume=str(self.root/'vae/epoch_0001.pt'), vae_epochs=2,
                           vae_output=str(self.root/'resumed')), 'vae')
        actual = load_checkpoint(self.root/'resumed/final.pt', 'vae')
        self.assertEqual(actual['updates'], expected['updates'])
        for key in actual['model']:
            self.assertTrue(torch.equal(actual['model'][key], expected['model'][key]), key)

    def test_cancel_mid_update_and_resume(self):
        from pixel_world_model.runtime import Cancellation
        cancel = Cancellation()
        original = torch.optim.Adam.step
        calls = 0
        def step(optim, *args, **kwargs):
            nonlocal calls
            result = original(optim, *args, **kwargs)
            calls += 1
            if calls == 1:
                cancel.cancelled = True
            return result
        with patch('pixel_world_model.train.Cancellation', return_value=cancel), patch.object(torch.optim.Adam, 'step', step):
            train(self.c, 'vae')
        ck = load_checkpoint(self.root/'vae/cancelled.pt', 'vae')
        self.assertEqual(ck['cursor'], 4)
        self.assertEqual(ck['updates'], 1)
        train(self.c | dict(vae_resume=str(self.root/'vae/cancelled.pt')), 'vae')
        self.assertEqual(load_checkpoint(self.root/'vae/final.pt', 'vae')['updates'], 2)

    def test_configuration_validation(self):
        for key, value in [('beta', float('nan')), ('vae_lr', float('inf')), ('blocks', 4), ('seed', -1), ('resolution', 100), ('horizon', 101)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(self.c | {key: value})

    def test_incompatible_checkpoint_and_settings(self):
        torch.save({'model': {}}, self.root/'legacy.pt')
        with self.assertRaisesRegex(ValueError, 'legacy'):
            load_checkpoint(self.root/'legacy.pt', 'vae')
        train(self.c, 'vae')
        self.c['vae_checkpoint'] = str(self.root/'vae/final.pt')
        meta_path = self.root/'data/manifest.json'
        meta = json.loads(meta_path.read_text()); meta['id'] = 'different'
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'match'):
            train(self.c, 'dynamics')
        save(self.c, self.root/'settings.json')
        self.assertEqual(load(self.root/'settings.json'), self.c)


if __name__ == '__main__':
    unittest.main()
