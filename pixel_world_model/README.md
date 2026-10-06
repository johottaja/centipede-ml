# Two-stage pixel world model

Launch the independent Tkinter experiment:

```sh
uv run python -m pixel_world_model
```

1. **Data:** choose a C51 `.zip` policy and a new, empty dataset directory. Collect trajectories with configurable random actions. The default is 50,000 transitions, 128×128 grayscale, four stacked frames, four game frames per action, and 10% validation trajectories.
2. **Step 1: VAE:** train, then choose `best.pt`, a periodic checkpoint, or `final.pt` and inspect reconstructions. The launcher opens a live curve of the running training objective and held-out validation loss. Inspect all four stack frames; confirm blaster, bullets, and centipede segments are resolved. Change the budget/settings and train again if necessary.
3. **Step 2: Dynamics:** explicitly select the VAE checkpoint, then train the residual latent predictor. Inspect held-out autoregressive predictions, initially over 15 steps.

All settings live in this folder's `settings.json`. The root settings and main launcher are independent. Shared game/C51 code is imported; game environment constructor defaults are recorded in the dataset. Reward shaping does not enter either world-model loss.

## CLI

CLI flags override this experiment's settings. `--settings PATH` selects another experimental configuration.

```sh
uv run python -m pixel_world_model collect --policy-model models/dqn_centipede --dataset pixel_world_model/data/run1
uv run python -m pixel_world_model train-vae --dataset pixel_world_model/data/run1 --vae-output pixel_world_model/checkpoints/vae_run1
uv run python -m pixel_world_model inspect-vae --dataset pixel_world_model/data/run1 --vae-checkpoint pixel_world_model/checkpoints/vae_run1/best.pt
uv run python -m pixel_world_model train-dynamics --dataset pixel_world_model/data/run1 --vae-checkpoint pixel_world_model/checkpoints/vae_run1/best.pt --dynamics-output pixel_world_model/checkpoints/dynamics_run1
uv run python -m pixel_world_model inspect-dynamics --dataset pixel_world_model/data/run1 --dynamics-checkpoint pixel_world_model/checkpoints/dynamics_run1/best.pt
```

Use `--help` for all configuration options. Paths entered in the launcher resolve against the project directory; CLI relative paths resolve against its working directory. CPU, CUDA, and MPS are selectable. `auto` prefers CUDA, then MPS, then CPU.

## Training and artifacts

Frames are uint8 `.npy` shards (512 frames each). Transition indices/actions and episode boundaries are saved separately, with a manifest recording preprocessing and collection settings. Reset frames repeat to fill the initial stack; subsequent channels are separated by four game frames. Stacks and rollouts never cross resets. Long games are cut into configurable maximum-length trajectories so a held-out split is always possible; the final collection budget may cut a trajectory short. True terminal flags remain distinct from these artificial boundaries.

### Cloud runs and TensorBoard

Each training output directory contains `run_config.json`, checkpoints, and TensorBoard event files under `tensorboard/`. Scalar logs include training and validation objectives plus VAE reconstruction BCE, pixel MSE, KL, or dynamics latent MSE. On each new best validation checkpoint, VAE runs log fixed held-out input/reconstruction/difference images; dynamics runs log fixed one-step actual/reconstructed/predicted comparisons. Logs flush after validation so curves remain available during cloud runs and after restart.

With the default VAE output directory, start TensorBoard on the remote machine:

```sh
uv run tensorboard --logdir pixel_world_model/checkpoints/vae/tensorboard --host 127.0.0.1 --port 6006
```

From your local machine, forward the port and open `http://localhost:6006`:

```sh
ssh -N -L 6006:127.0.0.1:6006 user@cloud-host
```

Use the configured `vae_output` or `dynamics_output` path when it differs from the default. Keep those output directories on persistent cloud storage if you need the curves, images, config, and checkpoints after the compute instance stops.

VAE input/output is `[0,1]`. Its loss is unweighted mean `BCEWithLogitsLoss` + beta × mean KL. BCE is averaged over batch, stack channels, and pixels; KL is averaged over batch and latent dimensions. The decoder produces raw logits for training and applies sigmoid when reconstructing/displaying images. Grayscale targets remain in `[0,1]` without binarization. Progress and inspection report reconstruction BCE and ordinary pixel MSE separately. BCE values cannot be compared numerically with previous MSE loss values; the KL beta is unchanged. Foreground weighting has been removed. A low global MSE can still hide missing small objects; inspect reconstructions rather than treating it as accuracy. Supported square resolutions are 64, 84, 128, 192, and 256. Nonmultiples of 16 use decoder interpolation to recover the requested output size.

Dynamics uses the deterministic VAE mean and next-latent MSE. Encoder and decoder are evaluation-only and frozen. Latent pairs are cached by dataset directory and VAE SHA-256 identity. Each dynamics run pins a byte-identical `associated_vae.pt` beside its checkpoints, so inspection cannot accidentally use a subsequently replaced VAE.

Checkpoints include model, optimizer, architecture, preprocessing, dataset identity, epoch/update cursor, batch ordering, and RNG state. `best.pt` uses deterministic held-out validation; `epoch_NNNN.pt` is periodic; `final.pt` completes the configured total epoch budget. Cancelling saves `cancelled.pt` after the current update. Choose a resume file and a larger **total** epoch budget to continue; architecture, dataset, VAE, batch size, and beta must match. Learning rate may be changed when resuming. Resuming an MSE or foreground-weighted checkpoint switches to BCE with logits and resets the best-validation comparison while preserving model, optimizer, and update progress. Use a larger total epoch budget and preferably a new output directory; retrain dynamics after improving the VAE because latent caches and dynamics checkpoints are tied to its exact weights.

Use a new output directory for a different experiment. The original `checkpoints/world_model.pt` is preserved but incompatible with this architecture; no automatic conversion occurs. Datasets/caches/checkpoints are gitignored by this folder's `.gitignore`.

## Inspection

Pygame windows show held-out data, with mouse controls and keyboard shortcuts:

- Space: play/pause; Left/Right: previous/next sample or rollout step.
- 1–4: select a stack frame, oldest to newest.
- `[` / `]`: shorten/extend the rollout horizon (1–100).
- N: next trajectory; R: reset current sequence; Q/Escape: close.

VAE inspection compares original, deterministic reconstruction, and absolute difference. Dynamics inspection shows initial frame, initial reconstruction, actual future, and predicted future. Predicted latents feed into the next prediction; actual frames are used only for comparisons/metrics. Actions are recorded from the actual trajectory. Autoplay stops at the end of a dynamics sequence. This validates a simulator under known actions; it does not run the occupancy-based C51 policy on decoded images.

Low validation loss alone is not the success criterion. Visually confirm sensible motion and stable images over 10–15 predictions after enough training. Short pipeline smoke tests cannot establish that scientific result.

## Tests

```sh
uv run python -m unittest discover -s pixel_world_model/tests -v
```

The optional native Tkinter smoke test requires a collected tiny dataset in `data/verification/rollout`:

```sh
uv run python -m pixel_world_model collect --dataset pixel_world_model/data/verification/rollout --transitions 40 --episode-limit 10 --resolution 64 --device cpu
uv run python -m pixel_world_model.tests.gui_smoke
```

It uses separate test settings and output directories, exercises all tabs, and verifies responsive progress and cancellation without changing the experiment settings.
