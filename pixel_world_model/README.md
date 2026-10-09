# Two-stage pixel world model

Launch the independent Tkinter experiment:

```sh
uv run python -m pixel_world_model
```

1. **Data:** choose a C51 `.zip` policy and a new, empty dataset directory. Collect trajectories with configurable random actions. The default plays 500 independently seeded episodes: 450 training and 50 validation. It keeps up to 100 randomly chosen samples per episode, one from each equal-duration interval, for up to 45,000 training and 5,000 validation samples. Inputs are 128×128 grayscale with four stacked frames and four game frames per action.
2. **Step 1: VAE:** train, then choose `best.pt`, a periodic checkpoint, or `final.pt` and inspect reconstructions. The launcher opens a live curve of the running training objective and held-out validation loss. Inspect all four stack frames; confirm blaster, bullets, and centipede segments are resolved. Change the budget/settings and train again if necessary.
3. **Step 2: Dynamics:** explicitly select the VAE checkpoint, then train the residual latent predictor. Inspect held-out autoregressive predictions, initially over 15 steps.

All settings live in this folder's `settings.json`. The root settings and main launcher are independent. Shared game/C51 code is imported; game environment constructor defaults are recorded in the dataset. Reward shaping does not enter either world-model loss.

## CLI

CLI flags override this experiment's settings. `--settings PATH` selects another experimental configuration.

```sh
uv run python -m pixel_world_model collect --policy-model models/dqn_centipede --dataset pixel_world_model/data/run1 --episodes 500 --validation-episodes 50 --samples-per-episode 100
uv run python -m pixel_world_model train-vae --dataset pixel_world_model/data/run1 --vae-output pixel_world_model/checkpoints/vae_run1
uv run python -m pixel_world_model inspect-vae --dataset pixel_world_model/data/run1 --vae-checkpoint pixel_world_model/checkpoints/vae_run1/best.pt
uv run python -m pixel_world_model train-dynamics --dataset pixel_world_model/data/run1 --vae-checkpoint pixel_world_model/checkpoints/vae_run1/best.pt --dynamics-output pixel_world_model/checkpoints/dynamics_run1
uv run python -m pixel_world_model inspect-dynamics --dataset pixel_world_model/data/run1 --dynamics-checkpoint pixel_world_model/checkpoints/dynamics_run1/best.pt
```

Use `--help` for all configuration options. Paths entered in the launcher resolve against the project directory; CLI relative paths resolve against its working directory. CPU, CUDA, and MPS are selectable. `auto` prefers CUDA, then MPS, then CPU.

## Training and artifacts

Frames are uint8 `.npy` shards (512 frames each). Full trajectories are retained on disk, including actions and terminal flags, so four-frame inputs, next-state targets, and 15-step inspection sequences remain genuinely consecutive. Separate `sampled_indices` select the sparsely spaced examples used by both trainers; unselected transitions are context only. This improves diversity per training epoch rather than reducing collection disk usage. Legacy datasets without sampled indices still train on all their transitions. Latent caches encode only selected examples.

Episode seeds are `seed + episode_index`; the split is decided before play using an independent seeded RNG. Sampling uses another RNG independent of action exploration. Short episodes contribute all their transitions when shorter than the sample budget, without duplicates. Games run to natural termination with a configurable safety cap of 10,000 agent transitions per episode. The manifest records each episode seed, split, sample count, and whether it ended naturally, at the cap, or due to cancellation. Reset frames repeat to fill the initial stack; subsequent channels are separated by four game frames. Stacks and rollouts never cross resets. Cancelling produces a partial dataset; its split totals may differ from the requested 450/50.

### Cloud runs and the browser inspector

Each training output directory contains `run_config.json`, checkpoints, and TensorBoard event files under `tensorboard/`. Scalar logs include training and validation objectives plus VAE reconstruction BCE, pixel MSE, KL, or dynamics latent MSE. On each new best validation checkpoint, VAE runs log fixed held-out input/reconstruction/difference images; dynamics runs log fixed one-step actual/reconstructed/predicted comparisons. Logs flush after validation so curves remain available during cloud runs and after restart.

Run the Streamlit inspector alongside training on the cloud instance. Training and validation objectives appear on the same interactive graph, refreshed every five seconds by default (with pause and refresh controls). Held-out comparisons use the same autoregressive rollouts as local Inspect dynamics, with sample, stack-frame, horizon, and prediction-step controls. The default follows `current.pt`, atomically saved after every completed epoch and on completion/cancellation, rather than only validation improvements. You can start the dashboard before the first snapshot exists; it waits while continuing to show available metrics. Training metrics flush within ten seconds; validation is available after each epoch.

```sh
uv run python -m pixel_world_model serve-vae --port 8765
# Or inspect the dynamics model:
uv run python -m pixel_world_model serve-dynamics --dataset /path/on/verda/dataset --dynamics-output pixel_world_model/checkpoints/dynamics --port 8765
```

Leave the inspector command running in a terminal on the instance. The server binds to `127.0.0.1` by default. From your local machine, forward the port using Verda CLI:

```sh
verda ssh <instance-id> --key ~/.ssh/your_key -- -L 8765:localhost:8765
```

Or use regular SSH with your chosen key:

```sh
ssh -i ~/.ssh/your_key -N -L 8765:127.0.0.1:8765 user@cloud-host
```

Keep that SSH command running and open <http://localhost:8765> in your local browser. Use `--settings PATH` or matching dataset/output flags for your run. Paths must exist on Verda; override the dataset path saved by your local GUI. Pass `--dynamics-checkpoint PATH` to inspect a specific checkpoint such as `best.pt` instead. To view training loss, point `--logdir PATH` at that run's TensorBoard event directory. Keep the output directory on persistent cloud storage if you need curves and checkpoints after the instance stops. Device defaults to `auto`: CUDA when available, otherwise MPS, otherwise CPU. Both training and inspection honor `--device` overrides.

TensorBoard remains available for users who want to browse raw event data:

```sh
uv run tensorboard --logdir pixel_world_model/checkpoints/vae/tensorboard --host 127.0.0.1 --port 6006
```

VAE input/output is `[0,1]`. Its loss is weighted `BCEWithLogitsLoss` + beta × mean KL. Non-black target pixels (`target > 0`) have weight `vae_foreground_weight` (default 9), and black pixels have weight 1. This balances approximately 10% foreground against 90% background. BCE is divided by the sum of pixel weights over batch, stack channels, and pixels; KL is averaged over batch and latent dimensions. The decoder produces raw logits for training and applies sigmoid when reconstructing/displaying images. Grayscale targets remain in `[0,1]` without binarization. Progress and inspection report reconstruction BCE and ordinary pixel MSE separately. BCE values cannot be compared numerically with previous MSE loss values; the KL beta is unchanged. Set `vae_foreground_weight` in settings, `--vae-foreground-weight 9` on the CLI, or “Non-black pixel weight (black = 1)” in the VAE GUI. The weight must be finite and positive; 1 restores unweighted BCE. A low global MSE can still hide missing small objects; inspect reconstructions rather than treating it as accuracy. Supported square resolutions are 64, 84, 128, 192, and 256. Nonmultiples of 16 use decoder interpolation to recover the requested output size.

Dynamics uses the deterministic VAE mean and next-latent MSE. Encoder and decoder are evaluation-only and frozen. Latent pairs are cached by dataset directory and VAE SHA-256 identity. Each dynamics run pins a byte-identical `associated_vae.pt` beside its checkpoints, so inspection cannot accidentally use a subsequently replaced VAE.

Checkpoints include model, optimizer, architecture, preprocessing, dataset identity, epoch/update cursor, batch ordering, and RNG state. `best.pt` uses deterministic held-out validation; `epoch_NNNN.pt` is periodic; `final.pt` completes the configured total epoch budget. Cancelling saves `cancelled.pt` after the current update. Choose a resume file and a larger **total** epoch budget to continue; architecture, dataset, VAE, batch size, and beta must match. Learning rate may be changed when resuming. Resuming a checkpoint with a different reconstruction objective or foreground weight resets the best-validation comparison while preserving model, optimizer, and update progress. Use a larger total epoch budget and preferably a new output directory; retrain dynamics after improving the VAE because latent caches and dynamics checkpoints are tied to its exact weights.

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
uv run python -m pixel_world_model collect --dataset pixel_world_model/data/verification/rollout --episodes 4 --validation-episodes 1 --samples-per-episode 5 --episode-limit 10 --resolution 64 --device cpu
uv run python -m pixel_world_model.tests.gui_smoke
```

It uses separate test settings and output directories, exercises all tabs, and verifies responsive progress and cancellation without changing the experiment settings.

Starting fresh dynamics training with a different VAE automatically archives the existing output directory to a sibling `<directory>-previous-<timestamp>` directory, preserving checkpoints, the associated VAE, and logs together. The new run uses the configured output path. Resuming still requires the original VAE; clear the dynamics resume field to train with a new VAE.

A frozen VAE may be used for dynamics training or inspection on a different dataset with the same resolution, frame stack, and frame gap. For example, train the VAE on `data/diverse`, then choose `data/default` in Step 2 to learn dynamics from the older consecutive-transition dataset. Older datasets without `sampled_indices` use all transitions automatically. Resuming either training stage still requires that stage’s original dataset, and dynamics inspection requires its training dataset.

Dynamics uses latent MSE plus `dynamics_pixel_loss_weight` (default 1) times weighted BCE between the decoded predicted next stack and the actual next stack. `dynamics_foreground_weight` defaults to 9, with black target pixels weighted 1 and normalization by total pixel weight, just like the VAE. Both settings are available in Step 2 and as `--dynamics-pixel-loss-weight` / `--dynamics-foreground-weight`. Set 0 for latent-only training. Training evaluates the weighted visual objective on a uniformly random subset of each latent batch (`dynamics_visual_batch`, default 16), reducing decoder work. This is a stochastic approximation of the full visual objective, with higher gradient variance. Latent MSE still uses the entire batch; validation evaluates every sample for consistent comparisons. Increase the visual batch for less noisy gradients, or set it to the dynamics batch size for full-batch visual training. Configure it in Step 2 or via `--dynamics-visual-batch`. Sampling RNG state is saved for reproducible resumes. The frozen decoder passes gradients back to dynamics without updating VAE weights; this adds computation and memory use. Progress/TensorBoard report `latent_mse`, `prediction_bce`, and `pixel_mse`. Resuming with a changed objective or weighting resets the best-validation comparison.

Collection uses four spawned CPU workers by default. Set **Collection workers** in the Data tab or pass `--collection-workers 4` to `collect`; use 1 for the original serial collector. Workers simulate, render, resize, and write independent frame shards, while the parent batches C51 inference on the selected device. GPU policy weights are loaded only once. Complete episodes are indexed contiguously when merged, preserving stack boundaries, trajectories, and validation assignments. Parallel exploration and sampling use episode-specific random streams, so changing between parallel worker counts preserves episode results (subject to device inference determinism); the original serial random stream differs. Cancellation retains completed and in-progress episodes and shuts down workers. Existing datasets remain compatible. Full episode frames are still stored for trajectory inspection; this change speeds collection without reducing storage requirements.

Choose **Sampling mode** in the Data tab: `sampled` spreads up to `samples_per_episode` training examples across each episode (recommended for VAE); `consecutive` selects every transition (recommended for dynamics), ignoring the sample limit. CLI equivalents are `--collection-mode sampled` and `--collection-mode consecutive`. Generate these into separate dataset directories and select the sampled directory in Step 1 and the consecutive directory in Step 2. Both modes preserve complete frame histories for correct four-frame stacks and inspector trajectories; sampled mode reduces training examples, not frame-storage size. Existing datasets keep their original sample selection. A VAE trained on sampled data can encode a consecutive dataset with matching preprocessing.
