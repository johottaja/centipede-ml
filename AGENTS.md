# Project Structure

```
centipede/
├── main.py                      # Entry point — boots the Tkinter Launcher GUI
├── pyproject.toml               # uv project config (Python ≥3.13, see deps below)
├── core/
│   ├── game.py                  # Headless GameEngine + all game objects
│   ├── env.py                   # Gymnasium wrapper (CentipedeEnv) around GameEngine
│   ├── train.py                 # DQN training script (Stable-Baselines3)
│   ├── watch.py                 # Watch a trained agent play (visual, pygame window)
│   └── play.py                  # Human-playable keyboard runner
├── gui/
│   ├── launcher.py              # Tkinter 3-tab launcher (Play / Train / Watch)
│   ├── hparam_panel.py          # Scrollable hyperparameter form (Train tab)
│   ├── progress_window.py       # Live training-progress window (reads train.py stdout)
│   └── tooltip.py               # Tooltip helper widget
└── models/                      # Saved model files (gitignored)
    ├── dqn_centipede.zip         # Final model (written at end of training)
    └── dqn_centipede_ckpt_*_steps.zip  # Checkpoints (every 100k steps)
```

---

## core/game.py — GameEngine

Pure game logic, no event loop. Call `reset()` then `step(action)` in a loop.

**Grid:** 30×31 tiles, 16×16 px each → 480×496 px.  
**Player zone:** bottom 5 rows (rows 26–30).  
**Centipede:** spawns at row 0, length 12 segments; respawns immediately when all cleared.  
**Mushroom field:** ~6% density in rows 1–25.

Key attributes on `GameEngine`:
- `score` — cumulative score (int)
- `segments_destroyed` — total centipede segments destroyed this episode (int)
- `player.lives` — remaining lives (starts at 3)
- `terminated` — True when lives reach 0

Reward values are constructor params (also settable per-run from the GUI):

| Event | Default reward |
|---|---|
| Mushroom hit (not destroyed) | +1 |
| Mushroom fully destroyed | +5 |
| Centipede body segment hit | +10 |
| Centipede head hit | +100 |

---

## core/env.py — CentipedeEnv

Standard `gymnasium.Env` wrapper.

- **Observation space:** `Box(0, 255, shape=(31, 30, 20), dtype=uint8)` — 4 stacked occupancy frames × 5 channels (player, mushrooms, centipede heads, centipede body, spiders). Each channel encodes fractional tile occupancy (0–255).
- **Action repeat:** agent picks one action every 4 game frames (`frame_skip=4` for training/watch; `frame_skip=1` for human play). Rewards are summed across repeated frames.
- **Action space:** `Discrete(6)` — NOOP / LEFT / RIGHT / UP / DOWN / FIRE
- **Reward:** score delta per agent step (includes survival bonus and proximity shaping)
- **Terminated:** player loses all 3 lives
- **`step()` info dict:** `{"score": int, "lives": int, "segments_destroyed": int, "spiders_destroyed": int}`

---

## core/train.py — C51 Training

Runs `C51` (Categorical DQN with `C51Policy` + custom `GridCNN` feature extractor) via Stable-Baselines3.
Each action-value is a categorical distribution over `n_atoms` (default 51) support points in `[v_min, v_max]`.
Spawned as a subprocess by the GUI; communicates progress via newline-delimited JSON on **stdout** (for the progress window). Human-readable logs (eval scores, progress, checkpoints) go to **stderr** — use `--quiet` for JSON-only output.

CLI usage loads hyperparameters from `settings.json` by default (`--settings PATH` to override). Any CLI flag overrides the corresponding settings value.

Uses `SubprocVecEnv` (default 4 parallel workers) to collect experience concurrently across multiple processes, keeping the GPU fed. Falls back to `DummyVecEnv` when `n_envs=1`.

**Callbacks:**
- `CheckpointCallback` — saves `models/dqn_centipede_ckpt_<N>_steps.zip` every 100k steps
- `EvalProgressCallback` — runs 10 deterministic games on a separate eval env every `eval-freq` steps (default 30k) and emits results
- `ProgressCallback` — emits progress stats every 5k steps

**stdout message types:**

| `type` | Fields | When |
|---|---|---|
| `device` | `device` | startup |
| `start` | `total` | before `model.learn()` |
| `progress` | `steps`, `total`, `pct`, `elapsed`, `eta`, `steps_per_sec` | every 5k steps |
| `eval` | `steps`, `episodes: [{score, segments_destroyed}, …]` | every `eval-freq` steps (default 30k, 10 games) |
| `done` | `elapsed` | training complete |
| `log` | `text` | non-JSON lines (errors, SB3 output) |

---

## core/watch.py

Loads a saved model and runs it visually for N episodes in a pygame window.  
Prints `score` and `total_reward` per episode to stdout.  
The launcher **Watch Agent** tab lists the final model and all checkpoints (`dqn_centipede_ckpt_*_steps.zip`); selection is persisted in `settings.json` as `watch-model`.  
Usage: `uv run python -m core.watch [--model PATH] [--episodes N] [--fps N]`  
`PATH` is without `.zip` (e.g. `models/dqn_centipede_ckpt_300000_steps`).

---

## core/play.py

Human-playable runner. Boots `CentipedeEnv(render_mode="human", frame_skip=1)` and maps keyboard input to actions each frame. Press `R` to restart after game over.  
Keys: arrow keys or WASD to move, Space to fire.

---

## gui/launcher.py — Launcher (Tkinter)

Three-tab window:
- **Play** — launches `core.play` in a subprocess
- **Train Agent** — shows `HParamPanel`, spawns `core.train` subprocess, opens `ProgressWindow`
- **Watch Agent** — launches `core.watch` in a subprocess (disabled if no model exists)

Status bar shows model file age.

---

## gui/hparam_panel.py — HParamPanel

Scrollable grid of labelled `Entry` widgets for every DQN hyperparameter (training, DQN, rewards). Validates inputs and returns a flat CLI arg list for `core/train.py`. Hover tooltips on every field.

---

## gui/progress_window.py — ProgressWindow

`tk.Toplevel` that reads the `core.train` subprocess stdout in a background thread, parses JSON messages, and updates a progress bar + stats grid every 300 ms. Handles cancel (terminates subprocess).

---

## Dependencies (pyproject.toml)

| Package | Purpose |
|---|---|
| `gymnasium` | RL environment API |
| `stable-baselines3` | DQN implementation |
| `torch` + `torchvision` | Neural network backend (auto-selects MPS / CUDA / CPU) |
| `pygame-ce` | Game rendering and human play |
| `numpy` | Observation arrays |
| `opencv-python-headless` | (available for future use) |
