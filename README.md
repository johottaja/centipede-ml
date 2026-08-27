# Centipede RL

A from-scratch Centipede clone trained with distributional DQN until the agent plays at a superhuman level.

This repository is a project-driven study of state-of-the-art deep RL. I wrote things down as I went, implemented the methods rather than treating them as a black box, and kept the failed experiments. Other branches in this repo are the graveyard of those attempts: different extractors, PPO, relative coordinates, earlier DQN variants. The lessons from those runs are why the agent on `main` works.

I had to step away from the project for a while. When I came back and finished the work, the agent finally reached superhuman play.

## Tech stack

- **PyTorch**: the network, the C51 projection, and the training loop.
- **Stable-Baselines3**: DQN infrastructure (replay, vectorized envs, target nets, callbacks). C51, prioritized replay, n-step returns, and the dueling head are custom on top of SB3's `DQN`.
- **Pygame**: the game itself: a headless engine for training and a rendered window for human play and watching the agent.
- **Gymnasium**: the environment API (`CentipedeEnv`) around that engine.

## How the agent sees the game

The playfield is a 30×31 tile grid. The observation is not pixels. Each frame is six occupancy channels (uint8, 0-255, fractional tile coverage):

1. player
2. mushrooms
3. centipede heads
4. centipede body
5. spiders
6. bullet

Training uses **frame skip 4**: the agent chooses one of 10 actions (move, fire, or both) every four game frames, and those four occupancy tensors are stacked. The network therefore receives a `(24, 31, 30)` tensor (channels-first after SB3). That stack is cheap motion: where the centipede was, where the spider is going, whether a shot is in flight.

## DQN: feature extractor and head

The policy is Categorical DQN (C51). A CNN turns the occupancy stack into a vector; a small MLP turns that vector into a return *distribution* per action.

### GridCNN

`GridCNN` is a three-layer conv net sized for this board, not for Atari RGB.

| Layer | Operation | Output |
|---|---|---|
| Conv 1 | 24 → 32 channels, 5×5, stride 2, pad 2, ReLU | 32 × 16 × 15 |
| Conv 2 | 32 → 64, 3×3, stride 2, pad 1, ReLU | 64 × 8 × 8 |
| Conv 3 | 64 → 64, 3×3, stride 1, pad 1, ReLU | 64 × 8 × 8 |
| Flatten | | 4096 |
| Linear | 4096 → 512, ReLU | 512 |

The first layer uses a 5×5 kernel and stride 2 so a mushroom, a segment, and the player (all tile-sized) have spatial context without a huge receptive field. Two stride-2 layers shrink 31×30 to 8×8; the last 3×3 mix is local structure at that resolution (clusters of mushrooms, a centipede turning, a spider in the player zone). Images are scaled to `[0, 1]` (`normalize_images=True`).

The 512-d vector is the whole state embedding. Everything after this is Q-learning, not vision.

### Hidden layer (dueling C51 head)

After the CNN, `net_arch` is a **single 512-unit hidden layer**. With dueling on, that layer is not one MLP. It is two streams that share the CNN features:

- **Value stream**: 512 → 512 → 51 logits. This is \(V(s)\): how good the occupancy grid is, regardless of action. Most of the return in Centipede is “am I about to die / is there a head on screen,” which does not need a separate map per action.
- **Advantage stream**: 512 → 512 → \(10 \times 51\) logits. This is \(A(s, a)\): which of the ten actions is better *in this state*.

Atom logits are combined as in Wang et al. (2016):

\[
Q(s, a) = V(s) + A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')
\]

The mean subtraction keeps \(V\) and \(A\) from trading an arbitrary offset. Softmax over the 51 atoms turns logits into a categorical distribution on a fixed support \([v_{\min}, v_{\max}]\). Greedy action selection uses the expected value of that distribution.

Fifty-one atoms with \(v_{\min}=-10\), \(v_{\max}=10\) (defaults that match the scaled reward scheme below) are enough to represent short-horizon returns when each event is order-1. Gradients are clipped at L2 norm 10.

## Five Rainbow extensions

Rainbow (Hessel et al., 2017) stacked six DQN improvements. This project uses **five of them**. It does **not** use Noisy Nets; exploration is still ε-greedy (20% of training annealing to ε = 0.005).

1. **Distributional RL (C51)** (Bellemare, Dabney & Munos, 2017). Each \(Q(s,a)\) is a distribution over atoms, not a scalar. The loss is cross-entropy against the projected Bellman target. Centipede returns are spiky (long quiet stretches, then a head hit or a death); a distribution keeps that multi-modality instead of averaging it away.

2. **Double Q-learning** (van Hasselt, Guez & Silver, 2016). The *online* net picks the greedy next action; the *target* net supplies that action’s distribution. That cuts the overestimation you get when the same net both selects and evaluates.

3. **Dueling networks** (Wang et al., 2016). Value vs advantage streams, as above. Useful here because many states share a value (“safe” vs “about to collide”) while only a few actions change the outcome.

4. **Prioritized experience replay** (Schaul et al., 2015). Transitions are sampled in proportion to C51 TD error (\(\lvert\delta_i\rvert^\alpha\)), with importance-sampling weights. α = 0.5, β starts at 0.4 and anneals to 1 so the bias correction is complete by the end of training. Rare events (head shots, deaths) stay in the gradient mix.

5. **Multi-step returns**: n-step TD with **n = 4** (Rainbow used 3). The bootstrap target sums four rewards before discounting into the next distribution. That shortens the credit-assignment path for a shot that takes a few frames to land, without going fully Monte Carlo.

## Reward structure

The learner does not see arcade points. It sees a shaped signal that is summed over the four skipped frames.

**What actually trains well** (values in `settings.json`) is a *unit-scaled* event table so a single hit fits in \([-1, 1]\) after env clipping, and C51’s \([-10, 10]\) support can represent a discounted return:

| Event | Reward |
|---|---|
| Centipede head | +1 |
| Spider | +1 |
| Centipede body | +0.1 |
| Mushroom destroyed | +0.05 |
| Mushroom hit | +0.01 |
| Death (centipede or spider collision) | -1 |

### Failed experiments (please try them anyway)

These knobs are still in the GUI and in `GameEngine`. They did not get me to superhuman play. They are honest leftovers from trying to *tell* the agent how to live instead of letting scoring teach it.

- **Survival bonus**: a constant added every frame the player is alive. It rewards stalling in a corner more than clearing the board.
- **Proximity penalty**: a distance-scaled penalty when a segment or spider is within N tiles. The agent learns to flee instead of shooting, and the gradient fights the hit rewards.
- **Depth discount**: hit rewards shrink as the centipede walks down the board (linear or exponential). The intent was “kill it high.” In practice it made lower-board shots look worthless and confused credit assignment on splits.

Set them to 0 unless you want to reproduce those dead ends, or beat them.

### Alternative scheme (arcade-scale)

This table should train about as well if C51’s support is widened to match long-horizon returns:

| Event | Reward |
|---|---|
| Head | +100 |
| Spider | +100 |
| Body | +10 |
| Mushroom | +1 |
| Death | -100 |

Use **\(v_{\max} = 4500\)**, **\(v_{\min} = -300\)**. Those bounds have to cover discounted returns, not a single event. The environment currently clips the *per-step* reward to \([-1, 1]\); the unit-scaled table is built for that clip. For arcade-scale rewards you need to lift or widen that clip, or the 100s never reach the replay buffer.

## Other hyperparameters that matter

From the run that reached strong play (`settings.json`):

| | |
|---|---|
| Timesteps | 8M |
| Parallel envs | 256 (`SubprocVecEnv`), to keep the GPU busy |
| Replay buffer | 750k |
| Learning starts | 80k |
| Batch size | 32 |
| Learning rate | 3e-4 (Adam) |
| γ | 0.99 |
| τ | 0.05 (soft target updates) |
| Train freq / gradient steps | every 3 env steps, 4 grad steps |
| Target update interval | 1000 |
| Max grad norm | 10 |

Eval is 30 deterministic games every 250k steps; checkpoints land on the same cadence.

## Running

Python ≥ 3.13. [uv](https://docs.astral.sh/uv/) recommended.

```bash
uv sync
```

### GUI

Play, train, and watch from one window (hyperparameter form + live progress).

```bash
uv run python main.py
```

### Play

```bash
uv run python -m core.play
```

Arrow keys or WASD, Space to fire, `R` after game over.

### Train

```bash
uv run python -m core.train
```

Loads `settings.json`; CLI flags override. Final weights: `models/dqn_centipede.zip`. Checkpoints: `models/dqn_centipede_ckpt_<N>_steps.zip`.

```
uv run python -m core.train --help
```

### Watch

```bash
uv run python -m core.watch --model models/dqn_centipede
```

`--model` is the path without `.zip`. Occupancy overlays: keys 1-6 / F1-F4.
