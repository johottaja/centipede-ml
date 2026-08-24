"""Training hyperparameter specs shared by the GUI and CLI."""
from __future__ import annotations

import json
import os
from typing import Any

SETTINGS_PATH = "settings.json"

# Each entry is either:
#   ("section", "Section Title")
#   (label, cli-key, default, type, min, max, tooltip)
#   For type "choice": min holds the list of allowed values, max is unused.
SPECS: list[tuple] = [
    ("section", "Training"),
    ("Timesteps",              "timesteps",              "100000", "int",   1,    None, "Total environment steps to train for"),
    ("Parallel envs",          "n-envs",                 "4",       "int",   1,    None, "Number of parallel environments (SubprocVecEnv); more envs = faster experience collection"),
    ("Seed",                   "seed",                   "0",       "int",   0,    None, "Random seed for reproducibility"),
    ("Eval frequency",         "eval-freq",              "30000",   "int",   1,    None, "Run deterministic eval games every N training steps"),
    ("Eval games",             "n-eval-episodes",        "10",      "int",   1,    None, "Number of deterministic eval games per evaluation"),
    ("Checkpoint frequency",   "checkpoint-freq",        "100000",  "int",   1,    None, "Save a model checkpoint every N training steps"),

    ("section", "C51 / CNN"),
    ("Learning rate",          "learning-rate",          "0.0001",  "float", 0,    1,    "Adam optimizer learning rate"),
    ("Buffer size",            "buffer-size",            "100000",  "int",   1,    None, "Replay buffer capacity"),
    ("Learning starts",        "learning-starts",        "10000",   "int",   0,    None, "Steps before first gradient update"),
    ("Batch size",             "batch-size",             "64",      "int",   1,    None, "Mini-batch size for each update"),
    ("Tau (soft update)",      "tau",                    "1.0",     "float", 0,    1,    "Soft update coefficient for target network (1 = hard copy)"),
    ("Gamma (discount)",       "gamma",                  "0.99",    "float", 0,    1,    "Discount factor for future rewards"),
    ("N-step",                 "n-steps",                "4",       "int",   1,    None, "N-step returns (Rainbow). 1 = one-step TD; typical Rainbow value is 3"),
    ("Train frequency",        "train-freq",             "4",       "int",   1,    None, "Steps between gradient updates"),
    ("Gradient steps",         "gradient-steps",         "1",       "int",   1,    None, "Gradient updates per train call"),
    ("Target update interval", "target-update-interval", "1000",    "int",   1,    None, "Steps between target network syncs"),
    ("Exploration fraction",   "exploration-fraction",   "0.1",     "float", 0,    1,    "Fraction of training spent exploring"),
    ("Final epsilon",          "exploration-final-eps",  "0.01",    "float", 0,    1,    "Epsilon at end of exploration schedule"),
    ("Net architecture",       "net-arch",               "256,256", "str",   None, None, "MLP head layer sizes after CNN, comma-separated (e.g. 256,256)"),
    ("Atoms (C51)",            "n-atoms",                "51",      "int",   3,    501,  "Number of atoms in the return distribution"),
    ("V-min",                  "v-min",                  "-10000",  "float", None, None, "Minimum return value in the C51 support"),
    ("V-max",                  "v-max",                  "10000",   "float", None, None, "Maximum return value in the C51 support"),
    ("Prioritized replay",     "prioritized-replay",     "true",    "choice", ["true", "false"], None, "Sample by TD-error priority and correct with importance-sampling weights (Schaul et al. 2015)"),
    ("PER alpha",              "per-alpha",              "0.6",     "float", 0,    1,    "Priority exponent (P(i) ∝ |δ_i|^α). 0 = uniform sampling"),
    ("PER beta",               "per-beta",               "0.4",     "float", 0,    1,    "Importance-sampling exponent at the start of training (annealed to 1)"),

    ("section", "Rewards"),
    ("Mushroom hit",           "reward-mushroom-hit",     "1",      "float", 0,    None, "Reward for hitting a mushroom without destroying it"),
    ("Mushroom destroy",       "reward-mushroom-destroy", "5",      "float", 0,    None, "Reward for fully destroying a mushroom"),
    ("Body segment hit",       "reward-body-hit",         "10",     "float", 0,    None, "Reward for hitting a centipede body segment"),
    ("Head hit",               "reward-head-hit",         "100",    "float", 0,    None, "Reward for hitting the centipede head"),
    ("Depth discount",         "reward-depth-discount",   "0.0",    "float", 0,    1,    "Fraction by which hit rewards are reduced at the bottom row (0 = disabled, 1 = zero reward at bottom)"),
    ("Depth discount fn",      "reward-depth-discount-fn","linear", "choice", ["linear", "exponential"], None, "Shape of the depth discount curve"),
    ("Spider hit",             "reward-spider-hit",        "300",    "float", 0,    None, "Reward for shooting a spider"),
    ("Spider collision penalty","reward-spider-penalty",   "1000",   "float", 0,    None, "Penalty (subtracted) when a spider touches the player"),
    ("Centipede collision penalty","reward-centipede-penalty","1000","float", 0,    None, "Penalty (subtracted) when a centipede segment touches the player"),
    ("Survival bonus",           "reward-survival",          "0.01",   "float", 0,    None, "Reward added every step the player stays alive"),
    ("Proximity penalty",        "reward-proximity-penalty", "1.0",    "float", 0,    None, "Max penalty per step when a threat is within proximity range (scales linearly with distance)"),
    ("Proximity range (tiles)",  "proximity-distance-tiles", "3",      "int",   1,    None, "Distance in tiles within which proximity penalty applies"),
]

FIELD_SPECS = [s for s in SPECS if s[0] != "section"]
SETTINGS_KEYS = {key for _, key, *_ in FIELD_SPECS}

# CentipedeEnv / GameEngine kwargs derived from the Rewards section.
ENV_REWARD_KEYS = (
    "reward_mushroom_hit",
    "reward_mushroom_destroy",
    "reward_body_hit",
    "reward_head_hit",
    "reward_depth_discount",
    "reward_depth_discount_fn",
    "reward_spider_hit",
    "reward_spider_penalty",
    "reward_centipede_penalty",
    "reward_survival",
    "reward_proximity_penalty",
    "proximity_distance_tiles",
)


def _coerce_value(raw: str, typ: str) -> Any:
    if typ == "int":
        return int(raw)
    if typ == "float":
        return float(raw)
    if typ in ("str", "choice"):
        return raw
    raise ValueError(f"unknown setting type: {typ}")


def load_settings_defaults(path: str = SETTINGS_PATH) -> dict[str, Any]:
    """Load settings.json merged with built-in defaults.

    Returns a dict keyed by argparse dest names (underscores), ready for
  ``parser.set_defaults(**...)``.
    """
    file_data: dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                file_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    defaults: dict[str, Any] = {}
    for _label, key, default, typ, *_ in FIELD_SPECS:
        raw = str(file_data.get(key, default))
        dest = key.replace("-", "_")
        defaults[dest] = _coerce_value(raw, typ)
    return defaults


def env_reward_kwargs(path: str = SETTINGS_PATH) -> dict[str, Any]:
    """Reward-related kwargs for CentipedeEnv, from settings.json."""
    defaults = load_settings_defaults(path)
    return {k: defaults[k] for k in ENV_REWARD_KEYS}
