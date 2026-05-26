import json
import os
import random
from pathlib import Path

import numpy as np


def set_global_seed(seed):
    """
    Set common Python/NumPy seeds.

    neat-python still has internal stochasticity through Python's random module,
    so seeding random is the important part for NEAT reproduction.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_run_config(output_dir, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / 'run_config.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, sort_keys=True)

    return path
