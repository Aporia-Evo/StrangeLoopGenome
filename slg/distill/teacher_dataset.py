import pickle
from pathlib import Path

import neat
import numpy as np

from slg.envs.gridworld import GridWorld


def load_top_genomes(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Missing top genomes file: {path}')

    with open(path, 'rb') as f:
        return pickle.load(f)


def make_recurrent_policy(genome, config):
    net = neat.nn.RecurrentNetwork.create(genome, config)

    def policy(obs):
        output = np.asarray(net.activate(obs), dtype=np.float32)
        action = int(np.argmax(output))
        return action, output

    return policy


def collect_teacher_episode(policy, seed, size=8, max_steps=96):
    env = GridWorld(size=size, max_steps=max_steps, seed=seed)
    obs = env.reset()
    done = False

    observations = []
    actions = []
    logits = []
    rewards = []
    infos = []

    while not done:
        action, output = policy(obs)

        observations.append(obs.copy())
        actions.append(action)
        logits.append(output.copy())

        obs, reward, done, info = env.step(action)
        rewards.append(float(reward))
        infos.append(dict(info))

    return {
        'observations': np.asarray(observations, dtype=np.float32),
        'actions': np.asarray(actions, dtype=np.int64),
        'logits': np.asarray(logits, dtype=np.float32),
        'rewards': np.asarray(rewards, dtype=np.float32),
        'foods': int(infos[-1]['foods_collected']) if infos else 0,
        'wall_hits': int(infos[-1]['wall_hits']) if infos else 0,
    }


def build_teacher_dataset(
    top_genomes_path,
    config,
    output_path,
    num_genomes=5,
    seeds_per_genome=50,
    seed_offset=0,
    min_foods=0,
    max_wall_hits=None,
):
    top_genomes = load_top_genomes(top_genomes_path)
    selected = top_genomes[:num_genomes]

    all_obs = []
    all_actions = []
    all_logits = []
    all_rewards = []
    episode_rows = []
    skipped = 0

    for genome_rank, (record, genome) in enumerate(selected):
        policy = make_recurrent_policy(genome, config)

        for local_seed in range(seeds_per_genome):
            seed = seed_offset + genome_rank * seeds_per_genome + local_seed
            episode = collect_teacher_episode(policy, seed=seed)

            if episode['foods'] < min_foods:
                skipped += 1
                continue

            if max_wall_hits is not None and episode['wall_hits'] > max_wall_hits:
                skipped += 1
                continue

            start = sum(len(x) for x in all_obs)
            length = len(episode['actions'])

            all_obs.append(episode['observations'])
            all_actions.append(episode['actions'])
            all_logits.append(episode['logits'])
            all_rewards.append(episode['rewards'])

            episode_rows.append({
                'genome_rank': genome_rank,
                'genome_id': record.genome_id,
                'generation': record.generation,
                'teacher_fitness': record.fitness,
                'seed': seed,
                'start': start,
                'length': length,
                'foods': episode['foods'],
                'wall_hits': episode['wall_hits'],
            })

    if not all_obs:
        raise ValueError(
            'Teacher dataset is empty after filtering. '
            'Lower --min-foods or increase --max-wall-hits.'
        )

    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
        logits=logits,
        rewards=rewards,
        episode_rows=np.asarray(episode_rows, dtype=object),
    )

    summary = {
        'output_path': str(output_path),
        'num_genomes': len(selected),
        'seeds_per_genome': seeds_per_genome,
        'num_episodes': len(episode_rows),
        'num_skipped_episodes': skipped,
        'num_samples': int(len(actions)),
        'observation_dim': int(observations.shape[1]),
        'num_actions': int(logits.shape[1]),
        'min_foods': min_foods,
        'max_wall_hits': max_wall_hits,
        'mean_episode_foods': float(np.mean([row['foods'] for row in episode_rows])),
        'mean_episode_wall_hits': float(np.mean([row['wall_hits'] for row in episode_rows])),
    }

    return summary
