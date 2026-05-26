import numpy as np


def structural_energy(num_nodes, num_connections, active_connections):
    return (
        0.01 * num_nodes
        + 0.005 * num_connections
        + 0.01 * active_connections
    )


def behavioral_energy(distance_to_food, agent_energy, action_entropy):
    return (
        0.1 * distance_to_food
        + 0.5 * max(0.0, 1.0 - agent_energy)
        - 0.02 * action_entropy
    )


def activation_energy(activations):
    activations = np.asarray(activations)

    if activations.size == 0:
        return 0.0

    magnitude = np.mean(np.abs(activations))
    variance = np.var(activations)

    too_dead = max(0.0, 0.05 - variance)
    too_hot = max(0.0, magnitude - 1.0)

    return too_dead + too_hot


def total_energy(
    distance_to_food,
    agent_energy,
    action_entropy,
    num_nodes=0,
    num_connections=0,
    active_connections=0,
    activations=None,
):
    e = behavioral_energy(
        distance_to_food=distance_to_food,
        agent_energy=agent_energy,
        action_entropy=action_entropy,
    )

    e += structural_energy(
        num_nodes=num_nodes,
        num_connections=num_connections,
        active_connections=active_connections,
    )

    if activations is not None:
        e += activation_energy(activations)

    return float(e)
