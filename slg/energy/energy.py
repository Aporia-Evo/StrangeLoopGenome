import numpy as np


def structural_energy(num_nodes, num_connections, active_connections):
    """
    Light structural pressure.

    The goal is not to minimize the network to death, but to prefer compact
    solutions when task performance is comparable.
    """
    return (
        0.002 * num_nodes
        + 0.001 * num_connections
        + 0.003 * active_connections
    )


def behavioral_energy(distance_to_food, agent_energy, action_entropy):
    """
    Energy as a weak search field.

    Distance and depleted agent energy matter, but should not dominate the task
    reward. Action entropy is rewarded slightly to avoid frozen policies.
    """
    return (
        0.03 * distance_to_food
        + 0.15 * max(0.0, 1.0 - agent_energy)
        - 0.03 * action_entropy
    )


def activation_energy(activations):
    activations = np.asarray(activations)

    if activations.size == 0:
        return 0.0

    magnitude = np.mean(np.abs(activations))
    variance = np.var(activations)

    too_dead = max(0.0, 0.03 - variance)
    too_hot = max(0.0, magnitude - 1.5)

    return 0.25 * (too_dead + too_hot)


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
