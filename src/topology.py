"""
Communication topology builders.

Adjacency convention: adj[j, i] = 1  means agent i receives from agent j.
Equivalently, j's message is visible to i at the next round.

Supported topologies:
  "star"         — one center (agent 0) connected to all leaves bidirectionally
  "chain"        — directed ring: 0→1→2→…→(N-1)→0
  "sparse_random"— each directed edge (j→i), j≠i, included with probability p
  "full"         — all-to-all (fully connected)
"""

import numpy as np


def build_adjacency(
    topology: str,
    n_agents: int,
    density: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """
    Build an N×N directed adjacency matrix.

    Args:
        topology:  One of "star", "chain", "sparse_random", "full".
        n_agents:  Number of agents N.
        density:   Edge probability for "sparse_random" topology.
        seed:      Random seed for "sparse_random".

    Returns:
        Integer numpy array of shape (N, N) with 0/1 entries and zero diagonal.
    """
    adj = np.zeros((n_agents, n_agents), dtype=int)

    if topology == "full":
        adj = np.ones((n_agents, n_agents), dtype=int)
        np.fill_diagonal(adj, 0)

    elif topology == "star":
        # Center = agent 0. Leaves send to center; center sends to leaves.
        for leaf in range(1, n_agents):
            adj[leaf, 0] = 1    # center receives from leaf
            adj[0, leaf] = 1    # leaf receives from center

    elif topology == "chain":
        # Directed chain: 0→1→2→…→(N-1)→0 (ring)
        for i in range(n_agents):
            nxt = (i + 1) % n_agents
            adj[i, nxt] = 1     # agent nxt receives from agent i

    elif topology == "sparse_random":
        rng = np.random.default_rng(seed)
        for j in range(n_agents):
            for i in range(n_agents):
                if i != j and rng.random() < density:
                    adj[j, i] = 1

    else:
        raise ValueError(f"Unknown topology: {topology!r}. "
                         "Choose from: star, chain, sparse_random, full.")

    return adj


def adjacency_to_list(adj: np.ndarray) -> list:
    """Convert adjacency matrix to nested list (for JSON serialisation)."""
    return adj.tolist()
