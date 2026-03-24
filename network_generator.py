"""
network_generator.py
--------------------
Generates synthetic time-varying networks (TVNs) for the MDDT reproduction.

A TVN is a graph G_t = (V, E_t) where V is fixed and E_t changes per time slot.
Each link has an integer capacity representing data transferable in that slot.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass
class Link:
    """A directed link active at a specific time slot."""
    u: int      # source node id (0-indexed)
    v: int      # destination node id (0-indexed)
    cap: int    # integer capacity in [cap_lo, cap_hi]
    t: int      # time slot this link is active


@dataclass
class TVN:
    """A time-varying network."""
    n_nodes: int
    T_max: int
    source: int
    dest: int
    links: List[Link]
    links_by_t: Dict[int, List[Link]] = field(default_factory=dict)


def generate_tvn(
    n_nodes: int,
    T_max: int,
    activation_prob: float,
    cap_lo: int,
    cap_hi: int,
    source: int,
    dest: int,
    seed: int,
) -> TVN:
    """
    Generate a synthetic time-varying network.

    For each ordered node pair (u, v) with u != v and each time slot t in [0, T_max-1],
    activate the link with probability `activation_prob` and assign a random integer
    capacity in [cap_lo, cap_hi].

    Uses numpy.random.default_rng(seed) for reproducibility.
    """
    rng = np.random.default_rng(seed)
    links = []
    links_by_t: Dict[int, List[Link]] = {t: [] for t in range(T_max)}

    for t in range(T_max):
        for u in range(n_nodes):
            for v in range(n_nodes):
                if u == v:
                    continue
                if rng.random() < activation_prob:
                    cap = int(rng.integers(cap_lo, cap_hi + 1))
                    lnk = Link(u=u, v=v, cap=cap, t=t)
                    links.append(lnk)
                    links_by_t[t].append(lnk)

    return TVN(
        n_nodes=n_nodes,
        T_max=T_max,
        source=source,
        dest=dest,
        links=links,
        links_by_t=links_by_t,
    )


def compute_d_values(tvn: TVN, n_points: int = 15) -> List[int]:
    """
    Compute a range of D (data volume) values for experiments.

    The range spans from a small value to 85% of the total capacity of links
    leaving the source node, expressed as integer values.

    This ensures all D values are achievable and produces meaningful Figure 3 trends.
    """
    max_cap = sum(lnk.cap for lnk in tvn.links if lnk.u == tvn.source)
    if max_cap == 0:
        return list(range(1, n_points + 1))

    d_min = max(1, max_cap // (n_points * 3))
    d_max = int(0.85 * max_cap)

    if d_max <= d_min:
        d_max = d_min + n_points

    raw = np.linspace(d_min, d_max, n_points)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for v in raw:
        iv = int(round(v))
        iv = max(1, iv)
        if iv not in seen:
            seen.add(iv)
            result.append(iv)

    return sorted(result)
