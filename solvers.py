"""
solvers.py
----------
Implements the three MDDT solvers:
  1. Binary Search (BS)         - baseline: full max-flow per binary search step
  2. Enhanced Binary Search (EBS) - reuses residual graph when expanding horizon
  3. One-Pass Solver            - incremental TEG expansion with full residual reuse

All solvers use a DFS-based Ford-Fulkerson augmenting path method on the
residual graph built by teg_builder.py.

Each solver returns (delivery_time: int, elapsed_seconds: float).
"""

import time
from typing import Any, Tuple

from network_generator import TVN
from teg_builder import (
    INF,
    ResGraph,
    add_time_layer,
    build_full_teg,
    copy_residual_graph,
    new_residual_graph,
    add_edge,
)


# ---------------------------------------------------------------------------
# DFS augmenting path (Ford-Fulkerson core)
# ---------------------------------------------------------------------------

def dfs_augmenting_path(graph: ResGraph, source: Any, sink: Any) -> int:
    """
    Find one augmenting path from source to sink using iterative DFS.

    Returns the amount of flow pushed along the path (bottleneck capacity),
    or 0 if no augmenting path exists.

    The residual graph is updated in-place when flow is pushed.

    Stack entries: (node, bottleneck_so_far, path)
      where path is a list of (parent_node, edge_index) pairs used to
      update residuals after reaching the sink.

    The visited set prevents revisiting nodes (avoids cycles in the residual).
    Edges with cap == 0 are skipped (saturated edges).
    """
    if source not in graph:
        return 0

    visited = {source}
    # stack: (current_node, bottleneck_to_here, path_so_far)
    stack = [(source, INF, [])]

    while stack:
        node, bottleneck, path = stack.pop()

        if node == sink:
            # Found an augmenting path — update residuals
            for parent, edge_idx in path:
                fwd_edge = graph[parent][edge_idx]
                rev_edge = graph[fwd_edge.to][fwd_edge.rev]
                fwd_edge.cap -= bottleneck
                rev_edge.cap += bottleneck
            return bottleneck

        for i, edge in enumerate(graph[node]):
            if edge.cap > 0 and edge.to not in visited:
                visited.add(edge.to)
                new_bottleneck = min(bottleneck, edge.cap)
                stack.append((edge.to, new_bottleneck, path + [(node, i)]))

    return 0


def max_flow(
    graph: ResGraph,
    source: Any,
    sink: Any,
    demand: int = INF,
) -> int:
    """
    Run dfs_augmenting_path repeatedly until no augmenting path exists
    or total flow pushed >= demand (early termination).

    Returns the total flow pushed in this call.
    The graph residual state is updated in-place.
    """
    total = 0
    while total < demand:
        pushed = dfs_augmenting_path(graph, source, sink)
        if pushed == 0:
            break
        total += pushed
    return total


# ---------------------------------------------------------------------------
# Solver 1: Binary Search (BS)
# ---------------------------------------------------------------------------

def solve_bs(tvn: TVN, D: int, T_max: int) -> Tuple[int, float]:
    """
    Binary Search solver for MDDT.

    Performs binary search over the time horizon [1, T_max].
    At each candidate T = mid, builds the full TEG from scratch and runs
    max-flow. Returns the minimum T where max-flow >= D.

    Returns: (delivery_time, elapsed_wall_clock_seconds)
    """
    start = time.perf_counter()

    lo, hi = 1, T_max
    result = T_max

    while lo <= hi:
        mid = (lo + hi) // 2
        graph = build_full_teg(tvn, mid, D)
        flow = max_flow(graph, 'S', 'T', D)

        if flow >= D:
            result = mid
            hi = mid - 1
        else:
            lo = mid + 1

    elapsed = time.perf_counter() - start
    return result, elapsed


# ---------------------------------------------------------------------------
# Solver 2: Enhanced Binary Search (EBS)
# ---------------------------------------------------------------------------

def solve_ebs(tvn: TVN, D: int, T_max: int) -> Tuple[int, float]:
    """
    Enhanced Binary Search solver for MDDT.

    Like BS, but reuses the residual graph when the candidate T increases
    (i.e., extends the current graph with new time layers and continues
    flow from the existing residual state).

    When the candidate T decreases, the graph is rebuilt from scratch.

    Returns: (delivery_time, elapsed_wall_clock_seconds)
    """
    start = time.perf_counter()

    lo, hi = 1, T_max
    result = T_max

    # Maintain current residual graph state
    current_graph: ResGraph = new_residual_graph()
    current_T: int = -1          # highest time layer added so far
    current_flow: int = 0        # total flow pushed into current_graph

    while lo <= hi:
        mid = (lo + hi) // 2

        if mid > current_T:
            # Extend the current graph with new layers
            if current_T < 0:
                # First iteration — build from scratch up to mid
                current_graph = new_residual_graph()
                for t in range(mid + 1):
                    add_time_layer(current_graph, tvn, t, D)
                current_flow = 0
                current_T = mid
            else:
                # Add only the new layers
                for t in range(current_T + 1, mid + 1):
                    add_time_layer(current_graph, tvn, t, D)
                current_T = mid

            # Push additional flow on top of existing residual
            additional = max_flow(current_graph, 'S', 'T', D)
            current_flow += additional
            flow = current_flow

        else:
            # mid <= current_T: cannot shrink the residual, must rebuild
            current_graph = new_residual_graph()
            for t in range(mid + 1):
                add_time_layer(current_graph, tvn, t, D)
            current_T = mid
            current_flow = max_flow(current_graph, 'S', 'T', D)
            flow = current_flow

        if flow >= D:
            result = mid
            hi = mid - 1
        else:
            lo = mid + 1

    elapsed = time.perf_counter() - start
    return result, elapsed


# ---------------------------------------------------------------------------
# Solver 3: One-Pass Solver
# ---------------------------------------------------------------------------

def solve_one_pass(tvn: TVN, D: int) -> Tuple[int, float]:
    """
    One-Pass solver for MDDT.

    Incrementally expands the TEG one time step at a time, reusing the
    residual graph across all steps. Augments flow as new edges are added.
    Stops as soon as accumulated flow reaches D.

    This is the key contribution of the paper: a single pass over time
    instead of repeated full max-flow computations.

    Returns: (delivery_time, elapsed_wall_clock_seconds)
    """
    start = time.perf_counter()

    graph = new_residual_graph()
    total_flow = 0

    for t in range(tvn.T_max + 1):
        add_time_layer(graph, tvn, t, D)

        # Augment as much as possible with the current graph
        while True:
            pushed = dfs_augmenting_path(graph, 'S', 'T')
            if pushed == 0:
                break
            total_flow += pushed
            if total_flow >= D:
                elapsed = time.perf_counter() - start
                return t, elapsed

    # D not achievable within T_max
    elapsed = time.perf_counter() - start
    return tvn.T_max, elapsed
