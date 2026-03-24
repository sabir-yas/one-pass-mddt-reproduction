"""
teg_builder.py
--------------
Constructs and manages Time-Expanded Graphs (TEGs) and residual graphs
for the MDDT solvers.

Node naming convention:
  'S'       - super-source
  'T'       - super-sink
  (v, t)    - node v at time step t (0-indexed for both)

Edge representation uses the standard augmenting-path residual graph pattern:
each forward edge (u->v, cap) is paired with a reverse edge (v->u, cap=0).
The `rev` field on each Edge stores the index of its reverse edge in the
adjacency list of the opposite endpoint, enabling O(1) residual updates.
"""

from collections import defaultdict
from typing import Any, Dict, List

from network_generator import TVN

INF = 10 ** 9


class Edge:
    """A directed edge in the residual graph."""
    __slots__ = ['to', 'cap', 'rev']

    def __init__(self, to: Any, cap: int, rev: int):
        self.to = to    # destination node name
        self.cap = cap  # remaining residual capacity
        self.rev = rev  # index of reverse edge in graph[to]

    def __repr__(self):
        return f"Edge(to={self.to!r}, cap={self.cap}, rev={self.rev})"


# Type alias for the residual graph adjacency list
ResGraph = Dict[Any, List[Edge]]


def new_residual_graph() -> ResGraph:
    """Return a new empty residual graph (defaultdict of lists)."""
    return defaultdict(list)


def add_edge(graph: ResGraph, u: Any, v: Any, cap: int) -> None:
    """
    Add a directed edge u->v with the given capacity to the residual graph,
    along with its paired reverse edge v->u with capacity 0.

    Maintains the rev-index invariant:
      graph[u][fwd_idx].rev  == rev_idx  (points to the reverse edge)
      graph[v][rev_idx].rev  == fwd_idx  (points back to the forward edge)
    """
    fwd_idx = len(graph[u])
    rev_idx = len(graph[v])
    graph[u].append(Edge(to=v, cap=cap, rev=rev_idx))
    graph[v].append(Edge(to=u, cap=0,   rev=fwd_idx))


def add_time_layer(
    graph: ResGraph,
    tvn: TVN,
    t: int,
    flow_cap_S: int,
) -> None:
    """
    Add all nodes and edges for time step t to the residual graph (in-place).

    Steps:
    1. t == 0: add super-source edge 'S' -> (source, 0) with cap = flow_cap_S.
       This single edge enforces the total flow limit D.
    2. t > 0: add storage edges (v, t-1) -> (v, t) with cap = INF for all v.
       These represent buffering data at a node between time slots.
    3. Add transmission edges for each link active at time t:
       (u, t) -> (v, t+1) with cap = link.cap.
       Data sent during slot t arrives at the next slot (t+1).
    4. Add super-sink edge (dest, t) -> 'T' with cap = INF.
       Data can be delivered at any time step.
    """
    # Step 1: super-source edge (only at t=0)
    if t == 0:
        add_edge(graph, 'S', (tvn.source, 0), flow_cap_S)

    # Step 2: storage edges (buffer at node between consecutive time steps)
    if t > 0:
        for v in range(tvn.n_nodes):
            add_edge(graph, (v, t - 1), (v, t), INF)

    # Step 3: transmission edges for links active at time t
    for lnk in tvn.links_by_t.get(t, []):
        add_edge(graph, (lnk.u, t), (lnk.v, t + 1), lnk.cap)

    # Step 4: super-sink edge — data delivered at time t reaches sink
    add_edge(graph, (tvn.dest, t), 'T', INF)


def build_full_teg(tvn: TVN, T: int, flow_cap_S: int) -> ResGraph:
    """
    Build a complete TEG for time horizon [0, T] from scratch.

    Used by the Binary Search solver which rebuilds the graph at each
    candidate time horizon.
    """
    graph = new_residual_graph()
    for t in range(T + 1):
        add_time_layer(graph, tvn, t, flow_cap_S)
    return graph


def copy_residual_graph(graph: ResGraph) -> ResGraph:
    """
    Return a deep copy of the residual graph.

    Each Edge object is a new instance so that modifying the copy does not
    affect the original. The rev index invariant is preserved because we
    copy entire adjacency lists in order.
    """
    new_graph = defaultdict(list)
    for node, edges in graph.items():
        new_graph[node] = [Edge(e.to, e.cap, e.rev) for e in edges]
    return new_graph
