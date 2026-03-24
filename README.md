# One-Pass MDDT Reproduction

Reproduction of the IEEE INFOCOM 2023 paper:
> *"One Pass is Sufficient: A Solver for Minimizing Data Delivery Time over Time-varying Networks"*

Implements three solvers for the MDDT (Minimizing Data Delivery Time) problem over
time-varying networks and reproduces the qualitative performance trends from Figures 2 and 3
of the paper.

---

## What This Does

- **Binary Search (BS)**: Baseline solver — binary search over time horizon, rebuilds the
  full Time-Expanded Graph (TEG) and runs max-flow from scratch at each step.
- **Enhanced Binary Search (EBS)**: Like BS but reuses the residual graph when the time
  horizon expands, avoiding some redundant computation.
- **One-Pass Solver**: The paper's contribution — incrementally expands the TEG one time
  step at a time, reusing the residual graph throughout. One pass over time.

**Figure 2** (delivery time vs. D): All three solvers find the same optimal answer — their
lines overlap, confirming correctness.

**Figure 3** (running time vs. D): One-Pass is significantly faster than BS and EBS,
especially at large data volumes — reproducing the paper's key result.

---

## Requirements

- Python 3.9+
- numpy, pandas, matplotlib

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/one-pass-mddt-reproduction.git
cd one-pass-mddt-reproduction

# 2. (Optional) create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Quick correctness check (~1 second)
Verifies all three solvers agree on a small network:
```bash
python experiment.py --smoke
```

### Small-scale experiment (~5–10 seconds)
Runs on a small network (8 nodes, 30 time slots), saves CSVs and figures:
```bash
python experiment.py --small
```

### Full experiment (~15–20 minutes)
Runs the full-scale experiment (15 nodes, 150 time slots, 3 trials, 15 D values).
Saves `results_raw.csv`, `results_agg.csv`, `figure2.png`, `figure3.png`:
```bash
python experiment.py --full
```

> **Note:** The full experiment is slow at large D values because the DFS-based
> Ford-Fulkerson pushes flow one augmenting path at a time. This is intentional —
> the point is to show the *relative* speedup of One-Pass over BS, not absolute speed.

### Generate figures from existing results
If you already have `results_agg.csv`:
```bash
python plot_results.py
```

---

## File Overview

| File | Purpose |
|------|---------|
| `network_generator.py` | Synthetic time-varying network (TVN) generation |
| `teg_builder.py` | Time-Expanded Graph construction and residual graph management |
| `solvers.py` | DFS max-flow core + BS, EBS, One-Pass solver implementations |
| `experiment.py` | Timed experiment loop, CSV output, correctness smoke test |
| `plot_results.py` | Figure 2 and Figure 3 generation |
| `requirements.txt` | Python dependencies |

---

## Synthetic Network Model

Since the paper uses proprietary STK-generated Starlink satellite traces (not publicly
available), we use a synthetic time-varying network:

- **Nodes**: Fixed set of `n` nodes (satellites/ground stations)
- **Links**: At each time slot `t`, each directed pair `(u, v)` is independently activated
  with probability `p` (Erdős–Rényi style), with a random integer capacity in `[cap_lo, cap_hi]`
- **Parameters**: `n_nodes=15`, `T_max=150`, `activation_prob=0.4`, `cap_lo=1`, `cap_hi=10`

This model preserves the key property of real satellite networks: links appear and disappear
over time, creating intermittent connectivity that the MDDT problem is designed to handle.

---

## Implementation Assumptions

The paper does not specify all implementation details. Our choices:

1. **Time discretization**: Equal-length time slots
2. **Augmenting path search**: DFS-based Ford-Fulkerson (iterative)
3. **Node storage**: Infinite buffering capacity between time slots
4. **TEG edges**: Transmission edge `(u,t) → (v,t+1)` with link capacity; storage edge
   `(v,t) → (v,t+1)` with infinite capacity
5. **Data source**: Synthetic random networks (not Starlink traces)

---

## Expected Output

After running the full experiment:

- **`figure2.png`**: All three solver lines overlap — they find the same minimum delivery time
- **`figure3.png`**: One-Pass line is below BS and EBS lines, gap widening as D increases

This reproduces the qualitative trend from Figure 3 of the original paper.

---

## Group Members

Aidan Campbell · Asael Garcia Cervantes · Yo Han Lee · Jabir Nure · Yaseer Sabir

CSCI 4800 — Reproduction of IEEE INFOCOM 2023
