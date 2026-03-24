"""
experiment.py
-------------
Drives the timed experiments comparing BS, EBS, and One-Pass solvers.

Usage:
    python experiment.py              # run full experiment + generate figures
    python experiment.py --smoke      # run correctness smoke test only
    python experiment.py --small      # run small-scale experiment (fast)
"""

import sys
import time
import pandas as pd

from network_generator import TVN, generate_tvn, compute_d_values
from solvers import solve_bs, solve_ebs, solve_one_pass


# ---------------------------------------------------------------------------
# Correctness smoke test
# ---------------------------------------------------------------------------

def run_smoke_test():
    """
    Verify all three solvers return the same delivery time on a small TVN.
    Raises AssertionError if any solver disagrees.
    """
    print("Running correctness smoke test...")
    tvn = generate_tvn(
        n_nodes=5, T_max=20, activation_prob=0.5,
        cap_lo=1, cap_hi=5, source=0, dest=4, seed=0
    )

    # Find achievable D values
    d_test = [1, 3, 5, 8, 12, 20]

    all_pass = True
    for D in d_test:
        t_bs,  _ = solve_bs(tvn, D, tvn.T_max)
        t_ebs, _ = solve_ebs(tvn, D, tvn.T_max)
        t_op,  _ = solve_one_pass(tvn, D)

        match = (t_bs == t_ebs == t_op)
        status = "PASS" if match else "FAIL"
        print(f"  D={D:3d}: BS={t_bs:3d}, EBS={t_ebs:3d}, OnePass={t_op:3d}  [{status}]")

        if not match:
            all_pass = False

    if all_pass:
        print("Smoke test PASSED: all solvers agree.\n")
    else:
        print("Smoke test FAILED: solvers disagree on some D values.\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_single_trial(tvn: TVN, D: int, T_max: int) -> dict:
    """
    Run all three solvers on the same TVN with data volume D.
    Returns a dict with timing and delivery-time results.
    """
    t_bs,  bs_time  = solve_bs(tvn, D, T_max)
    t_ebs, ebs_time = solve_ebs(tvn, D, T_max)
    t_op,  op_time  = solve_one_pass(tvn, D)

    return {
        'D':        D,
        'bs_T':     t_bs,
        'ebs_T':    t_ebs,
        'op_T':     t_op,
        'bs_time':  bs_time,
        'ebs_time': ebs_time,
        'op_time':  op_time,
    }


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

def run_experiment(
    n_nodes: int = 15,
    T_max: int = 150,
    activation_prob: float = 0.4,
    cap_lo: int = 1,
    cap_hi: int = 10,
    n_trials: int = 3,
    n_points: int = 15,
    seed_base: int = 42,
) -> pd.DataFrame:
    """
    Main experiment driver.

    For each D in d_values:
      For each trial i: generate a fresh random TVN (different seed per trial),
      run all three solvers, record timing and delivery time.

    Aggregates by averaging over trials for each D.
    Saves raw records to results_raw.csv and aggregated to results_agg.csv.
    Returns the aggregated DataFrame.
    """
    # Generate a reference TVN to determine D values
    ref_tvn = generate_tvn(
        n_nodes=n_nodes, T_max=T_max, activation_prob=activation_prob,
        cap_lo=cap_lo, cap_hi=cap_hi, source=0, dest=n_nodes - 1,
        seed=seed_base,
    )
    d_values = compute_d_values(ref_tvn, n_points=n_points)
    print(f"Experiment parameters: n_nodes={n_nodes}, T_max={T_max}, "
          f"activation_prob={activation_prob}, n_trials={n_trials}")
    print(f"D values ({len(d_values)}): {d_values}\n")

    records = []
    total_start = time.perf_counter()

    for d_idx, D in enumerate(d_values):
        d_start = time.perf_counter()
        print(f"  D={D:5d}  ({d_idx+1}/{len(d_values)}) ...", end='', flush=True)

        for trial in range(n_trials):
            seed = seed_base + trial * 1000
            tvn = generate_tvn(
                n_nodes=n_nodes, T_max=T_max, activation_prob=activation_prob,
                cap_lo=cap_lo, cap_hi=cap_hi, source=0, dest=n_nodes - 1,
                seed=seed,
            )
            row = run_single_trial(tvn, D, T_max)
            row['trial'] = trial
            records.append(row)

        d_elapsed = time.perf_counter() - d_start
        print(f" done ({d_elapsed:.2f}s)")

    total_elapsed = time.perf_counter() - total_start
    print(f"\nTotal experiment time: {total_elapsed:.1f}s")

    raw_df = pd.DataFrame(records)
    raw_df.to_csv('results_raw.csv', index=False)
    print("Saved results_raw.csv")

    agg_df = (
        raw_df
        .groupby('D', as_index=False)
        .agg(
            bs_T=('bs_T', 'mean'),
            ebs_T=('ebs_T', 'mean'),
            op_T=('op_T', 'mean'),
            bs_time=('bs_time', 'mean'),
            ebs_time=('ebs_time', 'mean'),
            op_time=('op_time', 'mean'),
        )
    )
    agg_df.to_csv('results_agg.csv', index=False)
    print("Saved results_agg.csv")

    return agg_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else '--full'

    if mode == '--smoke':
        run_smoke_test()

    elif mode == '--small':
        run_smoke_test()
        run_experiment(
            n_nodes=8, T_max=30, activation_prob=0.4,
            cap_lo=1, cap_hi=5, n_trials=2, n_points=8, seed_base=42,
        )
        import plot_results
        plot_results.main()

    else:  # --full
        run_smoke_test()
        run_experiment(
            n_nodes=15, T_max=150, activation_prob=0.4,
            cap_lo=1, cap_hi=10, n_trials=3, n_points=15, seed_base=42,
        )
        import plot_results
        plot_results.main()
