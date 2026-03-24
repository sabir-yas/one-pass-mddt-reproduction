"""
plot_results.py
---------------
Generates Figure 2 (delivery time vs. data volume) and
Figure 3 (running time vs. data volume) from experiment results.

Usage:
    python plot_results.py            # reads results_agg.csv, saves figures
    python plot_results.py <csv_path> # use a custom CSV file
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


SOLVER_STYLES = {
    'BS':      {'color': 'royalblue',   'marker': 'o', 'label': 'Binary Search (BS)'},
    'EBS':     {'color': 'darkorange',  'marker': 's', 'label': 'Enhanced BS (EBS)'},
    'OnePass': {'color': 'forestgreen', 'marker': '^', 'label': 'One-Pass Solver'},
}


def plot_figure2(df: pd.DataFrame, output_path: str = 'figure2.png') -> None:
    """
    Figure 2: Minimum delivery time vs. data volume D.

    All three solvers should produce the same delivery time (optimality check).
    A warning is printed if any solver disagrees.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(df['D'], df['bs_T'],  **{k: v for k, v in SOLVER_STYLES['BS'].items()},
            linewidth=1.8, markersize=6)
    ax.plot(df['D'], df['ebs_T'], **{k: v for k, v in SOLVER_STYLES['EBS'].items()},
            linewidth=1.8, markersize=6, linestyle='--')
    ax.plot(df['D'], df['op_T'],  **{k: v for k, v in SOLVER_STYLES['OnePass'].items()},
            linewidth=1.8, markersize=6, linestyle=':')

    ax.set_xlabel('Data Volume D', fontsize=12)
    ax.set_ylabel('Minimum Delivery Time (time slots)', fontsize=12)
    ax.set_title('Figure 2: Data Delivery Time vs. Data Volume', fontsize=13)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output_path}")

    # Correctness check: all solvers should agree
    tol = 1  # allow rounding differences from averaging over trials
    mismatches = df[
        (abs(df['bs_T'] - df['ebs_T']) > tol) |
        (abs(df['bs_T'] - df['op_T'])  > tol)
    ]
    if not mismatches.empty:
        print(f"WARNING: Solver delivery-time mismatch at D values: "
              f"{mismatches['D'].tolist()}")
        print("  This may indicate a bug in one of the solvers.")
    else:
        print("Figure 2 check: all solvers agree on delivery time. (Correctness OK)")


def plot_figure3(df: pd.DataFrame, output_path: str = 'figure3.png') -> None:
    """
    Figure 3: Running time vs. data volume D.

    The One-Pass solver should have the lowest running time, showing its
    efficiency advantage over BS and EBS.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(df['D'], df['bs_time'],  **{k: v for k, v in SOLVER_STYLES['BS'].items()},
            linewidth=1.8, markersize=6)
    ax.plot(df['D'], df['ebs_time'], **{k: v for k, v in SOLVER_STYLES['EBS'].items()},
            linewidth=1.8, markersize=6, linestyle='--')
    ax.plot(df['D'], df['op_time'],  **{k: v for k, v in SOLVER_STYLES['OnePass'].items()},
            linewidth=1.8, markersize=6, linestyle=':')

    ax.set_xlabel('Data Volume D', fontsize=12)
    ax.set_ylabel('Running Time (seconds)', fontsize=12)
    ax.set_title('Figure 3: Running Time vs. Data Volume', fontsize=13)
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {output_path}")

    # Qualitative check
    if df['op_time'].mean() < df['bs_time'].mean():
        print("Figure 3 check: One-Pass is faster than BS on average. (Trend reproduced)")
    else:
        print("Figure 3 check: One-Pass is NOT faster than BS on average. "
              "Consider increasing D range or problem size.")


def main(csv_path: str = 'results_agg.csv') -> None:
    """Load aggregated results and generate both figures."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Run experiment.py first.")
        sys.exit(1)

    print(f"Loaded {len(df)} data points from {csv_path}")
    plot_figure2(df)
    plot_figure3(df)
    print("\nDone. Figures saved as figure2.png and figure3.png")


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'results_agg.csv'
    main(csv_path)
