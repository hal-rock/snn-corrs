"""
Generate heatmaps of key metrics across a two-parameter sweep.

Extracts from each run:
- Mean correlation
- Noise fraction
- Explainable ICC

And plots heatmaps showing how each varies across the parameter space.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
import re

from analysis import analyze_parameter_point


def format_param_value(val) -> str:
    """Format a parameter value for directory name matching.

    Converts integer-like floats to int strings (1.0 -> "1").
    """
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)


def discover_sweep_structure(sweep_dir: str) -> Tuple[str, List[float], str, List[float]]:
    """
    Discover the parameter sweep structure from directory names.

    Expects structure like: sweep_dir/param1_val1/param2_val2/

    Returns:
        (param1_name, param1_values, param2_name, param2_values)
    """
    sweep_path = Path(sweep_dir)

    # Get first-level directories (first parameter)
    level1_dirs = sorted([d for d in sweep_path.iterdir() if d.is_dir()])

    if not level1_dirs:
        raise ValueError(f"No subdirectories found in {sweep_dir}")

    # Parse first parameter name and values
    # Use greedy match for name to handle underscores in variable names
    # e.g., "input_time_100" -> name="input_time", value="100"
    pattern = re.compile(r'^(.+)_([^_]+)$')
    param1_name = None
    param1_values = []

    for d in level1_dirs:
        match = pattern.match(d.name)
        if match:
            name, val = match.groups()
            if param1_name is None:
                param1_name = name
            elif name != param1_name:
                continue  # Skip non-matching directories
            try:
                param1_values.append(float(val))
            except ValueError:
                param1_values.append(val)

    # Get second-level directories (second parameter)
    sample_level1 = level1_dirs[0]
    level2_dirs = sorted([d for d in sample_level1.iterdir() if d.is_dir()])

    param2_name = None
    param2_values = []

    for d in level2_dirs:
        match = pattern.match(d.name)
        if match:
            name, val = match.groups()
            if param2_name is None:
                param2_name = name
            elif name != param2_name:
                continue
            try:
                param2_values.append(float(val))
            except ValueError:
                param2_values.append(val)

    # Sort values
    param1_values = sorted(set(param1_values))
    param2_values = sorted(set(param2_values))

    return param1_name, param1_values, param2_name, param2_values


def analyze_sweep(sweep_dir: str,
                  n_neurons_sample: int = 250,
                  n_bootstrap: int = 500,
                  seed: int = 42,
                  verbose: bool = True) -> dict:
    """
    Analyze all parameter points in a sweep.

    Returns dict with:
        - param1_name, param1_values
        - param2_name, param2_values
        - mean_corr: 2D array of mean correlations
        - noise_frac: 2D array of noise fractions
        - explainable_icc: 2D array of explainable ICC values
        - completed: 2D boolean array of which runs completed
    """
    sweep_path = Path(sweep_dir)

    # Discover structure
    param1_name, param1_values, param2_name, param2_values = discover_sweep_structure(sweep_dir)

    n1 = len(param1_values)
    n2 = len(param2_values)

    if verbose:
        print(f"Sweep structure: {param1_name} ({n1} values) x {param2_name} ({n2} values)")
        print(f"  {param1_name}: {param1_values}")
        print(f"  {param2_name}: {param2_values}")

    # Initialize result arrays
    mean_corr = np.full((n1, n2), np.nan)
    noise_frac = np.full((n1, n2), np.nan)
    explainable_icc = np.full((n1, n2), np.nan)
    completed = np.zeros((n1, n2), dtype=bool)

    # Analyze each point
    total_points = n1 * n2
    analyzed = 0

    for i, v1 in enumerate(param1_values):
        for j, v2 in enumerate(param2_values):
            # Construct directory path
            dir_name = sweep_path / f"{param1_name}_{format_param_value(v1)}" / f"{param2_name}_{format_param_value(v2)}"

            if not dir_name.exists():
                if verbose:
                    print(f"  [{analyzed+1}/{total_points}] {dir_name.name}: NOT FOUND")
                analyzed += 1
                continue

            # Check if run has completed (has stim directories with trials)
            stim_dirs = list(dir_name.glob("stim*"))
            if not stim_dirs:
                if verbose:
                    print(f"  [{analyzed+1}/{total_points}] {dir_name.name}: NO STIM DIRS")
                analyzed += 1
                continue

            # Check for trial directories in first stim
            trial_dirs = list(stim_dirs[0].glob("trial*"))
            if len(trial_dirs) < 10:  # Require at least some trials
                if verbose:
                    print(f"  [{analyzed+1}/{total_points}] {dir_name.name}: INCOMPLETE ({len(trial_dirs)} trials)")
                analyzed += 1
                continue

            try:
                if verbose:
                    print(f"  [{analyzed+1}/{total_points}] Analyzing {param1_name}={v1}, {param2_name}={v2}...")

                results = analyze_parameter_point(
                    str(dir_name),
                    n_neurons_sample=n_neurons_sample,
                    n_bootstrap=n_bootstrap,
                    analyze_inputs=False,
                    analyze_recurrent=False,
                    seed=seed,
                    verbose=False
                )

                stats = results.response_stats
                mean_corr[i, j] = np.mean(stats.correlations)
                noise_frac[i, j] = stats.mean_noise_var / stats.total_var
                explainable_icc[i, j] = stats.explainable_icc
                completed[i, j] = True

                if verbose:
                    print(f"    -> mean_corr={mean_corr[i,j]:.4f}, noise_frac={noise_frac[i,j]:.2%}, exp_icc={explainable_icc[i,j]:.4f}")

            except Exception as e:
                if verbose:
                    print(f"    -> ERROR: {e}")

            analyzed += 1

    return {
        'param1_name': param1_name,
        'param1_values': param1_values,
        'param2_name': param2_name,
        'param2_values': param2_values,
        'mean_corr': mean_corr,
        'noise_frac': noise_frac,
        'explainable_icc': explainable_icc,
        'completed': completed
    }


def plot_heatmaps(results: dict, fig_dir: str = "figures"):
    """
    Plot and save heatmaps for each metric.
    """
    fig_path = Path(fig_dir)
    fig_path.mkdir(exist_ok=True)

    param1_name = results['param1_name']
    param2_name = results['param2_name']
    param1_values = results['param1_values']
    param2_values = results['param2_values']

    metrics = [
        ('mean_corr', 'Mean Spike Correlation', 'RdBu_r'),
        ('noise_frac', 'Noise Fraction', 'viridis_r'),
        ('explainable_icc', 'Explainable ICC', 'viridis'),
    ]

    # Create combined figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (key, title, cmap) in zip(axes, metrics):
        data = results[key]

        # Mask NaN values
        masked_data = np.ma.masked_invalid(data)

        im = ax.imshow(masked_data, aspect='auto', cmap=cmap, origin='lower')

        # Set tick labels
        ax.set_xticks(range(len(param2_values)))
        ax.set_xticklabels([f"{v:.0f}" if isinstance(v, float) and v.is_integer() else f"{v}"
                           for v in param2_values], rotation=45, ha='right')
        ax.set_yticks(range(len(param1_values)))
        ax.set_yticklabels([f"{v}" for v in param1_values])

        ax.set_xlabel(param2_name)
        ax.set_ylabel(param1_name)
        ax.set_title(title)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)

        # Annotate cells with values
        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                if not np.isnan(data[i, j]):
                    val = data[i, j]
                    if key == 'noise_frac':
                        text = f"{val:.0%}"
                    else:
                        text = f"{val:.3f}"
                    # Choose text color based on background
                    color = 'white' if im.norm(val) < 0.5 else 'black'
                    ax.text(j, i, text, ha='center', va='center',
                            fontsize=7, color=color)

    plt.suptitle(f'Parameter Sweep: {param1_name} vs {param2_name}', fontsize=14)
    plt.tight_layout()

    combined_path = fig_path / 'sweep_heatmaps.png'
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined heatmap: {combined_path}")

    # Also save individual heatmaps
    for key, title, cmap in metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        data = results[key]
        masked_data = np.ma.masked_invalid(data)

        im = ax.imshow(masked_data, aspect='auto', cmap=cmap, origin='lower')

        ax.set_xticks(range(len(param2_values)))
        ax.set_xticklabels([f"{v:.0f}" if isinstance(v, float) and v.is_integer() else f"{v}"
                           for v in param2_values], rotation=45, ha='right')
        ax.set_yticks(range(len(param1_values)))
        ax.set_yticklabels([f"{v}" for v in param1_values])

        ax.set_xlabel(param2_name)
        ax.set_ylabel(param1_name)
        ax.set_title(title)

        cbar = plt.colorbar(im, ax=ax)

        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                if not np.isnan(data[i, j]):
                    val = data[i, j]
                    if key == 'noise_frac':
                        text = f"{val:.0%}"
                    else:
                        text = f"{val:.3f}"
                    color = 'white' if im.norm(val) < 0.5 else 'black'
                    ax.text(j, i, text, ha='center', va='center',
                            fontsize=8, color=color)

        plt.tight_layout()
        individual_path = fig_path / f'sweep_{key}.png'
        plt.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {key} heatmap: {individual_path}")

    plt.close('all')


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate heatmaps from parameter sweep')
    parser.add_argument('sweep_dir', type=str, nargs='?', default='outputs/full_sweep',
                        help='Path to sweep directory (default: outputs/full_sweep)')
    parser.add_argument('--n_neurons', type=int, default=250,
                        help='Number of neurons to sample (default: 250)')
    parser.add_argument('--n_bootstrap', type=int, default=500,
                        help='Bootstrap samples (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--fig_dir', type=str, default='figures',
                        help='Directory to save figures (default: figures)')

    args = parser.parse_args()

    print(f"Analyzing sweep: {args.sweep_dir}")
    print(f"Settings: n_neurons={args.n_neurons}, n_bootstrap={args.n_bootstrap}")
    print()

    results = analyze_sweep(
        args.sweep_dir,
        n_neurons_sample=args.n_neurons,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        verbose=True
    )

    n_completed = results['completed'].sum()
    n_total = results['completed'].size
    print(f"\nCompleted {n_completed}/{n_total} parameter points")

    if n_completed > 0:
        print("\nGenerating heatmaps...")
        plot_heatmaps(results, args.fig_dir)
        print("\nDone!")
    else:
        print("\nNo completed runs found - skipping heatmap generation")


if __name__ == '__main__':
    main()
