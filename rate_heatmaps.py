"""
Generate heatmaps of average firing rates across parameter sweeps.

Computes mean firing rates (spikes/second) separately for:
- Excitatory neurons
- Inhibitory neurons

Then plots heatmaps showing how these vary across the parameter space.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
import re

from analysis import load_spikes_vectorized, load_params, get_n_stimuli


# =============================================================================
# Sweep Traversal (adapted from sweep_heatmaps.py)
# =============================================================================

def format_param_value(val) -> str:
    """Format a parameter value for directory name matching."""
    if isinstance(val, float) and val.is_integer():
        return str(int(val))
    return str(val)


def discover_sweep_structure(sweep_dir: str) -> Tuple[str, List[float], str, List[float]]:
    """
    Discover the parameter sweep structure from directory names.

    Returns:
        (param1_name, param1_values, param2_name, param2_values)
    """
    sweep_path = Path(sweep_dir)
    level1_dirs = sorted([d for d in sweep_path.iterdir() if d.is_dir()])

    if not level1_dirs:
        raise ValueError(f"No subdirectories found in {sweep_dir}")

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
                continue
            try:
                param1_values.append(float(val))
            except ValueError:
                param1_values.append(val)

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

    param1_values = sorted(set(param1_values))
    param2_values = sorted(set(param2_values))

    return param1_name, param1_values, param2_name, param2_values


# =============================================================================
# Firing Rate Analysis
# =============================================================================

def compute_firing_rates(directory: str, params: dict) -> dict:
    """
    Compute average firing rates for excitatory and inhibitory neurons.

    If a no_input_mask.npy is present in `directory`, also reports rates
    split by whether the cell receives any external input. Cells with their
    input column zeroed (silenced) vs. driven (input-receiving) are reported
    separately for both E and I populations.

    Returns:
        Dict with keys:
          - 'exc_rate', 'inh_rate'  (always present, Hz)
          - 'exc_driven_rate', 'exc_silenced_rate',
            'inh_driven_rate', 'inh_silenced_rate'  (NaN if no mask, or if the
            corresponding subgroup is empty)
          - 'n_exc_silenced', 'n_inh_silenced'  (0 if no mask)
    """
    n_excite = params['n_excite']
    n_inhib = params['n_inhib']
    n_neurons = n_excite + n_inhib
    n_trials = params['n_trials']
    n_stimuli = get_n_stimuli(params, directory)

    # Time window for rate calculation (ms)
    start_time = 50  # Skip initial transient
    total_time = 1000
    duration_sec = (total_time - start_time) / 1000.0

    # Optional input-silencing mask (shape: n_neurons, True = silenced)
    mask_path = Path(directory) / 'no_input_mask.npy'
    no_input_mask = np.load(mask_path) if mask_path.exists() else None

    # Collect spike counts (neurons x trials) across stimuli
    exc_chunks = []
    inh_chunks = []

    for stim in range(n_stimuli):
        stim_dir = f"{directory}/stim{stim}"

        spikes = load_spikes_vectorized(
            stim_dir, n_trials, n_neurons,
            bin_size=total_time - start_time,
            total_time=total_time,
            start_from=start_time
        )
        # spikes shape: (n_neurons, n_trials)
        exc_chunks.append(spikes[:n_excite, :])
        inh_chunks.append(spikes[n_excite:, :])

    exc_counts = np.concatenate(exc_chunks, axis=1)  # (n_excite, n_trials*n_stimuli)
    inh_counts = np.concatenate(inh_chunks, axis=1)  # (n_inhib, ...)

    result = {
        'exc_rate': float(np.mean(exc_counts) / duration_sec),
        'inh_rate': float(np.mean(inh_counts) / duration_sec),
        'exc_driven_rate': np.nan,
        'exc_silenced_rate': np.nan,
        'inh_driven_rate': np.nan,
        'inh_silenced_rate': np.nan,
        'n_exc_silenced': 0,
        'n_inh_silenced': 0,
    }

    if no_input_mask is not None:
        e_silent = no_input_mask[:n_excite]
        i_silent = no_input_mask[n_excite:]
        result['n_exc_silenced'] = int(e_silent.sum())
        result['n_inh_silenced'] = int(i_silent.sum())
        if e_silent.any():
            result['exc_silenced_rate'] = float(np.mean(exc_counts[e_silent]) / duration_sec)
        if (~e_silent).any():
            result['exc_driven_rate'] = float(np.mean(exc_counts[~e_silent]) / duration_sec)
        if i_silent.any():
            result['inh_silenced_rate'] = float(np.mean(inh_counts[i_silent]) / duration_sec)
        if (~i_silent).any():
            result['inh_driven_rate'] = float(np.mean(inh_counts[~i_silent]) / duration_sec)

    return result


def analyze_sweep_firing_rates(sweep_dir: str, verbose: bool = True) -> dict:
    """
    Analyze firing rates across all parameter points in a sweep.

    Returns dict with:
        - param1_name, param1_values
        - param2_name, param2_values
        - exc_rate: 2D array of mean excitatory firing rates (Hz)
        - inh_rate: 2D array of mean inhibitory firing rates (Hz)
        - completed: 2D boolean array
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
    exc_rate = np.full((n1, n2), np.nan)
    inh_rate = np.full((n1, n2), np.nan)
    exc_driven_rate = np.full((n1, n2), np.nan)
    exc_silenced_rate = np.full((n1, n2), np.nan)
    inh_driven_rate = np.full((n1, n2), np.nan)
    inh_silenced_rate = np.full((n1, n2), np.nan)
    n_exc_silenced = np.zeros((n1, n2), dtype=int)
    n_inh_silenced = np.zeros((n1, n2), dtype=int)
    completed = np.zeros((n1, n2), dtype=bool)

    total_points = n1 * n2
    analyzed = 0

    for i, v1 in enumerate(param1_values):
        for j, v2 in enumerate(param2_values):
            dir_name = sweep_path / f"{param1_name}_{format_param_value(v1)}" / f"{param2_name}_{format_param_value(v2)}"

            if not dir_name.exists():
                if verbose:
                    print(f"  [{analyzed+1}/{total_points}] {dir_name.name}: NOT FOUND")
                analyzed += 1
                continue

            stim_dirs = list(dir_name.glob("stim*"))
            if not stim_dirs:
                if verbose:
                    print(f"  [{analyzed+1}/{total_points}] {dir_name.name}: NO STIM DIRS")
                analyzed += 1
                continue

            trial_dirs = list(stim_dirs[0].glob("trial*"))
            if len(trial_dirs) < 10:
                if verbose:
                    print(f"  [{analyzed+1}/{total_points}] {dir_name.name}: INCOMPLETE ({len(trial_dirs)} trials)")
                analyzed += 1
                continue

            try:
                if verbose:
                    print(f"  [{analyzed+1}/{total_points}] Analyzing {param1_name}={v1}, {param2_name}={v2}...")

                params = load_params(str(dir_name))
                rates = compute_firing_rates(str(dir_name), params)

                exc_rate[i, j] = rates['exc_rate']
                inh_rate[i, j] = rates['inh_rate']
                exc_driven_rate[i, j] = rates['exc_driven_rate']
                exc_silenced_rate[i, j] = rates['exc_silenced_rate']
                inh_driven_rate[i, j] = rates['inh_driven_rate']
                inh_silenced_rate[i, j] = rates['inh_silenced_rate']
                n_exc_silenced[i, j] = rates['n_exc_silenced']
                n_inh_silenced[i, j] = rates['n_inh_silenced']
                completed[i, j] = True

                if verbose:
                    msg = f"    -> E: {rates['exc_rate']:.1f} Hz, I: {rates['inh_rate']:.1f} Hz"
                    if rates['n_exc_silenced'] > 0 or rates['n_inh_silenced'] > 0:
                        msg += (f"  [E driven={rates['exc_driven_rate']:.1f} "
                                f"silenced={rates['exc_silenced_rate']:.1f}; "
                                f"I driven={rates['inh_driven_rate']:.1f} "
                                f"silenced={rates['inh_silenced_rate']:.1f}]")
                    print(msg)

            except Exception as e:
                if verbose:
                    print(f"    -> ERROR: {e}")

            analyzed += 1

    return {
        'param1_name': param1_name,
        'param1_values': param1_values,
        'param2_name': param2_name,
        'param2_values': param2_values,
        'exc_rate': exc_rate,
        'inh_rate': inh_rate,
        'exc_driven_rate': exc_driven_rate,
        'exc_silenced_rate': exc_silenced_rate,
        'inh_driven_rate': inh_driven_rate,
        'inh_silenced_rate': inh_silenced_rate,
        'n_exc_silenced': n_exc_silenced,
        'n_inh_silenced': n_inh_silenced,
        'completed': completed
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_heatmaps(results: dict, fig_dir: str = "figures"):
    """
    Plot and save heatmaps for firing rates.
    """
    fig_path = Path(fig_dir)
    fig_path.mkdir(exist_ok=True)

    param1_name = results['param1_name']
    param2_name = results['param2_name']
    param1_values = results['param1_values']
    param2_values = results['param2_values']

    metrics = [
        ('exc_rate', 'Excitatory Firing Rate (Hz)', 'YlOrRd'),
        ('inh_rate', 'Inhibitory Firing Rate (Hz)', 'YlGnBu'),
    ]

    # Create combined figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (key, title, cmap) in zip(axes, metrics):
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
        cbar.set_label('Hz')

        # Annotate cells
        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                if not np.isnan(data[i, j]):
                    val = data[i, j]
                    text = f"{val:.1f}"
                    # Choose text color based on normalized value
                    norm_val = (val - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-10)
                    color = 'white' if norm_val > 0.6 else 'black'
                    ax.text(j, i, text, ha='center', va='center',
                            fontsize=8, color=color)

    plt.suptitle(f'Firing Rates: {param1_name} vs {param2_name}', fontsize=14)
    plt.tight_layout()

    combined_path = fig_path / 'firing_rate_heatmaps.png'
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined heatmap: {combined_path}")

    # Save individual heatmaps
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
        cbar.set_label('Hz')

        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                if not np.isnan(data[i, j]):
                    val = data[i, j]
                    text = f"{val:.1f}"
                    norm_val = (val - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-10)
                    color = 'white' if norm_val > 0.6 else 'black'
                    ax.text(j, i, text, ha='center', va='center',
                            fontsize=8, color=color)

        plt.tight_layout()
        individual_path = fig_path / f'{key}_heatmap.png'
        plt.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {key} heatmap: {individual_path}")

    plt.close('all')


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate heatmaps of firing rates across parameter sweep'
    )
    parser.add_argument('sweep_dir', type=str, nargs='?', default='outputs/full_sweep',
                        help='Path to sweep directory (default: outputs/full_sweep)')
    parser.add_argument('--fig_dir', type=str, default='figures',
                        help='Directory to save figures (default: figures)')

    args = parser.parse_args()

    print(f"Analyzing sweep: {args.sweep_dir}")
    print()

    results = analyze_sweep_firing_rates(args.sweep_dir, verbose=True)

    n_completed = results['completed'].sum()
    n_total = results['completed'].size
    print(f"\nCompleted {n_completed}/{n_total} parameter points")

    if n_completed > 0:
        print("\nGenerating heatmaps...")
        plot_heatmaps(results, args.fig_dir)

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY: Firing rates across completed parameter points")
        print("="*60)
        for key, label in [('exc_rate', 'Excitatory'), ('inh_rate', 'Inhibitory')]:
            data = results[key]
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                print(f"  {label:24s}: mean={np.mean(valid):.1f} Hz, "
                      f"range=[{np.min(valid):.1f}, {np.max(valid):.1f}] Hz")

        # Driven vs. silenced breakdown (only for points that had a no_input_mask)
        any_silencing = (results['n_exc_silenced'].sum() + results['n_inh_silenced'].sum()) > 0
        if any_silencing:
            print("-" * 60)
            print("  Driven vs. silenced (only points with input silencing):")
            split_metrics = [
                ('exc_driven_rate',   'Excitatory (driven)'),
                ('exc_silenced_rate', 'Excitatory (silenced)'),
                ('inh_driven_rate',   'Inhibitory (driven)'),
                ('inh_silenced_rate', 'Inhibitory (silenced)'),
            ]
            for key, label in split_metrics:
                data = results[key]
                valid = data[~np.isnan(data)]
                if len(valid) > 0:
                    print(f"  {label:24s}: mean={np.mean(valid):.1f} Hz, "
                          f"range=[{np.min(valid):.1f}, {np.max(valid):.1f}] Hz")
            # Driven - silenced gap, per point, then aggregated
            e_gap = results['exc_driven_rate'] - results['exc_silenced_rate']
            i_gap = results['inh_driven_rate'] - results['inh_silenced_rate']
            e_gap_valid = e_gap[~np.isnan(e_gap)]
            i_gap_valid = i_gap[~np.isnan(i_gap)]
            if len(e_gap_valid) > 0:
                print(f"  {'E driven - silenced':24s}: mean={np.mean(e_gap_valid):+.2f} Hz, "
                      f"range=[{np.min(e_gap_valid):+.2f}, {np.max(e_gap_valid):+.2f}] Hz")
            if len(i_gap_valid) > 0:
                print(f"  {'I driven - silenced':24s}: mean={np.mean(i_gap_valid):+.2f} Hz, "
                      f"range=[{np.min(i_gap_valid):+.2f}, {np.max(i_gap_valid):+.2f}] Hz")
        print("="*60)
        print("\nDone!")
    else:
        print("\nNo completed runs found - skipping heatmap generation")


if __name__ == '__main__':
    main()
