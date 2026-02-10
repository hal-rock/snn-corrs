"""
Generate heatmaps of shared input-noise correlation relationships across parameter sweeps.

For each parameter point, computes Pearson correlation between:
- Pairwise spike (noise) correlations
- Shared input magnitudes (feedforward, excitatory, inhibitory, total recurrent)

Then plots heatmaps showing how these relationships vary across the parameter space.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
from scipy import stats
import re

from analysis import load_spikes_vectorized, load_connectivity, load_params, get_n_stimuli


# =============================================================================
# Partial Correlation
# =============================================================================

def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """
    Compute partial correlation r(X, Y | Z) using regression residuals.

    Returns the correlation between X and Y after removing the linear effect of Z
    from both variables.

    Args:
        x: First variable (e.g., noise correlations)
        y: Second variable (e.g., exc shared input)
        z: Control variable (e.g., inh shared input)

    Returns:
        (r, p): Partial correlation coefficient and p-value
    """
    # Regress X on Z and get residuals
    slope_xz, intercept_xz = np.polyfit(z, x, 1)
    resid_x = x - (slope_xz * z + intercept_xz)

    # Regress Y on Z and get residuals
    slope_yz, intercept_yz = np.polyfit(z, y, 1)
    resid_y = y - (slope_yz * z + intercept_yz)

    # Correlate residuals
    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p


# =============================================================================
# Sweep Traversal (adapted from sweep_heatmaps.py)
# =============================================================================

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


# =============================================================================
# Shared Input Analysis (adapted from analyze_shared_input.py)
# =============================================================================

def compute_spike_correlations(directory: str, params: dict,
                                neuron_indices: np.ndarray,
                                bin_size: float = 1000,
                                total_time: float = 1000,
                                start_from: float = 50) -> np.ndarray:
    """
    Compute mean pairwise spike correlations across stimuli for selected neurons.

    Args:
        directory: Path to parameter point directory
        params: Simulation parameters dict
        neuron_indices: Indices of neurons to analyze
        bin_size: Time bin size in ms for spike counting (default: 1000)
        total_time: Total simulation time in ms (default: 1000)
        start_from: Discard spikes before this time in ms (default: 50)

    Returns correlations for all pairs of the selected neurons.
    """
    n_neurons_total = params['n_excite'] + params['n_inhib']
    n_stimuli = get_n_stimuli(params, directory)
    n_trials = params['n_trials']
    n_selected = len(neuron_indices)

    # Collect spike counts: (n_selected, n_stimuli, n_trials)
    n_bins = int((total_time - start_from) / bin_size)
    all_responses = np.zeros((n_selected, n_stimuli, n_trials * n_bins))

    for stim in range(n_stimuli):
        stim_dir = f"{directory}/stim{stim}"
        spikes = load_spikes_vectorized(
            stim_dir, n_trials, n_neurons_total,
            bin_size=bin_size, total_time=total_time, start_from=start_from
        )
        # spikes shape: (n_neurons, n_trials * n_bins) when bin_size < total_time
        # Select our neurons
        all_responses[:, stim, :] = spikes[neuron_indices]

    # Compute correlations for each stimulus, then average
    n_pairs = n_selected * (n_selected - 1) // 2
    tril_indices = np.tril_indices(n_selected, k=-1)

    all_corrs = np.zeros((n_stimuli, n_pairs))
    for stim in range(n_stimuli):
        corr_mat = np.corrcoef(all_responses[:, stim, :])
        corr_mat[np.isnan(corr_mat)] = 0
        all_corrs[stim] = corr_mat[tril_indices]

    # Average across stimuli
    mean_corrs = np.mean(all_corrs, axis=0)
    return mean_corrs, tril_indices


def compute_shared_inputs(input_conn: np.ndarray, recurrent_conn: np.ndarray,
                          neuron_indices: np.ndarray, n_excite: int):
    """
    Compute shared input magnitudes for all pairs of selected neurons.

    Shared input is computed as: sum_k(w_ki * w_kj) for presynaptic neurons k.

    Returns:
        feedforward: Shared feedforward input for each pair
        recurrent_exc: Shared excitatory recurrent input
        recurrent_inh: Shared inhibitory recurrent input
        recurrent_total: Total shared recurrent input (exc - inh)
    """
    n_selected = len(neuron_indices)

    # Extract connectivity to selected neurons
    # input_conn: (n_inputs, n_neurons) -> weights from input to network
    # recurrent_conn: (n_neurons, n_neurons) -> weights from network to network
    # Convention: conn[j, i] = weight from i to j

    input_to_selected = input_conn[:, neuron_indices]  # (n_inputs, n_selected)
    recurrent_to_selected = recurrent_conn[neuron_indices]  # (n_selected, n_neurons)

    # Separate excitatory and inhibitory recurrent
    exc_to_selected = recurrent_to_selected[:, :n_excite]  # (n_selected, n_excite)
    inh_to_selected = recurrent_to_selected[:, n_excite:]  # (n_selected, n_inhib)

    # Compute shared input for each pair
    # For pair (i, j): shared_input = sum_k(w_ki * w_kj)
    # This can be computed as: (W @ W.T)[i, j] for the off-diagonal elements

    # Feedforward shared input
    ff_shared_matrix = input_to_selected.T @ input_to_selected  # (n_selected, n_selected)

    # Excitatory recurrent shared input
    exc_shared_matrix = exc_to_selected @ exc_to_selected.T

    # Inhibitory recurrent shared input
    inh_shared_matrix = inh_to_selected @ inh_to_selected.T

    # Total recurrent (exc - inh, since inhibition reduces correlations)
    total_rec_shared_matrix = exc_shared_matrix - inh_shared_matrix

    # Extract lower triangular (pairs)
    tril = np.tril_indices(n_selected, k=-1)

    feedforward = ff_shared_matrix[tril]
    recurrent_exc = exc_shared_matrix[tril]
    recurrent_inh = inh_shared_matrix[tril]
    recurrent_total = total_rec_shared_matrix[tril]

    return feedforward, recurrent_exc, recurrent_inh, recurrent_total


def analyze_shared_input_correlations(directory: str,
                                       n_neurons_sample: int = 200,
                                       seed: int = 42,
                                       bin_size: float = 1000,
                                       total_time: float = 1000,
                                       start_from: float = 50) -> dict:
    """
    Compute correlations between spike correlations and shared input types.

    Args:
        directory: Path to parameter point directory
        n_neurons_sample: Number of neurons to sample
        seed: Random seed for reproducibility
        bin_size: Time bin size in ms for spike counting (default: 1000)
        total_time: Total simulation time in ms (default: 1000)
        start_from: Discard spikes before this time in ms (default: 50)

    Returns dict with Pearson r values for each shared input type.
    """
    np.random.seed(seed)

    params = load_params(directory)
    # Analyze single stimulus for noise correlations
    params_copy = params.copy()
    params_copy['n_stimuli'] = 1

    input_conn, recurrent_conn = load_connectivity(directory)

    # Adjust for different synapse timescales of inhibition
    e_to_i_ratio = 2 / (3.3 * 2.26)
    recurrent_conn_adj = recurrent_conn.copy()
    recurrent_conn_adj[:, params['n_excite']:] *= -e_to_i_ratio

    n_excite = params['n_excite']
    n_inhib = params['n_inhib']
    n_neurons = n_excite + n_inhib

    # Sample neurons
    neuron_indices = np.random.choice(n_neurons, n_neurons_sample, replace=False)

    # Compute spike correlations
    spike_corrs, _ = compute_spike_correlations(
        directory, params_copy, neuron_indices,
        bin_size=bin_size, total_time=total_time, start_from=start_from
    )

    # Compute shared input magnitudes
    ff_shared, exc_shared, inh_shared, total_rec_shared = compute_shared_inputs(
        input_conn, recurrent_conn_adj, neuron_indices, n_excite
    )

    # Compute Pearson correlations
    results = {}

    for name, shared in [('feedforward', ff_shared),
                          ('excitatory', exc_shared),
                          ('inhibitory', inh_shared),
                          ('total_recurrent', total_rec_shared)]:
        r, p = stats.pearsonr(shared, spike_corrs)
        results[name] = {'r': r, 'p': p}

    # Compute partial correlations for E and I, controlling for each other
    # r(noise_corr, exc_shared | inh_shared)
    r_exc_partial, p_exc_partial = partial_correlation(spike_corrs, exc_shared, inh_shared)
    results['excitatory_partial'] = {'r': r_exc_partial, 'p': p_exc_partial}

    # r(noise_corr, inh_shared | exc_shared)
    r_inh_partial, p_inh_partial = partial_correlation(spike_corrs, inh_shared, exc_shared)
    results['inhibitory_partial'] = {'r': r_inh_partial, 'p': p_inh_partial}

    return results


# =============================================================================
# Sweep Analysis
# =============================================================================

def analyze_sweep_shared_input(sweep_dir: str,
                                n_neurons_sample: int = 200,
                                seed: int = 42,
                                verbose: bool = True,
                                bin_size: float = 1000,
                                total_time: float = 1000,
                                start_from: float = 50) -> dict:
    """
    Analyze shared input-noise correlation relationships across all parameter points.

    Args:
        sweep_dir: Path to sweep directory
        n_neurons_sample: Number of neurons to sample for analysis
        seed: Random seed for reproducibility
        verbose: Print progress
        bin_size: Time bin size in ms for spike counting (default: 1000, full trial)
        total_time: Total simulation time in ms (default: 1000)
        start_from: Discard spikes before this time in ms (default: 50)

    Returns dict with:
        - param1_name, param1_values
        - param2_name, param2_values
        - r_feedforward: 2D array of Pearson r values
        - r_excitatory: 2D array
        - r_inhibitory: 2D array
        - r_total_recurrent: 2D array
        - r_excitatory_partial: 2D array (controlling for I)
        - r_inhibitory_partial: 2D array (controlling for E)
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
    r_feedforward = np.full((n1, n2), np.nan)
    r_excitatory = np.full((n1, n2), np.nan)
    r_inhibitory = np.full((n1, n2), np.nan)
    r_total_recurrent = np.full((n1, n2), np.nan)
    r_excitatory_partial = np.full((n1, n2), np.nan)
    r_inhibitory_partial = np.full((n1, n2), np.nan)
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

            # Check if run has completed
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

                results = analyze_shared_input_correlations(
                    str(dir_name),
                    n_neurons_sample=n_neurons_sample,
                    seed=seed,
                    bin_size=bin_size,
                    total_time=total_time,
                    start_from=start_from
                )

                r_feedforward[i, j] = results['feedforward']['r']
                r_excitatory[i, j] = results['excitatory']['r']
                r_inhibitory[i, j] = results['inhibitory']['r']
                r_total_recurrent[i, j] = results['total_recurrent']['r']
                r_excitatory_partial[i, j] = results['excitatory_partial']['r']
                r_inhibitory_partial[i, j] = results['inhibitory_partial']['r']
                completed[i, j] = True

                if verbose:
                    print(f"    -> r_ff={r_feedforward[i,j]:.3f}, r_exc={r_excitatory[i,j]:.3f}, "
                          f"r_inh={r_inhibitory[i,j]:.3f}, r_tot={r_total_recurrent[i,j]:.3f}")
                    print(f"    -> r_exc|inh={r_excitatory_partial[i,j]:.3f}, "
                          f"r_inh|exc={r_inhibitory_partial[i,j]:.3f}")

            except Exception as e:
                if verbose:
                    print(f"    -> ERROR: {e}")

            analyzed += 1

    return {
        'param1_name': param1_name,
        'param1_values': param1_values,
        'param2_name': param2_name,
        'param2_values': param2_values,
        'r_feedforward': r_feedforward,
        'r_excitatory': r_excitatory,
        'r_inhibitory': r_inhibitory,
        'r_total_recurrent': r_total_recurrent,
        'r_excitatory_partial': r_excitatory_partial,
        'r_inhibitory_partial': r_inhibitory_partial,
        'completed': completed
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_heatmaps(results: dict, fig_dir: str = "figures"):
    """
    Plot and save heatmaps for shared input-correlation relationships.
    """
    fig_path = Path(fig_dir)
    fig_path.mkdir(exist_ok=True)

    param1_name = results['param1_name']
    param2_name = results['param2_name']
    param1_values = results['param1_values']
    param2_values = results['param2_values']

    metrics = [
        ('r_feedforward', 'r(Noise Corr, FF Shared Input)', 'RdBu_r'),
        ('r_excitatory', 'r(Noise Corr, Exc Shared Input)', 'RdBu_r'),
        ('r_inhibitory', 'r(Noise Corr, Inh Shared Input)', 'RdBu_r'),
        ('r_total_recurrent', 'r(Noise Corr, Total Rec Shared)', 'RdBu_r'),
    ]

    # Create combined figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Use symmetric color limits centered at 0 for correlation coefficients
    all_r_values = np.concatenate([
        results['r_feedforward'].flatten(),
        results['r_excitatory'].flatten(),
        results['r_inhibitory'].flatten(),
        results['r_total_recurrent'].flatten()
    ])
    all_r_values = all_r_values[~np.isnan(all_r_values)]
    if len(all_r_values) > 0:
        vmax = np.max(np.abs(all_r_values)) * 1.1
        vmin = -vmax
    else:
        vmin, vmax = -1, 1

    for ax, (key, title, cmap) in zip(axes, metrics):
        data = results[key]
        masked_data = np.ma.masked_invalid(data)

        im = ax.imshow(masked_data, aspect='auto', cmap=cmap, origin='lower',
                       vmin=vmin, vmax=vmax)

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
        cbar.set_label('Pearson r')

        # Annotate cells with values
        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                if not np.isnan(data[i, j]):
                    val = data[i, j]
                    text = f"{val:.2f}"
                    # Choose text color based on background
                    color = 'white' if abs(val) > vmax * 0.6 else 'black'
                    ax.text(j, i, text, ha='center', va='center',
                            fontsize=8, color=color)

    plt.suptitle(f'Shared Input - Noise Correlation Relationships\n{param1_name} vs {param2_name}',
                 fontsize=14)
    plt.tight_layout()

    combined_path = fig_path / 'shared_input_corr_heatmaps.png'
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined heatmap: {combined_path}")

    # Create separate figure for partial correlations (E/I controlling for each other)
    partial_metrics = [
        ('r_excitatory_partial', 'r(Noise Corr, Exc Shared | Inh Shared)', 'RdBu_r'),
        ('r_inhibitory_partial', 'r(Noise Corr, Inh Shared | Exc Shared)', 'RdBu_r'),
    ]

    # Compute color limits for partial correlations
    partial_r_values = np.concatenate([
        results['r_excitatory_partial'].flatten(),
        results['r_inhibitory_partial'].flatten()
    ])
    partial_r_values = partial_r_values[~np.isnan(partial_r_values)]
    if len(partial_r_values) > 0:
        vmax_partial = np.max(np.abs(partial_r_values)) * 1.1
        vmin_partial = -vmax_partial
    else:
        vmin_partial, vmax_partial = -1, 1

    fig_partial, axes_partial = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (key, title, cmap) in zip(axes_partial, partial_metrics):
        data = results[key]
        masked_data = np.ma.masked_invalid(data)

        im = ax.imshow(masked_data, aspect='auto', cmap=cmap, origin='lower',
                       vmin=vmin_partial, vmax=vmax_partial)

        ax.set_xticks(range(len(param2_values)))
        ax.set_xticklabels([f"{v:.0f}" if isinstance(v, float) and v.is_integer() else f"{v}"
                           for v in param2_values], rotation=45, ha='right')
        ax.set_yticks(range(len(param1_values)))
        ax.set_yticklabels([f"{v}" for v in param1_values])

        ax.set_xlabel(param2_name)
        ax.set_ylabel(param1_name)
        ax.set_title(title)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Partial r')

        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                if not np.isnan(data[i, j]):
                    val = data[i, j]
                    text = f"{val:.2f}"
                    color = 'white' if abs(val) > vmax_partial * 0.6 else 'black'
                    ax.text(j, i, text, ha='center', va='center',
                            fontsize=8, color=color)

    plt.suptitle(f'Partial Correlations (E/I Controlling for Each Other)\n{param1_name} vs {param2_name}',
                 fontsize=14)
    plt.tight_layout()

    partial_path = fig_path / 'partial_corr_heatmaps.png'
    plt.savefig(partial_path, dpi=150, bbox_inches='tight')
    print(f"Saved partial correlation heatmap: {partial_path}")

    # Also save individual heatmaps (including partial correlations)
    all_metrics = metrics + partial_metrics
    for key, title, cmap in all_metrics:
        # Use appropriate color limits
        if 'partial' in key:
            v_min, v_max = vmin_partial, vmax_partial
        else:
            v_min, v_max = vmin, vmax
        fig, ax = plt.subplots(figsize=(8, 6))
        data = results[key]
        masked_data = np.ma.masked_invalid(data)

        im = ax.imshow(masked_data, aspect='auto', cmap=cmap, origin='lower',
                       vmin=v_min, vmax=v_max)

        ax.set_xticks(range(len(param2_values)))
        ax.set_xticklabels([f"{v:.0f}" if isinstance(v, float) and v.is_integer() else f"{v}"
                           for v in param2_values], rotation=45, ha='right')
        ax.set_yticks(range(len(param1_values)))
        ax.set_yticklabels([f"{v}" for v in param1_values])

        ax.set_xlabel(param2_name)
        ax.set_ylabel(param1_name)
        ax.set_title(title)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Partial r' if 'partial' in key else 'Pearson r')

        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                if not np.isnan(data[i, j]):
                    val = data[i, j]
                    text = f"{val:.2f}"
                    color = 'white' if abs(val) > v_max * 0.6 else 'black'
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
        description='Generate heatmaps of shared input-noise correlation relationships across parameter sweep'
    )
    parser.add_argument('sweep_dir', type=str, nargs='?', default='outputs/full_sweep',
                        help='Path to sweep directory (default: outputs/full_sweep)')
    parser.add_argument('--n_neurons', type=int, default=200,
                        help='Number of neurons to sample (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--fig_dir', type=str, default='figures',
                        help='Directory to save figures (default: figures)')
    parser.add_argument('--bin_size', type=float, default=1000,
                        help='Time bin size in ms for spike counting (default: 1000, full trial)')
    parser.add_argument('--total_time', type=float, default=1000,
                        help='Total simulation time in ms (default: 1000)')
    parser.add_argument('--start_from', type=float, default=50,
                        help='Discard spikes before this time in ms (default: 50)')

    args = parser.parse_args()

    print(f"Analyzing sweep: {args.sweep_dir}")
    print(f"Settings: n_neurons={args.n_neurons}, seed={args.seed}, bin_size={args.bin_size}ms")
    print()

    results = analyze_sweep_shared_input(
        args.sweep_dir,
        n_neurons_sample=args.n_neurons,
        seed=args.seed,
        verbose=True,
        bin_size=args.bin_size,
        total_time=args.total_time,
        start_from=args.start_from
    )

    n_completed = results['completed'].sum()
    n_total = results['completed'].size
    print(f"\nCompleted {n_completed}/{n_total} parameter points")

    if n_completed > 0:
        print("\nGenerating heatmaps...")
        plot_heatmaps(results, args.fig_dir)

        # Print summary statistics
        print("\n" + "="*70)
        print("SUMMARY: Mean r values across completed parameter points")
        print("="*70)
        for key in ['r_feedforward', 'r_excitatory', 'r_inhibitory', 'r_total_recurrent']:
            data = results[key]
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                print(f"  {key:25s}: mean={np.mean(valid):+.3f}, "
                      f"range=[{np.min(valid):+.3f}, {np.max(valid):+.3f}]")
        print("-"*70)
        print("Partial correlations (controlling for other input type):")
        for key in ['r_excitatory_partial', 'r_inhibitory_partial']:
            data = results[key]
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                print(f"  {key:25s}: mean={np.mean(valid):+.3f}, "
                      f"range=[{np.min(valid):+.3f}, {np.max(valid):+.3f}]")
        print("="*70)
        print("\nDone!")
    else:
        print("\nNo completed runs found - skipping heatmap generation")


if __name__ == '__main__':
    main()
