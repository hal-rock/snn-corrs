"""
Examine shared input vs noise correlation relationships for a single parameter point.

Selects a specific parameter value from a sweep and plots scatterplots showing
the relationships between shared inputs (FF, E, I) and noise correlations
across neuron pairs.

Usage:
    python examine_inputs.py sweep_dir --param1_val VALUE1 --param2_val VALUE2
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from analysis import load_connectivity, load_params
from input_heatmaps import (
    load_input_rates,
    compute_network_rates_per_stim,
    compute_spike_correlations_single_stim,
    compute_rate_weighted_shared_inputs,
    discover_sweep_structure,
    format_param_value,
)


def compute_regression_predictions(y: np.ndarray, predictors: list) -> np.ndarray:
    """
    Compute predicted values from multiple linear regression.

    Args:
        y: Dependent variable (e.g., noise correlations)
        predictors: List of predictor arrays

    Returns:
        Predicted y values
    """
    X = np.column_stack([np.ones(len(y))] + predictors)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    return y_pred


def analyze_single_point(directory: str,
                         n_neurons_sample: int = 200,
                         seed: int = 42,
                         bin_size: float = 1000,
                         total_time: float = None,
                         start_from: float = 50,
                         single_stimulus: bool = False) -> dict:
    """
    Compute shared inputs and spike correlations for a single parameter point.

    Args:
        directory: Path to parameter point directory
        n_neurons_sample: Number of neurons to sample
        seed: Random seed for reproducibility
        bin_size: Time bin size in ms for spike counting
        total_time: Total simulation time in ms (None = read from params)
        start_from: Discard spikes before this time in ms
        single_stimulus: If True, use only stim0. If False (default), combine
                         across all stimuli.

    Returns dict with arrays for plotting scatterplots.
    """
    np.random.seed(seed)

    params = load_params(directory)
    n_excite = params['n_excite']
    n_inhib = params['n_inhib']
    n_neurons = n_excite + n_inhib
    n_stimuli = params.get('n_stimuli', 1)

    if total_time is None:
        total_time = params.get('simulation_time', 1.05) * 1000

    effective_time = total_time - start_from
    if bin_size > effective_time:
        bin_size = effective_time

    # Load connectivity (same for all stimuli)
    input_conn, recurrent_conn = load_connectivity(directory)

    # Sample neurons (same across stimuli)
    neuron_indices = np.random.choice(n_neurons, n_neurons_sample, replace=False)

    if single_stimulus:
        stim_indices = [0]
    else:
        stim_indices = list(range(n_stimuli))

    # Accumulate data across stimuli
    all_spike_corrs = []
    all_ff_shared = []
    all_exc_shared = []
    all_inh_shared = []
    all_stim_ids = []
    all_geo_mean_rates = []

    # Get tril indices for pairs (same for all stimuli)
    n_selected = len(neuron_indices)
    tril_i, tril_j = np.tril_indices(n_selected, k=-1)

    for stim in stim_indices:
        stim_dir = f"{directory}/stim{stim}"

        # Load input rates for this stimulus
        input_rates = load_input_rates(stim_dir)

        # Compute network firing rates for this stimulus
        network_rates = compute_network_rates_per_stim(
            stim_dir, params,
            total_time=total_time, start_from=start_from
        )

        # Compute spike correlations for this stimulus
        spike_corrs = compute_spike_correlations_single_stim(
            stim_dir, params, neuron_indices,
            bin_size=bin_size, total_time=total_time, start_from=start_from
        )

        # Compute rate-weighted shared inputs for this stimulus
        ff_shared, exc_shared, inh_shared = compute_rate_weighted_shared_inputs(
            input_conn, recurrent_conn, neuron_indices, n_excite,
            input_rates, network_rates
        )

        # Compute geometric mean firing rate for each pair
        selected_rates = network_rates[neuron_indices]
        geo_mean_rates = np.sqrt(selected_rates[tril_i] * selected_rates[tril_j])

        n_pairs = len(spike_corrs)
        all_spike_corrs.append(spike_corrs)
        all_ff_shared.append(ff_shared)
        all_exc_shared.append(exc_shared)
        all_inh_shared.append(inh_shared)
        all_stim_ids.append(np.full(n_pairs, stim))
        all_geo_mean_rates.append(geo_mean_rates)

    # Concatenate across stimuli
    spike_corrs = np.concatenate(all_spike_corrs)
    ff_shared = np.concatenate(all_ff_shared)
    exc_shared = np.concatenate(all_exc_shared)
    inh_shared = np.concatenate(all_inh_shared)
    stim_ids = np.concatenate(all_stim_ids)
    geo_mean_rates = np.concatenate(all_geo_mean_rates)

    # Compute regression predictions
    predicted_corrs = compute_regression_predictions(
        spike_corrs, [ff_shared, exc_shared, inh_shared]
    )

    # Compute R² for full model
    ss_res = np.sum((spike_corrs - predicted_corrs) ** 2)
    ss_tot = np.sum((spike_corrs - np.mean(spike_corrs)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {
        'spike_corrs': spike_corrs,
        'ff_shared': ff_shared,
        'exc_shared': exc_shared,
        'inh_shared': inh_shared,
        'predicted_corrs': predicted_corrs,
        'stim_ids': stim_ids,
        'geo_mean_rates': geo_mean_rates,
        'r2': r2,
        'n_pairs': len(spike_corrs),
        'n_stimuli_used': len(stim_indices),
    }


def plot_scatterplots(results: dict, title_prefix: str = ""):
    """
    Plot 2x3 scatterplots of shared input vs noise correlation relationships.
    """
    spike_corrs = results['spike_corrs']
    ff_shared = results['ff_shared']
    exc_shared = results['exc_shared']
    inh_shared = results['inh_shared']
    predicted_corrs = results['predicted_corrs']
    geo_mean_rates = results['geo_mean_rates']
    r2 = results['r2']
    n_pairs = results['n_pairs']

    # Compute residuals
    residuals = spike_corrs - predicted_corrs

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    plot_data = [
        (axes[0, 0], ff_shared, 'FF Shared Input', 'tab:blue'),
        (axes[0, 1], exc_shared, 'Exc Shared Input', 'tab:green'),
        (axes[0, 2], inh_shared, 'Inh Shared Input', 'tab:red'),
    ]

    for ax, x_data, x_label, color in plot_data:
        # Scatter plot
        ax.scatter(x_data, spike_corrs, alpha=0.3, s=10, c=color, edgecolors='none')

        # Regression line
        slope, intercept = np.polyfit(x_data, spike_corrs, 1)
        x_line = np.array([x_data.min(), x_data.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k-', linewidth=2)

        # Stats
        r, p = stats.pearsonr(x_data, spike_corrs)
        p_str = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
        ax.text(0.05, 0.95, f"r = {r:.3f}\n{p_str}\nn = {n_pairs}",
                transform=ax.transAxes, verticalalignment='top',
                fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel(x_label)
        ax.set_ylabel('Noise Correlation')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Fourth panel: actual vs predicted from multiple regression
    ax = axes[1, 0]
    ax.scatter(predicted_corrs, spike_corrs, alpha=0.3, s=10, c='tab:purple', edgecolors='none')

    # Identity line
    lims = [min(predicted_corrs.min(), spike_corrs.min()),
            max(predicted_corrs.max(), spike_corrs.max())]
    ax.plot(lims, lims, 'k--', linewidth=1.5, label='y=x')

    # Regression line (should be close to identity if model fits well)
    slope, intercept = np.polyfit(predicted_corrs, spike_corrs, 1)
    x_line = np.array([predicted_corrs.min(), predicted_corrs.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'k-', linewidth=2)

    ax.text(0.05, 0.95, f"R² = {r2:.3f}\nn = {n_pairs}",
            transform=ax.transAxes, verticalalignment='top',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Predicted (FF + Exc + Inh)')
    ax.set_ylabel('Actual Noise Correlation')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    # Fifth panel: residuals vs geometric mean firing rate
    ax = axes[1, 1]
    ax.scatter(geo_mean_rates, residuals, alpha=0.3, s=10, c='tab:orange', edgecolors='none')

    # Regression line
    slope, intercept = np.polyfit(geo_mean_rates, residuals, 1)
    x_line = np.array([geo_mean_rates.min(), geo_mean_rates.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'k-', linewidth=2)

    # Stats
    r, p = stats.pearsonr(geo_mean_rates, residuals)
    p_str = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
    ax.text(0.05, 0.95, f"r = {r:.3f}\n{p_str}",
            transform=ax.transAxes, verticalalignment='top',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Geometric Mean Firing Rate (Hz)')
    ax.set_ylabel('Residual (Actual - Predicted)')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # Sixth panel: leave empty or hide
    axes[1, 2].axis('off')

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14)

    plt.tight_layout()
    return fig


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Examine shared input vs noise correlation for a single parameter point'
    )
    parser.add_argument('sweep_dir', type=str,
                        help='Path to sweep directory')
    parser.add_argument('--param1_val', type=float, required=True,
                        help='Value of first parameter')
    parser.add_argument('--param2_val', type=float, required=True,
                        help='Value of second parameter')
    parser.add_argument('--n_neurons', type=int, default=200,
                        help='Number of neurons to sample (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--bin_size', type=float, default=1000,
                        help='Time bin size in ms (default: 1000)')
    parser.add_argument('--total_time', type=float, default=None,
                        help='Total simulation time in ms (default: from params)')
    parser.add_argument('--start_from', type=float, default=50,
                        help='Discard spikes before this time in ms (default: 50)')
    parser.add_argument('--single_stimulus', action='store_true',
                        help='Use only stim0 (default: combine across all stimuli)')

    args = parser.parse_args()

    # Discover sweep structure
    param1_name, param1_values, param2_name, param2_values = discover_sweep_structure(args.sweep_dir)

    print(f"Sweep structure: {param1_name} x {param2_name}")
    print(f"  {param1_name}: {param1_values}")
    print(f"  {param2_name}: {param2_values}")

    # Construct path to target directory
    sweep_path = Path(args.sweep_dir)
    target_dir = (sweep_path /
                  f"{param1_name}_{format_param_value(args.param1_val)}" /
                  f"{param2_name}_{format_param_value(args.param2_val)}")

    if not target_dir.exists():
        print(f"\nError: Directory not found: {target_dir}")
        print(f"Available {param1_name} values: {param1_values}")
        print(f"Available {param2_name} values: {param2_values}")
        return

    stim_mode = "single stimulus (stim0)" if args.single_stimulus else "all stimuli combined"
    print(f"\nAnalyzing: {target_dir}")
    print(f"Settings: n_neurons={args.n_neurons}, seed={args.seed}, bin_size={args.bin_size}ms")
    print(f"Stimulus mode: {stim_mode}")

    results = analyze_single_point(
        str(target_dir),
        n_neurons_sample=args.n_neurons,
        seed=args.seed,
        bin_size=args.bin_size,
        total_time=args.total_time,
        start_from=args.start_from,
        single_stimulus=args.single_stimulus
    )

    n_stim_str = f" ({results['n_stimuli_used']} stimuli)" if results['n_stimuli_used'] > 1 else ""
    print(f"\nAnalyzed {results['n_pairs']} neuron pair-stimulus combinations{n_stim_str}")
    print(f"Full model R² = {results['r2']:.3f}")

    # Print individual correlations
    for name, data in [('FF', results['ff_shared']),
                       ('Exc', results['exc_shared']),
                       ('Inh', results['inh_shared'])]:
        r, p = stats.pearsonr(data, results['spike_corrs'])
        print(f"  r(noise, {name}) = {r:+.3f} (p={p:.2e})")

    title = f"{param1_name}={args.param1_val}, {param2_name}={args.param2_val}"
    plot_scatterplots(results, title_prefix=title)
    plt.show()


if __name__ == '__main__':
    main()
