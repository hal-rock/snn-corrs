"""
Analyze relationship between spike correlations and shared input magnitude.

Examines how pairwise spike correlations relate to:
- Feedforward shared input
- Recurrent excitatory shared input
- Recurrent inhibitory shared input
- Total recurrent shared input
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from analysis import load_spikes_vectorized, load_connectivity, load_params, get_n_stimuli


def compute_spike_correlations(directory: str, params: dict, neuron_indices: np.ndarray,
                                bin_size: float = 1000, total_time: float = 1000,
                                start_from: float = 50, end_time: float = None) -> np.ndarray:
    """
    Compute mean pairwise spike correlations across stimuli for selected neurons.

    Args:
        directory: Path to parameter point directory
        params: Dict with n_excite, n_inhib, n_stimuli, n_trials
        neuron_indices: Array of neuron indices to analyze
        bin_size: Time bin size in ms for spike counting
        total_time: Total simulation time in ms
        start_from: Discard spikes before this time in ms
        end_time: Discard spikes at or after this time in ms (default: None, uses total_time)

    Returns correlations for all pairs of the selected neurons.
    """
    n_neurons_total = params['n_excite'] + params['n_inhib']
    n_stimuli = get_n_stimuli(params, directory)
    n_trials = params['n_trials']
    n_selected = len(neuron_indices)

    # Compute number of bins per trial (same logic as load_spikes_vectorized)
    effective_end = end_time if end_time is not None else total_time
    effective_time = effective_end - start_from
    n_bins = int(np.floor(effective_time / bin_size))

    # Handle edge case: bin_size >= effective_time
    if n_bins < 1:
        n_bins = 1
        bin_size = effective_time

    n_samples = n_trials * n_bins  # Total samples per stimulus

    # Collect spike counts: (n_selected, n_stimuli, n_samples)
    all_responses = np.zeros((n_selected, n_stimuli, n_samples))

    for stim in range(n_stimuli):
        stim_dir = f"{directory}/stim{stim}"
        spikes = load_spikes_vectorized(
            stim_dir, n_trials, n_neurons_total,
            bin_size=bin_size, total_time=total_time, start_from=start_from,
            end_time=end_time
        )
        # Select our neurons - spikes shape is (n_neurons, n_trials * n_bins)
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
        recurrent_total: Total shared recurrent input
    """
    n_selected = len(neuron_indices)
    n_pairs = n_selected * (n_selected - 1) // 2

    # Extract connectivity to selected neurons
    # input_conn: (n_inputs, n_neurons) -> weights from input to network
    # recurrent_conn: (n_neurons, n_neurons) -> weights from network to network
    # Convention: conn[j, i] = weight from i to j -- opposite what you might expect

    input_to_selected = input_conn[:, neuron_indices]  # (n_inputs, n_selected)
    recurrent_to_selected = recurrent_conn[neuron_indices]  # (n_neurons, n_selected)

    # Separate excitatory and inhibitory recurrent
    exc_to_selected = recurrent_to_selected[:, :n_excite]  # (n_excite, n_selected)
    inh_to_selected = recurrent_to_selected[:, n_excite:]  # (n_inhib, n_selected)

    # Compute shared input for each pair
    # For pair (i, j): shared_input = sum_k(w_ki * w_kj)
    # This can be computed as: (W @ W.T)[i, j] for the off-diagonal elements

    # Feedforward shared input
    ff_shared_matrix = input_to_selected.T @ input_to_selected  # (n_selected, n_selected)

    # Excitatory recurrent shared input
    print(exc_to_selected.shape)
    exc_shared_matrix = exc_to_selected @ exc_to_selected.T

    # Inhibitory recurrent shared input
    print(inh_to_selected.shape)
    inh_shared_matrix = inh_to_selected @ inh_to_selected.T

    # Total recurrent
    #total_rec_shared_matrix = recurrent_to_selected @ recurrent_to_selected.T
    total_rec_shared_matrix = exc_shared_matrix - inh_shared_matrix

    # Extract lower triangular (pairs)
    tril = np.tril_indices(n_selected, k=-1)

    feedforward = ff_shared_matrix[tril]
    recurrent_exc = exc_shared_matrix[tril]
    recurrent_inh = inh_shared_matrix[tril]
    recurrent_total = total_rec_shared_matrix[tril]

    return feedforward, recurrent_exc, recurrent_inh, recurrent_total


def plot_relationship(spike_corrs, shared_input, xlabel, title, ax, subsample=1000):
    """Plot scatterplot of spike correlation vs shared input with regression line."""
    # Subsample for plotting if too many points
    n_points = len(spike_corrs)
    if n_points > subsample:
        idx = np.random.choice(n_points, subsample, replace=False)
        x = shared_input[idx]
        y = spike_corrs[idx]
    else:
        x = shared_input
        y = spike_corrs

    # Compute correlation on full data
    r, p = stats.pearsonr(shared_input, spike_corrs)

    ax.scatter(x, y, alpha=0.3, s=10, c='steelblue')

    # Add regression line
    slope, intercept = np.polyfit(shared_input, spike_corrs, 1)
    x_line = np.array([shared_input.min(), shared_input.max()])
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Spike Correlation')
    ax.set_title(f'{title}\nr = {r:.3f}, p = {p:.2e}')

    return r, p


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze relationship between spike correlations and shared input magnitude'
    )
    parser.add_argument('data_dir', type=str, nargs='?',
                        default='outputs/full_sweep/kappa_0/input_time_200',
                        help='Path to parameter point directory')
    parser.add_argument('--bin_size', type=float, default=1000,
                        help='Time bin size in ms for spike counting (default: 1000, full trial)')
    parser.add_argument('--total_time', type=float, default=1000,
                        help='Total simulation time in ms (default: 1000)')
    parser.add_argument('--start_from', type=float, default=50,
                        help='Discard spikes before this time in ms (default: 50)')
    parser.add_argument('--end_time', type=float, default=None,
                        help='Discard spikes at or after this time in ms (default: None, uses total_time)')
    parser.add_argument('--n_pairs', type=int, default=20000,
                        help='Approximate number of neuron pairs to analyze (default: 20000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--fig_dir', type=str, default='figures',
                        help='Directory to save figures (default: figures)')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display plot interactively')

    args = parser.parse_args()

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(exist_ok=True)

    np.random.seed(args.seed)

    # Compute number of bins for reporting
    effective_end = args.end_time if args.end_time is not None else args.total_time
    n_bins = int(np.floor((effective_end - args.start_from) / args.bin_size))

    print(f"Analyzing: {args.data_dir}")
    end_time_str = f"{args.end_time}ms" if args.end_time else f"{args.total_time}ms (total_time)"
    print(f"Settings: bin_size={args.bin_size}ms, start_from={args.start_from}ms, end_time={end_time_str}")
    print(f"  -> {n_bins} bin(s) per trial")
    print()

    print("Loading parameters and connectivity...")
    params = load_params(args.data_dir)
    # just analyze the first stimulus since we are interested in single-stim corrs
    params['n_stimuli'] = 1
    input_conn, recurrent_conn = load_connectivity(args.data_dir)
    # need to adjust for diff synapse timescales of inhibition
    e_to_i_ratio = 2/(3.3 * 2.26) # roughly
    recurrent_conn[:, params['n_excite']:] *= -e_to_i_ratio

    n_excite = params['n_excite']
    n_inhib = params['n_inhib']
    n_neurons = n_excite + n_inhib

    # Sample neurons to get approximately n_pairs_to_analyze pairs
    # n_pairs = n*(n-1)/2, so n ≈ sqrt(2*n_pairs)
    n_neurons_sample = int(np.ceil(np.sqrt(2 * args.n_pairs))) + 1

    actual_n_pairs = n_neurons_sample * (n_neurons_sample - 1) // 2
    print(f"Sampling {n_neurons_sample} neurons -> {actual_n_pairs} pairs")

    neuron_indices = np.random.choice(n_neurons, n_neurons_sample, replace=False)

    print("Computing spike correlations...")
    spike_corrs, tril_indices = compute_spike_correlations(
        args.data_dir, params, neuron_indices,
        bin_size=args.bin_size, total_time=args.total_time, start_from=args.start_from,
        end_time=args.end_time
    )

    print("Computing shared input magnitudes...")
    ff_shared, exc_shared, inh_shared, total_rec_shared = compute_shared_inputs(
        input_conn, recurrent_conn, neuron_indices, n_excite
    )

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Title includes bin size for clarity
    bin_label = f"{int(args.bin_size)}ms bins" if args.bin_size < args.total_time else "full trial"
    fig.suptitle(f'Spike Correlation ({bin_label}) vs Shared Input Magnitude', fontsize=14)

    results = {}

    # Feedforward
    r, p = plot_relationship(
        spike_corrs, ff_shared,
        'Feedforward Shared Input (Σ w_ki·w_kj)',
        'Feedforward Input',
        axes[0, 0]
    )
    results['feedforward'] = (r, p)

    # Excitatory recurrent
    r, p = plot_relationship(
        spike_corrs, exc_shared,
        'Excitatory Shared Input (Σ w_ki·w_kj)',
        'Recurrent Excitatory',
        axes[0, 1]
    )
    results['recurrent_excitatory'] = (r, p)

    # Inhibitory recurrent
    r, p = plot_relationship(
        spike_corrs, inh_shared,
        'Inhibitory Shared Input (Σ w_ki·w_kj)',
        'Recurrent Inhibitory',
        axes[1, 0]
    )
    results['recurrent_inhibitory'] = (r, p)

    # Total recurrent
    r, p = plot_relationship(
        spike_corrs, total_rec_shared,
        'Total Recurrent Shared Input (Σ w_ki·w_kj)',
        'Total Recurrent',
        axes[1, 1]
    )
    results['recurrent_total'] = (r, p)

    plt.tight_layout()

    # Include bin size in filename for different analyses
    bin_suffix = f"_{int(args.bin_size)}ms" if args.bin_size < 1000 else ""
    fig_path = fig_dir / f'spike_corr_vs_shared_input{bin_suffix}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")

    # Print summary
    print("\n" + "="*60)
    print(f"CORRELATION RESULTS: Spike Correlation ({bin_label}) vs Shared Input")
    print("="*60)
    print(f"Number of pairs analyzed: {len(spike_corrs)}")
    print(f"Mean spike correlation: {np.mean(spike_corrs):.4f} (std: {np.std(spike_corrs):.4f})")
    print("\nPearson correlations (r) between spike correlation and shared input:")
    print("-"*60)
    for name, (r, p) in results.items():
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:25s}: r = {r:+.4f}  (p = {p:.2e}) {sig}")
    print("-"*60)
    print("Significance: * p<0.05, ** p<0.01, *** p<0.001")

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
