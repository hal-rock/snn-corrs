"""
Clean analysis module for SNN simulation outputs.

Provides functions to:
- Load spike data efficiently (vectorized)
- Compute correlations and their statistics
- Decompose variance into signal vs noise components
- Analyze a single parameter point or sweep results
"""
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

from corr_analyses import (
    calculate_icc,
    calculate_explainable_variance_proportion,
    bootstrap_all_pairs_variance
)


# =============================================================================
# Data Loading
# =============================================================================

def load_spikes_vectorized(directory: str, n_trials: int, n_neurons: int,
                           bin_size: float = 1000, total_time: float = 1000,
                           start_from: float = 50, end_time: float = None) -> np.ndarray:
    """
    Load spike data from trial directories into a binned matrix.

    Vectorized implementation - O(n_spikes) instead of O(n_neurons * n_spikes).

    Args:
        directory: Path containing trial0/, trial1/, etc. subdirectories
        n_trials: Number of trials to load
        n_neurons: Total number of neurons
        bin_size: Time bin size in ms
        total_time: Total simulation time in ms (used if end_time not specified)
        start_from: Discard spikes before this time (ms)
        end_time: Discard spikes at or after this time (ms). If None, uses total_time.

    Returns:
        Array of shape (n_neurons, n_trials * n_bins) with spike counts
    """
    # Use end_time if specified, otherwise fall back to total_time
    effective_end = end_time if end_time is not None else total_time
    effective_time = effective_end - start_from
    n_bins = int(np.floor(effective_time / bin_size))

    # Handle edge case: bin_size >= effective_time
    if n_bins < 1:
        n_bins = 1
        bin_size = effective_time

    result = np.zeros((n_neurons, n_trials * n_bins), dtype=np.uint8)

    for trial in range(n_trials):
        spike_file = f'{directory}/trial{trial}/spike_monitor0.npz'
        data = np.load(spike_file)
        idxs = data['idxs']
        times = data['times']
        data.close()

        # Filter to times within [start_from, end_time)
        mask = (times > start_from) & (times < effective_end)
        idxs = idxs[mask]
        times = times[mask] - start_from

        # Compute bin indices directly
        bin_indices = np.floor(times / bin_size).astype(np.int32)

        # Filter out-of-range bins
        valid = (bin_indices >= 0) & (bin_indices < n_bins)
        idxs = idxs[valid]
        bin_indices = bin_indices[valid]

        # Offset for this trial's columns
        col_indices = bin_indices + trial * n_bins

        # Count spikes with unbuffered addition
        np.add.at(result, (idxs, col_indices), 1)

    return result


def load_input_spikes(directory: str, n_trials: int, n_inputs: int) -> np.ndarray:
    """
    Load input spike counts per input neuron per trial.

    Args:
        directory: Stimulus directory containing input_times.npz and input_spikes.npz
        n_trials: Number of trials
        n_inputs: Number of input neurons

    Returns:
        Array of shape (n_inputs, n_trials) with spike counts
    """
    inpt_spikes = np.load(f'{directory}/input_spikes.npz')
    inpt_times = np.load(f'{directory}/input_times.npz')

    # Count spikes per input neuron per trial
    counts = np.zeros((n_inputs, n_trials), dtype=np.int32)
    for trial in range(min(n_trials, len(inpt_spikes.files))):
        spikes = inpt_spikes[f'arr_{trial}']
        times = inpt_times[f'arr_{trial}']
        for inp_idx in range(n_inputs):
            counts[inp_idx, trial] = np.sum(times == inp_idx)

    inpt_spikes.close()
    inpt_times.close()
    return counts


def load_connectivity(directory: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load input and recurrent connectivity matrices.

    Args:
        directory: Directory containing conn_mats.npz

    Returns:
        Tuple of (input_conn, recurrent_conn)
    """
    conns = np.load(f'{directory}/conn_mats.npz')
    input_conn = conns['inpt']
    recurrent_conn = conns['recurrent']
    conns.close()
    return input_conn, recurrent_conn


def load_params(directory: str) -> dict:
    """Load parameters from params.json or config.json."""
    params_file = Path(directory) / 'params.json'
    if params_file.exists():
        with open(params_file) as f:
            return json.load(f)

    config_file = Path(directory) / 'config.json'
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)

    raise FileNotFoundError(f"No params.json or config.json found in {directory}")


# =============================================================================
# Correlation Analysis
# =============================================================================

@dataclass
class CorrelationStats:
    """Statistics for a set of pairwise correlations."""
    correlations: np.ndarray      # Raw correlations (n_stimuli, n_pairs)
    z_correlations: np.ndarray    # Fisher z-transformed
    noise_variances: np.ndarray   # Bootstrap noise variance per pair (n_stimuli, n_pairs)
    mean_noise_var: float         # Average noise variance
    total_var: float              # Total variance of z-correlations
    icc: float                    # Intraclass correlation coefficient
    explainable_icc: float        # ICC adjusted for measurement noise
    signal_var: float             # Between-stimulus variance
    residual_var: float           # Unexplained within-stimulus variance


def compute_pairwise_correlations(responses: np.ndarray,
                                  neuron_indices: Optional[np.ndarray] = None
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise correlations for selected neurons.

    Args:
        responses: Array of shape (n_neurons, n_stimuli, n_trials)
        neuron_indices: Subset of neurons to use (default: all)

    Returns:
        Tuple of (correlations, tril_indices) where correlations has shape
        (n_stimuli, n_pairs)
    """
    if neuron_indices is not None:
        responses = responses[neuron_indices]

    n_neurons, n_stimuli, n_trials = responses.shape
    tril_indices = np.tril_indices(n_neurons, k=-1)
    n_pairs = len(tril_indices[0])

    correlations = np.zeros((n_stimuli, n_pairs))

    for stim in range(n_stimuli):
        corr_mat = np.corrcoef(responses[:, stim, :])
        corr_mat[np.isnan(corr_mat)] = 0
        correlations[stim] = corr_mat[tril_indices]

    return correlations, tril_indices


def analyze_correlations(responses: np.ndarray,
                        neuron_indices: Optional[np.ndarray] = None,
                        n_bootstrap: int = 1000,
                        verbose: bool = True) -> CorrelationStats:
    """
    Full correlation analysis with variance decomposition.

    Args:
        responses: Array of shape (n_neurons, n_stimuli, n_trials)
        neuron_indices: Subset of neurons to analyze
        n_bootstrap: Number of bootstrap samples for noise estimation
        verbose: Print progress

    Returns:
        CorrelationStats dataclass with all statistics
    """
    if neuron_indices is not None:
        responses = responses[neuron_indices]

    n_neurons, n_stimuli, n_trials = responses.shape

    # Compute correlations
    correlations, _ = compute_pairwise_correlations(responses)
    z_corrs = np.arctanh(np.clip(correlations, -0.999, 0.999))

    # Bootstrap noise variance estimation
    if verbose:
        print(f"Bootstrapping noise variance ({n_bootstrap} samples)...")
    noise_vars = bootstrap_all_pairs_variance(responses, num_bootstrap_samples=n_bootstrap)
    mean_noise_var = np.mean(noise_vars)

    # Total variance
    total_var = np.var(z_corrs)

    # ICC
    icc = calculate_icc(z_corrs)

    # Explainable ICC
    explainable_icc, signal_var, residual_var = calculate_explainable_variance_proportion(
        z_corrs, mean_noise_var
    )

    return CorrelationStats(
        correlations=correlations,
        z_correlations=z_corrs,
        noise_variances=noise_vars,
        mean_noise_var=mean_noise_var,
        total_var=total_var,
        icc=icc,
        explainable_icc=explainable_icc,
        signal_var=signal_var,
        residual_var=residual_var
    )


def compute_snr(z_correlations: np.ndarray, noise_variances: np.ndarray) -> np.ndarray:
    """
    Compute signal-to-noise ratio for each pair.

    Args:
        z_correlations: Fisher z-transformed correlations (n_stimuli, n_pairs)
        noise_variances: Noise variance per pair (n_stimuli, n_pairs)

    Returns:
        SNR array of shape (n_stimuli, n_pairs)
    """
    return np.power(z_correlations, 2) / noise_variances


# =============================================================================
# Input Analysis
# =============================================================================

def compute_input_correlations(input_counts: np.ndarray,
                               input_conn: np.ndarray,
                               neuron_indices: np.ndarray,
                               n_stimuli: int,
                               n_trials: int) -> np.ndarray:
    """
    Compute correlations of total feedforward input to selected neurons.

    Args:
        input_counts: Input spike counts (n_inputs, n_stimuli * n_trials)
        input_conn: Input connectivity matrix (n_inputs, n_neurons)
        neuron_indices: Neurons to analyze
        n_stimuli: Number of stimuli
        n_trials: Number of trials per stimulus

    Returns:
        Correlations array of shape (n_stimuli, n_pairs)
    """
    n_neurons = len(neuron_indices)
    n_pairs = n_neurons * (n_neurons - 1) // 2
    tril = np.tril_indices(n_neurons, k=-1)

    correlations = np.zeros((n_stimuli, n_pairs))

    # Get connectivity to selected neurons
    conn_subset = input_conn[:, neuron_indices]

    for stim in range(n_stimuli):
        # Get input counts for this stimulus
        trial_slice = slice(stim * n_trials, (stim + 1) * n_trials)
        counts = input_counts[:, trial_slice]  # (n_inputs, n_trials)

        # Compute total input to each neuron: (n_neurons, n_trials)
        total_input = conn_subset.T @ counts

        # Correlations
        corr_mat = np.corrcoef(total_input)
        corr_mat[np.isnan(corr_mat)] = 0
        correlations[stim] = corr_mat[tril]

    return correlations


def compute_recurrent_input_correlations(spike_data: np.ndarray,
                                         recurrent_conn: np.ndarray,
                                         neuron_indices: np.ndarray,
                                         n_excite: int,
                                         n_stimuli: int,
                                         n_trials: int,
                                         e_to_i_ratio: float = 1/(3.3 * 2.26)
                                         ) -> np.ndarray:
    """
    Compute correlations of recurrent input to selected neurons.

    Args:
        spike_data: Binned spike counts (n_neurons, n_stimuli * n_trials * n_bins)
        recurrent_conn: Recurrent connectivity matrix
        neuron_indices: Neurons to analyze
        n_excite: Number of excitatory neurons (for sign correction)
        n_stimuli: Number of stimuli
        n_trials: Number of trials
        e_to_i_ratio: Ratio to apply to inhibitory weights

    Returns:
        Correlations array of shape (n_stimuli, n_pairs)
    """
    n_neurons = len(neuron_indices)
    n_pairs = n_neurons * (n_neurons - 1) // 2
    tril = np.tril_indices(n_neurons, k=-1)

    # Make inhibitory weights negative
    conn = recurrent_conn.copy()
    conn[:n_excite, n_excite:] *= -e_to_i_ratio
    conn[n_excite:, n_excite:] *= -e_to_i_ratio

    # Compute recurrent input: (n_neurons, time)
    recurrent_input = conn @ spike_data

    # Select neurons of interest
    recurrent_input = recurrent_input[neuron_indices]

    # Reshape to (n_neurons, n_stimuli, n_trials_x_bins) then sum over bins
    # For now, assume single bin per trial (bin_size = total_time)
    n_total = spike_data.shape[1]
    n_per_stim = n_total // n_stimuli

    correlations = np.zeros((n_stimuli, n_pairs))
    for stim in range(n_stimuli):
        stim_slice = slice(stim * n_per_stim, (stim + 1) * n_per_stim)
        data = recurrent_input[:, stim_slice]
        corr_mat = np.corrcoef(data)
        corr_mat[np.isnan(corr_mat)] = 0
        correlations[stim] = corr_mat[tril]

    return correlations


# =============================================================================
# Single Point Analysis
# =============================================================================

@dataclass
class AnalysisResults:
    """Complete analysis results for a single parameter point."""
    params: dict

    # Response correlations
    response_stats: CorrelationStats

    # Input correlations (optional)
    input_correlations: Optional[np.ndarray] = None
    input_stats: Optional[CorrelationStats] = None

    # Recurrent input correlations (optional)
    recurrent_correlations: Optional[np.ndarray] = None
    recurrent_stats: Optional[CorrelationStats] = None

    # Metadata
    n_neurons_analyzed: int = 0
    neuron_indices: Optional[np.ndarray] = None


def analyze_parameter_point(directory: str,
                           n_neurons_sample: int = 250,
                           n_bootstrap: int = 1000,
                           analyze_inputs: bool = True,
                           analyze_recurrent: bool = True,
                           seed: Optional[int] = None,
                           verbose: bool = True,
                           bin_size: float = 1000,
                           total_time: float = 1000,
                           start_from: float = 50,
                           end_time: float = None) -> AnalysisResults:
    """
    Complete analysis of a single parameter point.

    Args:
        directory: Path to parameter point directory (containing stim0/, stim1/, etc.)
        n_neurons_sample: Number of neurons to randomly sample for analysis
        n_bootstrap: Bootstrap samples for noise estimation
        analyze_inputs: Whether to analyze feedforward input correlations
        analyze_recurrent: Whether to analyze recurrent input correlations
        seed: Random seed for neuron sampling
        verbose: Print progress
        bin_size: Time bin size in ms for spike counting (default: 1000, full trial)
        total_time: Total simulation time in ms (default: 1000)
        start_from: Discard spikes before this time in ms (default: 50)
        end_time: Discard spikes at or after this time in ms (default: None, uses total_time)

    Returns:
        AnalysisResults dataclass with all computed statistics
    """
    directory = Path(directory)

    # Load parameters
    params = load_params(directory)
    n_excite = params.get('n_excite', 4000)
    n_inhib = params.get('n_inhib', 1000)
    n_neurons = n_excite + n_inhib
    n_inputs = params.get('n_input', 3000)
    n_trials = params.get('n_trials', 100)
    n_stimuli = params.get('n_stimuli', 3)

    # Calculate number of bins, handling edge cases
    effective_end = end_time if end_time is not None else total_time
    effective_time = effective_end - start_from
    n_bins = int(np.floor(effective_time / bin_size))
    if n_bins < 1:
        # bin_size >= effective_time: fall back to single bin covering full time
        n_bins = 1
        bin_size = effective_time
        if verbose:
            print(f"  Note: bin_size >= effective time, using single {effective_time}ms bin")

    n_samples = n_trials * n_bins  # total samples per stimulus for correlation

    if verbose:
        print(f"Analyzing {directory}")
        print(f"  {n_stimuli} stimuli, {n_trials} trials, {n_neurons} neurons")
        if n_bins > 1:
            print(f"  Using {n_bins} bins of {bin_size}ms each ({n_samples} samples per stimulus)")

    # Sample neurons
    if seed is not None:
        np.random.seed(seed)
    neuron_indices = np.random.choice(n_neurons, n_neurons_sample, replace=False)

    # Load spike data for all stimuli
    if verbose:
        print("  Loading spike data...")

    # Shape: (n_neurons, n_stimuli, n_samples) where n_samples = n_trials * n_bins
    all_responses = np.zeros((n_neurons, n_stimuli, n_samples))
    all_spikes = []  # Keep full spike data for recurrent analysis

    for stim in range(n_stimuli):
        stim_dir = directory / f'stim{stim}'
        spikes = load_spikes_vectorized(
            str(stim_dir), n_trials, n_neurons,
            bin_size=bin_size, total_time=total_time, start_from=start_from,
            end_time=end_time
        )
        all_spikes.append(spikes)
        # spikes shape is (n_neurons, n_trials * n_bins)
        # Keep flat - each time bin treated as independent sample for correlation
        all_responses[:, stim, :] = spikes

    # Analyze response correlations
    if verbose:
        print("  Analyzing response correlations...")
    response_stats = analyze_correlations(
        all_responses, neuron_indices, n_bootstrap, verbose
    )

    results = AnalysisResults(
        params=params,
        response_stats=response_stats,
        n_neurons_analyzed=n_neurons_sample,
        neuron_indices=neuron_indices
    )

    # Analyze feedforward inputs
    if analyze_inputs:
        if verbose:
            print("  Analyzing feedforward input correlations...")

        input_conn, recurrent_conn = load_connectivity(directory)

        # Load input spikes for all stimuli
        all_input_counts = []
        for stim in range(n_stimuli):
            stim_dir = directory / f'stim{stim}'
            counts = load_input_spikes(str(stim_dir), n_trials, n_inputs)
            all_input_counts.append(counts)

        # Reshape to (n_inputs, n_stimuli * n_trials)
        input_counts = np.concatenate(all_input_counts, axis=1)

        # Compute input to each neuron per trial
        input_to_neurons = np.zeros((n_neurons_sample, n_stimuli, n_trials))
        conn_subset = input_conn[:, neuron_indices]
        for stim in range(n_stimuli):
            counts = all_input_counts[stim]
            input_to_neurons[:, stim, :] = conn_subset.T @ counts

        input_corrs, _ = compute_pairwise_correlations(input_to_neurons)
        results.input_correlations = input_corrs

        # Full stats for inputs
        results.input_stats = analyze_correlations(
            input_to_neurons, None, n_bootstrap, verbose
        )

    # Analyze recurrent inputs
    if analyze_recurrent:
        if verbose:
            print("  Analyzing recurrent input correlations...")

        if not analyze_inputs:
            input_conn, recurrent_conn = load_connectivity(directory)

        # Stack all spike data
        all_spike_data = np.concatenate(all_spikes, axis=1)

        recurrent_corrs = compute_recurrent_input_correlations(
            all_spike_data, recurrent_conn, neuron_indices,
            n_excite, n_stimuli, n_trials
        )
        results.recurrent_correlations = recurrent_corrs

        # For recurrent stats, reshape the recurrent input
        conn = recurrent_conn.copy()
        e_to_i_ratio = 1 / (3.3 * 2.26)
        conn[:n_excite, n_excite:] *= -e_to_i_ratio
        conn[n_excite:, n_excite:] *= -e_to_i_ratio

        recurrent_input = conn @ all_spike_data
        recurrent_input = recurrent_input[neuron_indices]

        # Reshape to (n_neurons, n_stimuli, n_trials)
        n_per_stim = all_spike_data.shape[1] // n_stimuli
        recurrent_responses = np.zeros((n_neurons_sample, n_stimuli, n_trials))
        for stim in range(n_stimuli):
            stim_slice = slice(stim * n_per_stim, (stim + 1) * n_per_stim)
            # Sum over time within each trial
            data = recurrent_input[:, stim_slice].reshape(n_neurons_sample, n_trials, -1)
            recurrent_responses[:, stim, :] = data.sum(axis=2)

        results.recurrent_stats = analyze_correlations(
            recurrent_responses, None, n_bootstrap, verbose
        )

    return results


def print_results_summary(results: AnalysisResults):
    """Print a summary of analysis results."""
    print("\n" + "="*60)
    print("ANALYSIS RESULTS SUMMARY")
    print("="*60)

    # Parameters
    print("\nParameters:")
    for key in ['kappa', 'input_time', 'n_trials', 'n_stimuli']:
        if key in results.params:
            print(f"  {key}: {results.params[key]}")

    # Response correlations
    print(f"\nResponse Correlations ({results.n_neurons_analyzed} neurons):")
    stats = results.response_stats
    print(f"  Mean correlation: {np.mean(stats.correlations):.4f}")
    print(f"  Total variance (z): {stats.total_var:.4f}")
    print(f"  Noise variance: {stats.mean_noise_var:.4f}")
    print(f"  Noise fraction: {stats.mean_noise_var / stats.total_var:.2%}")
    print(f"  ICC: {stats.icc:.4f}")
    print(f"  Explainable ICC: {stats.explainable_icc:.4f}")

    # Input correlations
    if results.input_stats is not None:
        print(f"\nFeedforward Input Correlations:")
        stats = results.input_stats
        print(f"  Mean correlation: {np.mean(stats.correlations):.4f}")
        print(f"  Noise fraction: {stats.mean_noise_var / stats.total_var:.2%}")
        print(f"  Explainable ICC: {stats.explainable_icc:.4f}")

    # Recurrent correlations
    if results.recurrent_stats is not None:
        print(f"\nRecurrent Input Correlations:")
        stats = results.recurrent_stats
        print(f"  Mean correlation: {np.mean(stats.correlations):.4f}")
        print(f"  Noise fraction: {stats.mean_noise_var / stats.total_var:.2%}")
        print(f"  Explainable ICC: {stats.explainable_icc:.4f}")

    print("="*60)


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze SNN simulation outputs')
    parser.add_argument('directory', type=str,
                        help='Path to parameter point directory')
    parser.add_argument('--n_neurons', type=int, default=250,
                        help='Number of neurons to sample (default: 250)')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Bootstrap samples (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no_inputs', action='store_true',
                        help='Skip feedforward input analysis')
    parser.add_argument('--no_recurrent', action='store_true',
                        help='Skip recurrent input analysis')

    args = parser.parse_args()

    results = analyze_parameter_point(
        args.directory,
        n_neurons_sample=args.n_neurons,
        n_bootstrap=args.n_bootstrap,
        analyze_inputs=not args.no_inputs,
        analyze_recurrent=not args.no_recurrent,
        seed=args.seed,
        verbose=True
    )

    print_results_summary(results)
