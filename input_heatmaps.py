"""
Generate heatmaps of rate-weighted shared input vs noise correlation relationships.

Unlike corr_heatmaps.py which uses weight overlap only (Σ w_ki w_kj),
this script weights shared input by presynaptic firing rates (Σ w_ki w_kj r_k).

This captures the actual covariance of inputs, since for Poisson inputs:
    Cov(I_i, I_j) = Σ_k w_ki w_kj Var(s_k) ∝ Σ_k w_ki w_kj r_k

For each parameter point, computes Pearson correlation between:
- Pairwise spike (noise) correlations
- Rate-weighted shared input magnitudes (feedforward, excitatory, inhibitory)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
from scipy import stats
import re

from analysis import load_spikes_vectorized, load_connectivity, load_params, get_n_stimuli


# =============================================================================
# Data Loading
# =============================================================================

def load_input_rates(stim_dir: str) -> np.ndarray:
    """
    Load input firing rates for a stimulus.

    Args:
        stim_dir: Path to stimulus directory (e.g., .../stim0/)

    Returns:
        Array of shape (n_inputs,) with firing rates in Hz
    """
    rates_file = Path(stim_dir) / 'input_rates.npy'
    return np.load(rates_file)


def compute_network_rates(directory: str, params: dict,
                          total_time: float = None,
                          start_from: float = 50) -> np.ndarray:
    """
    Compute mean firing rates of all network neurons across trials.

    Args:
        directory: Path to parameter point directory
        params: Simulation parameters dict
        total_time: Total simulation time in ms (None = read from params)
        start_from: Discard spikes before this time in ms

    Returns:
        Array of shape (n_neurons,) with mean firing rates in Hz
    """
    n_neurons = params['n_excite'] + params['n_inhib']
    n_trials = params['n_trials']
    n_stimuli = get_n_stimuli(params, directory)

    # Get total_time from params if not specified
    if total_time is None:
        total_time = params.get('simulation_time', 1.05) * 1000

    # Use full trial as single bin for rate computation
    effective_time = total_time - start_from  # ms

    # Accumulate spike counts across all stimuli and trials
    total_counts = np.zeros(n_neurons, dtype=np.float64)
    total_samples = 0

    for stim in range(n_stimuli):
        stim_dir = f"{directory}/stim{stim}"
        # Load with bin_size = effective_time to get total spike count per trial
        spikes = load_spikes_vectorized(
            stim_dir, n_trials, n_neurons,
            bin_size=effective_time, total_time=total_time, start_from=start_from
        )
        # spikes shape: (n_neurons, n_trials)
        total_counts += spikes.sum(axis=1)
        total_samples += n_trials

    # Convert to Hz: counts / (time in seconds) / n_samples
    rates = total_counts / (effective_time / 1000) / total_samples
    return rates


def compute_network_rates_per_stim(stim_dir: str, params: dict,
                                    total_time: float = None,
                                    start_from: float = 50) -> np.ndarray:
    """
    Compute mean firing rates for a single stimulus.

    Args:
        stim_dir: Path to stimulus directory
        params: Simulation parameters dict
        total_time: Total simulation time in ms (None = read from params)
        start_from: Discard spikes before this time in ms

    Returns:
        Array of shape (n_neurons,) with mean firing rates in Hz
    """
    n_neurons = params['n_excite'] + params['n_inhib']
    n_trials = params['n_trials']

    # Get total_time from params if not specified
    if total_time is None:
        total_time = params.get('simulation_time', 1.05) * 1000

    effective_time = total_time - start_from

    spikes = load_spikes_vectorized(
        stim_dir, n_trials, n_neurons,
        bin_size=effective_time, total_time=total_time, start_from=start_from
    )
    # spikes shape: (n_neurons, n_trials)
    # Mean across trials, convert to Hz
    rates = spikes.mean(axis=1) / (effective_time / 1000)
    return rates


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


def partial_correlation_multiple(x: np.ndarray, y: np.ndarray,
                                  controls: List[np.ndarray]) -> Tuple[float, float]:
    """
    Compute partial correlation r(X, Y | Z1, Z2, ...) controlling for multiple variables.

    Uses OLS regression to remove linear effects of all control variables.

    Args:
        x: First variable (e.g., noise correlations)
        y: Second variable (e.g., shared input)
        controls: List of control variables to partial out

    Returns:
        (r, p): Partial correlation coefficient and p-value
    """
    if len(controls) == 0:
        return stats.pearsonr(x, y)

    # Build design matrix with intercept
    Z = np.column_stack([np.ones(len(x))] + controls)

    # Regress X on controls and get residuals
    coeffs_x, _, _, _ = np.linalg.lstsq(Z, x, rcond=None)
    resid_x = x - Z @ coeffs_x

    # Regress Y on controls and get residuals
    coeffs_y, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
    resid_y = y - Z @ coeffs_y

    # Correlate residuals
    r, p = stats.pearsonr(resid_x, resid_y)
    return r, p


def multiple_regression_r2(y: np.ndarray, predictors: List[np.ndarray]) -> float:
    """
    Compute R² from multiple linear regression.

    Args:
        y: Dependent variable (e.g., noise correlations)
        predictors: List of predictor variables

    Returns:
        R² value (proportion of variance explained)
    """
    if len(predictors) == 0:
        return 0.0

    # Build design matrix with intercept
    X = np.column_stack([np.ones(len(y))] + predictors)

    # OLS fit
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    # Compute R²
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - ss_res / ss_tot
    return r2


# =============================================================================
# Sweep Traversal
# =============================================================================

def format_param_value(val) -> str:
    """Format a parameter value for directory name matching."""
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

    param1_values = sorted(set(param1_values))
    param2_values = sorted(set(param2_values))

    return param1_name, param1_values, param2_name, param2_values


# =============================================================================
# Shared Input Analysis (Rate-Weighted)
# =============================================================================

def compute_spike_correlations_single_stim(stim_dir: str, params: dict,
                                           neuron_indices: np.ndarray,
                                           bin_size: float = 1000,
                                           total_time: float = None,
                                           start_from: float = 50) -> np.ndarray:
    """
    Compute pairwise spike correlations for a single stimulus.

    Args:
        stim_dir: Path to stimulus directory (e.g., .../stim0/)
        params: Simulation parameters dict
        neuron_indices: Indices of neurons to analyze
        bin_size: Time bin size in ms for spike counting
        total_time: Total simulation time in ms (None = read from params)
        start_from: Discard spikes before this time in ms

    Returns:
        corrs: Pairwise correlations for lower triangle (n_pairs,)
    """
    n_neurons_total = params['n_excite'] + params['n_inhib']
    n_trials = params['n_trials']
    n_selected = len(neuron_indices)

    if total_time is None:
        total_time = params.get('simulation_time', 1.05) * 1000

    effective_time = total_time - start_from
    if bin_size > effective_time:
        bin_size = effective_time

    spikes = load_spikes_vectorized(
        stim_dir, n_trials, n_neurons_total,
        bin_size=bin_size, total_time=total_time, start_from=start_from
    )
    responses = spikes[neuron_indices]  # (n_selected, n_samples)

    corr_mat = np.corrcoef(responses)
    corr_mat[np.isnan(corr_mat)] = 0

    tril_indices = np.tril_indices(n_selected, k=-1)
    return corr_mat[tril_indices]


def compute_spike_correlations(directory: str, params: dict,
                                neuron_indices: np.ndarray,
                                bin_size: float = 1000,
                                total_time: float = None,
                                start_from: float = 50) -> np.ndarray:
    """
    Compute mean pairwise spike correlations across stimuli for selected neurons.

    Args:
        directory: Path to parameter point directory
        params: Simulation parameters dict
        neuron_indices: Indices of neurons to analyze
        bin_size: Time bin size in ms for spike counting
        total_time: Total simulation time in ms (None = read from params)
        start_from: Discard spikes before this time in ms

    Returns:
        (mean_corrs, tril_indices): Mean correlations and triangle indices
    """
    n_neurons_total = params['n_excite'] + params['n_inhib']
    n_stimuli = get_n_stimuli(params, directory)
    n_trials = params['n_trials']
    n_selected = len(neuron_indices)

    # Get total_time from params if not specified
    if total_time is None:
        total_time = params.get('simulation_time', 1.05) * 1000

    # Ensure bin_size doesn't exceed effective time
    effective_time = total_time - start_from
    if bin_size > effective_time:
        bin_size = effective_time

    n_bins = int(np.floor(effective_time / bin_size))
    if n_bins < 1:
        n_bins = 1
    n_samples = n_trials * n_bins

    all_responses = np.zeros((n_selected, n_stimuli, n_samples))

    for stim in range(n_stimuli):
        stim_dir = f"{directory}/stim{stim}"
        spikes = load_spikes_vectorized(
            stim_dir, n_trials, n_neurons_total,
            bin_size=bin_size, total_time=total_time, start_from=start_from
        )
        all_responses[:, stim, :] = spikes[neuron_indices]

    # Compute correlations for each stimulus, then average
    n_pairs = n_selected * (n_selected - 1) // 2
    tril_indices = np.tril_indices(n_selected, k=-1)

    all_corrs = np.zeros((n_stimuli, n_pairs))
    for stim in range(n_stimuli):
        corr_mat = np.corrcoef(all_responses[:, stim, :])
        corr_mat[np.isnan(corr_mat)] = 0
        all_corrs[stim] = corr_mat[tril_indices]

    mean_corrs = np.mean(all_corrs, axis=0)
    return mean_corrs, tril_indices


def compute_rate_weighted_shared_inputs(
    input_conn: np.ndarray,
    recurrent_conn: np.ndarray,
    neuron_indices: np.ndarray,
    n_excite: int,
    input_rates: np.ndarray,
    network_rates: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute normalized rate-weighted shared input for all pairs of selected neurons.

    Computes cosine similarity of effective weight vectors, where effective weight
    is w * sqrt(r). This equals the correlation of inputs for independent Poisson:

        shared_ij = Σ_k(w_ki w_kj r_k) / sqrt(Σ_k(w_ki² r_k) × Σ_k(w_kj² r_k))

    Args:
        input_conn: Input connectivity (n_inputs, n_neurons)
        recurrent_conn: Recurrent connectivity (n_neurons, n_neurons)
        neuron_indices: Indices of neurons to analyze
        n_excite: Number of excitatory neurons
        input_rates: Firing rates of input neurons (n_inputs,) in Hz
        network_rates: Firing rates of network neurons (n_neurons,) in Hz

    Returns:
        feedforward: Normalized FF shared input (input correlation) for each pair
        recurrent_exc: Normalized excitatory recurrent shared input
        recurrent_inh: Normalized inhibitory recurrent shared input
    """
    n_selected = len(neuron_indices)

    # Extract connectivity to selected neurons
    # input_conn[k, i] = weight from input k to neuron i
    # recurrent_conn[j, i] = weight from neuron i to neuron j
    input_to_selected = input_conn[:, neuron_indices]  # (n_inputs, n_selected)
    recurrent_to_selected = recurrent_conn[neuron_indices]  # (n_selected, n_neurons)

    # Separate excitatory and inhibitory recurrent
    # recurrent_to_selected[a, k] = weight from neuron k to selected neuron a
    exc_to_selected = recurrent_to_selected[:, :n_excite]  # (n_selected, n_excite)
    inh_to_selected = recurrent_to_selected[:, n_excite:]  # (n_selected, n_inhib)

    # Get rates for each population
    exc_rates = network_rates[:n_excite]  # (n_excite,)
    inh_rates = network_rates[n_excite:]  # (n_inhib,)

    # Compute effective weights: e_ki = w_ki * sqrt(r_k)
    # Then normalized shared input = cosine similarity of effective weight vectors
    # hacking this to not be the sqrt since I'm not sure it is
    sqrt_input_rates = np.sqrt(input_rates)
    sqrt_exc_rates = np.sqrt(exc_rates)
    sqrt_inh_rates = np.sqrt(inh_rates)
    """
    sqrt_input_rates = input_rates
    sqrt_exc_rates = exc_rates
    sqrt_inh_rates = inh_rates
    """

    # Feedforward effective weights: (n_inputs, n_selected)
    ff_effective = input_to_selected * sqrt_input_rates[:, None]
    # Dot product matrix
    ff_dot_matrix = ff_effective.T @ ff_effective  # (n_selected, n_selected)
    # Norms for each neuron
    ff_norms = np.sqrt(np.diag(ff_dot_matrix))  # (n_selected,)
    # Normalize: divide by outer product of norms
    ff_norm_matrix = np.outer(ff_norms, ff_norms)
    ff_norm_matrix[ff_norm_matrix == 0] = 1  # avoid division by zero
    ff_shared_matrix = ff_dot_matrix / ff_norm_matrix

    # Excitatory recurrent effective weights: (n_selected, n_excite)
    exc_effective = exc_to_selected * sqrt_exc_rates[None, :]
    exc_dot_matrix = exc_effective @ exc_effective.T  # (n_selected, n_selected)
    exc_norms = np.sqrt(np.diag(exc_dot_matrix))
    exc_norm_matrix = np.outer(exc_norms, exc_norms)
    exc_norm_matrix[exc_norm_matrix == 0] = 1
    exc_shared_matrix = exc_dot_matrix / exc_norm_matrix

    # Inhibitory recurrent effective weights: (n_selected, n_inhib)
    inh_effective = inh_to_selected * sqrt_inh_rates[None, :]
    inh_dot_matrix = inh_effective @ inh_effective.T  # (n_selected, n_selected)
    inh_norms = np.sqrt(np.diag(inh_dot_matrix))
    inh_norm_matrix = np.outer(inh_norms, inh_norms)
    inh_norm_matrix[inh_norm_matrix == 0] = 1
    inh_shared_matrix = inh_dot_matrix / inh_norm_matrix

    # Extract lower triangular (pairs)
    tril = np.tril_indices(n_selected, k=-1)

    feedforward = ff_shared_matrix[tril]
    recurrent_exc = exc_shared_matrix[tril]
    recurrent_inh = inh_shared_matrix[tril]

    return feedforward, recurrent_exc, recurrent_inh


def analyze_rate_weighted_correlations(directory: str,
                                        n_neurons_sample: int = 200,
                                        seed: int = 42,
                                        bin_size: float = 1000,
                                        total_time: float = None,
                                        start_from: float = 50,
                                        single_stimulus: bool = False) -> dict:
    """
    Compute correlations between spike correlations and rate-weighted shared input.

    Args:
        directory: Path to parameter point directory
        n_neurons_sample: Number of neurons to sample
        seed: Random seed for reproducibility
        bin_size: Time bin size in ms for spike counting
        total_time: Total simulation time in ms (None = read from params)
        start_from: Discard spikes before this time in ms
        single_stimulus: If True, use only stim0. If False (default), combine
                         across all stimuli treating each (pair, stim) as unique.

    Returns dict with Pearson r values for each shared input type.
    """
    np.random.seed(seed)

    params = load_params(directory)
    n_excite = params['n_excite']
    n_inhib = params['n_inhib']
    n_neurons = n_excite + n_inhib
    n_stimuli = get_n_stimuli(params, directory)

    # Get total_time from params if not specified
    if total_time is None:
        total_time = params.get('simulation_time', 1.05) * 1000

    # Ensure bin_size doesn't exceed effective time
    effective_time = total_time - start_from
    if bin_size > effective_time:
        bin_size = effective_time

    # Load connectivity (same for all stimuli)
    input_conn, recurrent_conn = load_connectivity(directory)

    # Sample neurons (same across stimuli)
    neuron_indices = np.random.choice(n_neurons, n_neurons_sample, replace=False)

    if single_stimulus:
        # Old behavior: only stim0
        stim_indices = [0]
    else:
        # New behavior: all stimuli
        stim_indices = list(range(n_stimuli))

    # Accumulate data across stimuli
    all_spike_corrs = []
    all_ff_shared = []
    all_exc_shared = []
    all_inh_shared = []
    all_stim_ids = []

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

        n_pairs = len(spike_corrs)
        all_spike_corrs.append(spike_corrs)
        all_ff_shared.append(ff_shared)
        all_exc_shared.append(exc_shared)
        all_inh_shared.append(inh_shared)
        all_stim_ids.append(np.full(n_pairs, stim))

    # Concatenate across stimuli
    spike_corrs = np.concatenate(all_spike_corrs)
    ff_shared = np.concatenate(all_ff_shared)
    exc_shared = np.concatenate(all_exc_shared)
    inh_shared = np.concatenate(all_inh_shared)
    stim_ids = np.concatenate(all_stim_ids)

    # Compute Pearson correlations (marginal)
    results = {}

    for name, shared in [('feedforward', ff_shared),
                          ('excitatory', exc_shared),
                          ('inhibitory', inh_shared)]:
        r, p = stats.pearsonr(shared, spike_corrs)
        results[name] = {'r': r, 'p': p}

    # Compute partial correlations controlling for all other input types
    # FF partial: r(noise_corr, ff | exc, inh)
    r_ff_partial, p_ff_partial = partial_correlation_multiple(
        spike_corrs, ff_shared, [exc_shared, inh_shared]
    )
    results['feedforward_partial'] = {'r': r_ff_partial, 'p': p_ff_partial}

    # Exc partial: r(noise_corr, exc | inh, ff)
    r_exc_partial, p_exc_partial = partial_correlation_multiple(
        spike_corrs, exc_shared, [inh_shared, ff_shared]
    )
    results['excitatory_partial'] = {'r': r_exc_partial, 'p': p_exc_partial}

    # Inh partial: r(noise_corr, inh | exc, ff)
    r_inh_partial, p_inh_partial = partial_correlation_multiple(
        spike_corrs, inh_shared, [exc_shared, ff_shared]
    )
    results['inhibitory_partial'] = {'r': r_inh_partial, 'p': p_inh_partial}

    # Full regression R²: how much variance explained by all three predictors
    r2_full = multiple_regression_r2(spike_corrs, [ff_shared, exc_shared, inh_shared])
    results['full_r2'] = {'r2': r2_full}

    # Store stim_ids for potential downstream use
    results['_stim_ids'] = stim_ids
    results['_n_stimuli_used'] = len(stim_indices)

    return results


# =============================================================================
# Sweep Analysis
# =============================================================================

def analyze_sweep_rate_weighted(sweep_dir: str,
                                 n_neurons_sample: int = 200,
                                 seed: int = 42,
                                 verbose: bool = True,
                                 bin_size: float = 1000,
                                 total_time: float = None,
                                 start_from: float = 50,
                                 single_stimulus: bool = False) -> dict:
    """
    Analyze rate-weighted shared input relationships across all parameter points.

    Args:
        sweep_dir: Path to sweep directory
        n_neurons_sample: Number of neurons to sample for analysis
        seed: Random seed for reproducibility
        verbose: Print progress
        bin_size: Time bin size in ms for spike counting
        total_time: Total simulation time in ms (None = read from params per point)
        start_from: Discard spikes before this time in ms
        single_stimulus: If True, use only stim0. If False (default), combine
                         across all stimuli.

    Returns dict with parameter info and 2D arrays of Pearson r values.
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
    r_feedforward_partial = np.full((n1, n2), np.nan)
    r_excitatory_partial = np.full((n1, n2), np.nan)
    r_inhibitory_partial = np.full((n1, n2), np.nan)
    r2_full = np.full((n1, n2), np.nan)
    completed = np.zeros((n1, n2), dtype=bool)

    # Analyze each point
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

            # Check for input_rates.npy
            if not (stim_dirs[0] / 'input_rates.npy').exists():
                if verbose:
                    print(f"  [{analyzed+1}/{total_points}] {dir_name.name}: NO INPUT RATES")
                analyzed += 1
                continue

            try:
                if verbose:
                    print(f"  [{analyzed+1}/{total_points}] Analyzing {param1_name}={v1}, {param2_name}={v2}...")

                results = analyze_rate_weighted_correlations(
                    str(dir_name),
                    n_neurons_sample=n_neurons_sample,
                    seed=seed,
                    bin_size=bin_size,
                    total_time=total_time,
                    start_from=start_from,
                    single_stimulus=single_stimulus
                )

                r_feedforward[i, j] = results['feedforward']['r']
                r_excitatory[i, j] = results['excitatory']['r']
                r_inhibitory[i, j] = results['inhibitory']['r']
                r_feedforward_partial[i, j] = results['feedforward_partial']['r']
                r_excitatory_partial[i, j] = results['excitatory_partial']['r']
                r_inhibitory_partial[i, j] = results['inhibitory_partial']['r']
                r2_full[i, j] = results['full_r2']['r2']
                completed[i, j] = True

                if verbose:
                    print(f"    -> r_ff={r_feedforward[i,j]:.3f}, r_exc={r_excitatory[i,j]:.3f}, "
                          f"r_inh={r_inhibitory[i,j]:.3f}")
                    print(f"    -> partials: r_ff|EI={r_feedforward_partial[i,j]:.3f}, "
                          f"r_exc|IF={r_excitatory_partial[i,j]:.3f}, "
                          f"r_inh|EF={r_inhibitory_partial[i,j]:.3f}")
                    print(f"    -> full R²={r2_full[i,j]:.3f}")

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
        'r_feedforward_partial': r_feedforward_partial,
        'r_excitatory_partial': r_excitatory_partial,
        'r_inhibitory_partial': r_inhibitory_partial,
        'r2_full': r2_full,
        'completed': completed
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_heatmaps(results: dict, fig_dir: str = "figures"):
    """
    Plot and save heatmaps for rate-weighted shared input-correlation relationships.
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
    ]

    # Create combined figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Use symmetric color limits centered at 0
    all_r_values = np.concatenate([
        results['r_feedforward'].flatten(),
        results['r_excitatory'].flatten(),
        results['r_inhibitory'].flatten()
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

        ax.set_xticks(range(len(param2_values)))
        ax.set_xticklabels([f"{v:.0f}" if isinstance(v, float) and v.is_integer() else f"{v}"
                           for v in param2_values], rotation=45, ha='right')
        ax.set_yticks(range(len(param1_values)))
        ax.set_yticklabels([f"{v}" for v in param1_values])

        ax.set_xlabel(param2_name)
        ax.set_ylabel(param1_name)
        ax.set_title(title)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Pearson r')

        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                if not np.isnan(data[i, j]):
                    val = data[i, j]
                    text = f"{val:.2f}"
                    color = 'white' if abs(val) > vmax * 0.6 else 'black'
                    ax.text(j, i, text, ha='center', va='center',
                            fontsize=8, color=color)

    plt.suptitle(f'Rate-Weighted Shared Input - Noise Correlation\n{param1_name} vs {param2_name}',
                 fontsize=14)
    plt.tight_layout()

    combined_path = fig_path / 'rate_weighted_input_heatmaps.png'
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined heatmap: {combined_path}")

    # Partial correlations figure (2x2: three partials + R²)
    partial_metrics = [
        ('r_feedforward_partial', 'r(Noise, FF | Exc, Inh)', 'RdBu_r'),
        ('r_excitatory_partial', 'r(Noise, Exc | Inh, FF)', 'RdBu_r'),
        ('r_inhibitory_partial', 'r(Noise, Inh | Exc, FF)', 'RdBu_r'),
        ('r2_full', 'Full R² (FF + Exc + Inh)', 'viridis'),
    ]

    partial_r_values = np.concatenate([
        results['r_feedforward_partial'].flatten(),
        results['r_excitatory_partial'].flatten(),
        results['r_inhibitory_partial'].flatten()
    ])
    partial_r_values = partial_r_values[~np.isnan(partial_r_values)]
    if len(partial_r_values) > 0:
        vmax_partial = np.max(np.abs(partial_r_values)) * 1.1
        vmin_partial = -vmax_partial
    else:
        vmin_partial, vmax_partial = -1, 1

    fig_partial, axes_partial = plt.subplots(2, 2, figsize=(12, 10))
    axes_partial = axes_partial.flatten()

    for ax, (key, title, cmap) in zip(axes_partial, partial_metrics):
        data = results[key]
        masked_data = np.ma.masked_invalid(data)

        # R² uses different color scale (0 to max, not symmetric)
        if key == 'r2_full':
            r2_values = data[~np.isnan(data)]
            if len(r2_values) > 0:
                v_min, v_max = 0, max(0.1, np.max(r2_values) * 1.1)
            else:
                v_min, v_max = 0, 1
        else:
            v_min, v_max = vmin_partial, vmax_partial

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
        cbar.set_label('R²' if key == 'r2_full' else 'Partial r')

        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                if not np.isnan(data[i, j]):
                    val = data[i, j]
                    text = f"{val:.2f}"
                    if key == 'r2_full':
                        color = 'white' if val > v_max * 0.5 else 'black'
                    else:
                        color = 'white' if abs(val) > vmax_partial * 0.6 else 'black'
                    ax.text(j, i, text, ha='center', va='center',
                            fontsize=8, color=color)

    plt.suptitle(f'Rate-Weighted Partial Correlations & Full R²\n{param1_name} vs {param2_name}',
                 fontsize=14)
    plt.tight_layout()

    partial_path = fig_path / 'rate_weighted_partial_heatmaps.png'
    plt.savefig(partial_path, dpi=150, bbox_inches='tight')
    print(f"Saved partial correlation heatmap: {partial_path}")

    # Individual heatmaps
    all_metrics = metrics + partial_metrics
    for key, title, cmap in all_metrics:
        if key == 'r2_full':
            r2_values = results[key][~np.isnan(results[key])]
            if len(r2_values) > 0:
                v_min, v_max = 0, max(0.1, np.max(r2_values) * 1.1)
            else:
                v_min, v_max = 0, 1
        elif 'partial' in key:
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
        if key == 'r2_full':
            cbar.set_label('R²')
        elif 'partial' in key:
            cbar.set_label('Partial r')
        else:
            cbar.set_label('Pearson r')

        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                if not np.isnan(data[i, j]):
                    val = data[i, j]
                    text = f"{val:.2f}"
                    if key == 'r2_full':
                        color = 'white' if val > v_max * 0.5 else 'black'
                    else:
                        color = 'white' if abs(val) > v_max * 0.6 else 'black'
                    ax.text(j, i, text, ha='center', va='center',
                            fontsize=8, color=color)

        plt.tight_layout()
        individual_path = fig_path / f'rate_weighted_{key}_heatmap.png'
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
        description='Generate heatmaps of rate-weighted shared input vs noise correlation relationships'
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
    parser.add_argument('--total_time', type=float, default=None,
                        help='Total simulation time in ms (default: read from params)')
    parser.add_argument('--start_from', type=float, default=50,
                        help='Discard spikes before this time in ms (default: 50)')
    parser.add_argument('--single_stimulus', action='store_true',
                        help='Use only stim0 (default: combine across all stimuli)')

    args = parser.parse_args()

    stim_mode = "single stimulus (stim0)" if args.single_stimulus else "all stimuli combined"
    print(f"Analyzing sweep: {args.sweep_dir}")
    print(f"Settings: n_neurons={args.n_neurons}, seed={args.seed}, bin_size={args.bin_size}ms")
    print(f"Stimulus mode: {stim_mode}")
    print(f"Using RATE-WEIGHTED shared input (Σ w_ki w_kj r_k)")
    print()

    results = analyze_sweep_rate_weighted(
        args.sweep_dir,
        n_neurons_sample=args.n_neurons,
        seed=args.seed,
        verbose=True,
        bin_size=args.bin_size,
        total_time=args.total_time,
        start_from=args.start_from,
        single_stimulus=args.single_stimulus
    )

    n_completed = results['completed'].sum()
    n_total = results['completed'].size
    print(f"\nCompleted {n_completed}/{n_total} parameter points")

    if n_completed > 0:
        print("\nGenerating heatmaps...")
        plot_heatmaps(results, args.fig_dir)

        print("\n" + "="*70)
        print("SUMMARY: Mean values across completed parameter points")
        print("="*70)
        print("Marginal correlations:")
        for key in ['r_feedforward', 'r_excitatory', 'r_inhibitory']:
            data = results[key]
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                print(f"  {key:25s}: mean={np.mean(valid):+.3f}, "
                      f"range=[{np.min(valid):+.3f}, {np.max(valid):+.3f}]")
        print("-"*70)
        print("Partial correlations (controlling for all other input types):")
        for key in ['r_feedforward_partial', 'r_excitatory_partial', 'r_inhibitory_partial']:
            data = results[key]
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                print(f"  {key:25s}: mean={np.mean(valid):+.3f}, "
                      f"range=[{np.min(valid):+.3f}, {np.max(valid):+.3f}]")
        print("-"*70)
        print("Full model R² (FF + Exc + Inh -> Noise Corr):")
        data = results['r2_full']
        valid = data[~np.isnan(data)]
        if len(valid) > 0:
            print(f"  {'r2_full':25s}: mean={np.mean(valid):.3f}, "
                  f"range=[{np.min(valid):.3f}, {np.max(valid):.3f}]")
        print("="*70)
        print("\nDone!")
    else:
        print("\nNo completed runs found - skipping heatmap generation")


if __name__ == '__main__':
    main()
