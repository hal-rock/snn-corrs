"""
Generate heatmaps of actual input covariance vs noise correlation relationships.

Unlike input_heatmaps.py which uses the Poisson proxy (Σ w_ki w_kj r_k),
this script computes actual trial-by-trial input covariance by:
1. Loading real per-trial spike counts for external inputs and network neurons
2. Multiplying through connectivity matrices to get each neuron's actual received input
3. Computing pairwise correlations of these inputs across trials

This captures real correlations in presynaptic firing (e.g., correlated network
activity for recurrent inputs) that the Poisson proxy misses.

For each parameter point, computes Pearson correlation between:
- Pairwise spike (noise) correlations
- Actual input correlations (feedforward, excitatory, inhibitory)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
from scipy import stats

from analysis import load_spikes_vectorized, load_connectivity, load_params, get_n_stimuli
from input_heatmaps import (
    discover_sweep_structure, format_param_value,
    partial_correlation_multiple, multiple_regression_r2,
    compute_spike_correlations_single_stim,
)


# =============================================================================
# Data Loading
# =============================================================================

def load_input_spike_counts(stim_dir: str, n_trials: int, n_inputs: int,
                            start_from: float = 50,
                            total_time: float = 1050) -> np.ndarray:
    """
    Load per-trial spike counts for external input neurons (vectorized).

    NOTE: File naming is swapped in the simulation code:
    - input_times.npz actually contains neuron INDICES
    - input_spikes.npz actually contains spike TIMES

    Args:
        stim_dir: Path to stimulus directory
        n_trials: Number of trials
        n_inputs: Number of input neurons
        start_from: Only count spikes at or after this time (ms)
        total_time: Only count spikes before this time (ms)

    Returns:
        Array of shape (n_inputs, n_trials) with spike counts
    """
    # input_times.npz stores neuron indices, input_spikes.npz stores spike times
    inpt_indices_file = np.load(f'{stim_dir}/input_times.npz')
    inpt_times_file = np.load(f'{stim_dir}/input_spikes.npz')

    counts = np.zeros((n_inputs, n_trials), dtype=np.int32)

    for trial in range(min(n_trials, len(inpt_indices_file.files))):
        indices = inpt_indices_file[f'arr_{trial}']  # neuron indices
        times = inpt_times_file[f'arr_{trial}']      # spike times in ms

        # Filter by time window
        mask = (times >= start_from) & (times < total_time)
        filtered_indices = indices[mask].astype(int)

        if len(filtered_indices) > 0:
            counts[:, trial] = np.bincount(filtered_indices, minlength=n_inputs)[:n_inputs]

    inpt_indices_file.close()
    inpt_times_file.close()
    return counts


def load_trial_mean_currents(stim_dir: str, n_trials: int,
                              neuron_indices: np.ndarray, n_excite: int,
                              start_from: float = 50.0,
                              total_time: float = 1050.0) -> tuple:
    """
    Load recorded currents and time-average over [start_from, total_time).

    Args:
        stim_dir: Path to stimulus directory (contains trial0/, trial1/, ...)
        n_trials: Number of trials
        neuron_indices: Indices of neurons to load (0-indexed across E+I)
        n_excite: Number of excitatory neurons
        start_from: Start of time window in ms
        total_time: End of time window in ms

    Returns:
        ff_input: (n_selected, n_trials) — time-averaged I_ext (pA)
        exc_input: (n_selected, n_trials) — time-averaged I_rec_exc (pA)
        inh_input: (n_selected, n_trials) — time-averaged I_inh (pA, negative)
    """
    n_selected = len(neuron_indices)

    # Split neuron indices into E and I populations
    e_mask = neuron_indices < n_excite
    i_mask = ~e_mask
    e_idx = neuron_indices[e_mask]
    i_idx = neuron_indices[i_mask] - n_excite
    e_pos = np.where(e_mask)[0]
    i_pos = np.where(i_mask)[0]

    ff_input = np.zeros((n_selected, n_trials))
    exc_input = np.zeros((n_selected, n_trials))
    inh_input = np.zeros((n_selected, n_trials))

    for trial in range(n_trials):
        trial_dir = f'{stim_dir}/trial{trial}'

        if len(e_idx) > 0:
            with np.load(f'{trial_dir}/currents_e.npz') as ce:
                t = ce['t']
                tmask = (t >= start_from) & (t < total_time)
                ff_input[e_pos, trial] = ce['I_ext'][e_idx][:, tmask].mean(axis=1)
                exc_input[e_pos, trial] = ce['I_rec_exc'][e_idx][:, tmask].mean(axis=1)
                inh_input[e_pos, trial] = ce['I_inh'][e_idx][:, tmask].mean(axis=1)

        if len(i_idx) > 0:
            with np.load(f'{trial_dir}/currents_i.npz') as ci:
                t = ci['t']
                tmask = (t >= start_from) & (t < total_time)
                ff_input[i_pos, trial] = ci['I_ext'][i_idx][:, tmask].mean(axis=1)
                exc_input[i_pos, trial] = ci['I_rec_exc'][i_idx][:, tmask].mean(axis=1)
                inh_input[i_pos, trial] = ci['I_inh'][i_idx][:, tmask].mean(axis=1)

    return ff_input, exc_input, inh_input


# =============================================================================
# Core Computation
# =============================================================================

def pairwise_input_correlations(
    ff_input: np.ndarray,
    exc_input: np.ndarray,
    inh_input: np.ndarray,
    use_covariance: bool = False,
    compute_total: bool = False
) -> tuple:
    """
    Compute pairwise input correlations from per-neuron, per-trial input arrays.

    Args:
        ff_input: (n_selected, n_trials) — feedforward input per neuron per trial
        exc_input: (n_selected, n_trials) — excitatory recurrent input
        inh_input: (n_selected, n_trials) — inhibitory recurrent input
        use_covariance: If True, compute raw covariance instead of correlation
        compute_total: If True, also compute total (FF+Exc+Inh) pairwise correlations.
            Only meaningful when all inputs share common units (e.g. pA from recorded currents).

    Returns:
        feedforward, recurrent_exc, recurrent_inh, exc_ff_cross, inh_ff_cross
        Each is (n_pairs,) array. If compute_total, also returns total as 6th element.
    """
    n = ff_input.shape[0]
    pairwise_func = np.cov if use_covariance else np.corrcoef

    input_list = [ff_input, exc_input, inh_input]
    if compute_total:
        input_list.append(ff_input + exc_input + inh_input)

    all_inputs = np.vstack(input_list)
    full_mat = pairwise_func(all_inputs)
    full_mat[np.isnan(full_mat)] = 0

    tril = np.tril_indices(n, k=-1)

    # Same-type (diagonal blocks)
    feedforward = full_mat[:n, :n][tril]
    recurrent_exc = full_mat[n:2*n, n:2*n][tril]
    recurrent_inh = full_mat[2*n:3*n, 2*n:3*n][tril]

    # Cross-type (off-diagonal blocks), symmetrized over pair
    ff_exc_block = full_mat[:n, n:2*n]
    exc_ff_cross = (ff_exc_block[tril[0], tril[1]] + ff_exc_block[tril[1], tril[0]]) / 2

    ff_inh_block = full_mat[:n, 2*n:3*n]
    inh_ff_cross = (ff_inh_block[tril[0], tril[1]] + ff_inh_block[tril[1], tril[0]]) / 2

    if compute_total:
        total = full_mat[3*n:, 3*n:][tril]
        return feedforward, recurrent_exc, recurrent_inh, exc_ff_cross, inh_ff_cross, total

    return feedforward, recurrent_exc, recurrent_inh, exc_ff_cross, inh_ff_cross


def compute_actual_input_correlations(
    input_conn: np.ndarray,
    recurrent_conn: np.ndarray,
    neuron_indices: np.ndarray,
    n_excite: int,
    input_counts: np.ndarray,
    network_counts: np.ndarray,
    use_covariance: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute actual pairwise input correlations from trial-by-trial spike data.

    Computes per-neuron weighted input via W @ spike_counts, then delegates
    to pairwise_input_correlations for the pairwise correlation computation.

    Args:
        input_conn: Input connectivity (n_inputs, n_neurons)
        recurrent_conn: Recurrent connectivity (n_neurons, n_neurons)
        neuron_indices: Indices of neurons to analyze
        n_excite: Number of excitatory neurons
        input_counts: External input spike counts (n_inputs, n_trials)
        network_counts: Network spike counts (n_neurons, n_trials)
        use_covariance: If True, compute raw covariance instead of correlation

    Returns:
        feedforward, recurrent_exc, recurrent_inh, exc_ff_cross, inh_ff_cross
        Each is (n_pairs,) array of pairwise correlations/covariances.
    """
    # Connectivity slices for selected postsynaptic neurons
    input_to_selected = input_conn[:, neuron_indices]       # (n_inputs, n_selected)
    rec_to_selected = recurrent_conn[neuron_indices, :]     # (n_selected, n_neurons)

    # Compute actual weighted input per trial for each neuron
    ff_input = input_to_selected.T @ input_counts
    exc_input = rec_to_selected[:, :n_excite] @ network_counts[:n_excite]
    inh_input = rec_to_selected[:, n_excite:] @ network_counts[n_excite:]

    return pairwise_input_correlations(
        ff_input, exc_input, inh_input, use_covariance=use_covariance
    )


# =============================================================================
# Per-Parameter-Point Analysis
# =============================================================================

def analyze_actual_input_correlations(directory: str,
                                      n_neurons_sample: int = 200,
                                      seed: int = 42,
                                      bin_size: float = 1000,
                                      total_time: float = None,
                                      start_from: float = 50,
                                      single_stimulus: bool = False,
                                      use_covariance: bool = False,
                                      use_currents: bool = False) -> dict:
    """
    Compute correlations between spike noise correlations and actual input correlations/covariances.

    Args:
        directory: Path to parameter point directory
        n_neurons_sample: Number of neurons to sample
        seed: Random seed for reproducibility
        bin_size: Time bin size in ms for spike counting (output noise correlations)
        total_time: Total simulation time in ms (None = read from params)
        start_from: Discard spikes before this time in ms
        single_stimulus: If True, use only stim0
        use_covariance: If True, use raw covariance instead of correlation for inputs
        use_currents: If True, use recorded currents instead of W @ spike_counts

    Returns dict with Pearson r values for each input type.
    """
    np.random.seed(seed)

    params = load_params(directory)
    n_excite = params['n_excite']
    n_inhib = params['n_inhib']
    n_neurons = n_excite + n_inhib
    n_inputs = params.get('n_inputs', 3000)
    n_stimuli = get_n_stimuli(params, directory)
    n_trials = params['n_trials']

    if total_time is None:
        total_time = params.get('simulation_time', 1.05) * 1000

    effective_time = total_time - start_from
    if bin_size > effective_time:
        bin_size = effective_time

    # Load connectivity (only needed for W @ counts method)
    if not use_currents:
        input_conn, recurrent_conn = load_connectivity(directory)

    # Sample neurons
    neuron_indices = np.random.choice(n_neurons, n_neurons_sample, replace=False)

    stim_indices = [0] if single_stimulus else list(range(n_stimuli))

    # Accumulate across stimuli
    all_spike_corrs = []
    all_ff_corrs = []
    all_exc_corrs = []
    all_inh_corrs = []
    all_exc_ff_cross = []
    all_inh_ff_cross = []
    all_total_corrs = []

    for stim in stim_indices:
        stim_dir = f"{directory}/stim{stim}"

        if use_currents:
            # Load recorded currents, time-average per trial
            ff_input, exc_input, inh_input = load_trial_mean_currents(
                stim_dir, n_trials, neuron_indices, n_excite,
                start_from=start_from, total_time=total_time
            )
            ff_corrs, exc_corrs, inh_corrs, exc_ff, inh_ff, total_corrs = \
                pairwise_input_correlations(
                    ff_input, exc_input, inh_input,
                    use_covariance=use_covariance, compute_total=True
                )
            all_total_corrs.append(total_corrs)
        else:
            # Original method: W @ spike_counts
            input_counts = load_input_spike_counts(
                stim_dir, n_trials, n_inputs,
                start_from=start_from, total_time=total_time
            )
            network_counts = load_spikes_vectorized(
                stim_dir, n_trials, n_neurons,
                bin_size=effective_time, total_time=total_time, start_from=start_from
            )
            ff_corrs, exc_corrs, inh_corrs, exc_ff, inh_ff = compute_actual_input_correlations(
                input_conn, recurrent_conn, neuron_indices, n_excite,
                input_counts, network_counts, use_covariance=use_covariance
            )

        # Compute output noise correlations
        spike_corrs = compute_spike_correlations_single_stim(
            stim_dir, params, neuron_indices,
            bin_size=bin_size, total_time=total_time, start_from=start_from
        )

        all_spike_corrs.append(spike_corrs)
        all_ff_corrs.append(ff_corrs)
        all_exc_corrs.append(exc_corrs)
        all_inh_corrs.append(inh_corrs)
        all_exc_ff_cross.append(exc_ff)
        all_inh_ff_cross.append(inh_ff)

    # Concatenate across stimuli
    spike_corrs = np.concatenate(all_spike_corrs)
    ff_corrs = np.concatenate(all_ff_corrs)
    exc_corrs = np.concatenate(all_exc_corrs)
    inh_corrs = np.concatenate(all_inh_corrs)
    exc_ff_cross = np.concatenate(all_exc_ff_cross)
    inh_ff_cross = np.concatenate(all_inh_ff_cross)

    all_predictors = [ff_corrs, exc_corrs, inh_corrs, exc_ff_cross, inh_ff_cross]

    # Marginal Pearson correlations
    results = {}
    for name, input_corr in [('feedforward', ff_corrs),
                              ('excitatory', exc_corrs),
                              ('inhibitory', inh_corrs),
                              ('exc_ff_cross', exc_ff_cross),
                              ('inh_ff_cross', inh_ff_cross)]:
        r, p = stats.pearsonr(input_corr, spike_corrs)
        results[name] = {'r': r, 'p': p}

    # Total input correlation (only available with recorded currents)
    if use_currents:
        total_corrs = np.concatenate(all_total_corrs)
        r, p = stats.pearsonr(total_corrs, spike_corrs)
        results['total'] = {'r': r, 'p': p}
        r2_total = multiple_regression_r2(spike_corrs, [total_corrs])
        results['total_r2'] = {'r2': r2_total}

    # Partial correlations (each controlling for all others)
    predictor_names = ['feedforward', 'excitatory', 'inhibitory', 'exc_ff_cross', 'inh_ff_cross']
    predictor_arrays = [ff_corrs, exc_corrs, inh_corrs, exc_ff_cross, inh_ff_cross]

    for idx, name in enumerate(predictor_names):
        controls = [predictor_arrays[k] for k in range(len(predictor_arrays)) if k != idx]
        r_partial, p_partial = partial_correlation_multiple(
            spike_corrs, predictor_arrays[idx], controls
        )
        results[f'{name}_partial'] = {'r': r_partial, 'p': p_partial}

    # Full regression R² with all 5 predictors
    r2_full = multiple_regression_r2(spike_corrs, all_predictors)
    results['full_r2'] = {'r2': r2_full}

    # Also compute R² without cross terms for comparison
    r2_same_only = multiple_regression_r2(spike_corrs, [ff_corrs, exc_corrs, inh_corrs])
    results['same_type_r2'] = {'r2': r2_same_only}

    results['_n_stimuli_used'] = len(stim_indices)

    return results


# =============================================================================
# Sweep Analysis
# =============================================================================

def analyze_sweep_actual(sweep_dir: str,
                         n_neurons_sample: int = 200,
                         seed: int = 42,
                         verbose: bool = True,
                         bin_size: float = 1000,
                         total_time: float = None,
                         start_from: float = 50,
                         single_stimulus: bool = False,
                         use_covariance: bool = False,
                         use_currents: bool = False) -> dict:
    """
    Analyze actual input covariance relationships across all parameter points.
    """
    sweep_path = Path(sweep_dir)

    param1_name, param1_values, param2_name, param2_values = discover_sweep_structure(sweep_dir)

    n1 = len(param1_values)
    n2 = len(param2_values)

    if verbose:
        print(f"Sweep structure: {param1_name} ({n1} values) x {param2_name} ({n2} values)")
        print(f"  {param1_name}: {param1_values}")
        print(f"  {param2_name}: {param2_values}")

    r_feedforward = np.full((n1, n2), np.nan)
    r_excitatory = np.full((n1, n2), np.nan)
    r_inhibitory = np.full((n1, n2), np.nan)
    r_exc_ff_cross = np.full((n1, n2), np.nan)
    r_inh_ff_cross = np.full((n1, n2), np.nan)
    r_feedforward_partial = np.full((n1, n2), np.nan)
    r_excitatory_partial = np.full((n1, n2), np.nan)
    r_inhibitory_partial = np.full((n1, n2), np.nan)
    r_exc_ff_cross_partial = np.full((n1, n2), np.nan)
    r_inh_ff_cross_partial = np.full((n1, n2), np.nan)
    r2_full = np.full((n1, n2), np.nan)
    r2_same_only = np.full((n1, n2), np.nan)
    r_total = np.full((n1, n2), np.nan)
    r2_total = np.full((n1, n2), np.nan)
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

            # Check for required data
            if use_currents:
                trial_dirs_check = list(stim_dirs[0].glob("trial*"))
                if not trial_dirs_check or not (trial_dirs_check[0] / 'currents_e.npz').exists():
                    if verbose:
                        print(f"  [{analyzed+1}/{total_points}] {dir_name.name}: NO CURRENT DATA")
                    analyzed += 1
                    continue
            else:
                if not (stim_dirs[0] / 'input_times.npz').exists():
                    if verbose:
                        print(f"  [{analyzed+1}/{total_points}] {dir_name.name}: NO INPUT SPIKE DATA")
                    analyzed += 1
                    continue

            try:
                if verbose:
                    print(f"  [{analyzed+1}/{total_points}] Analyzing {param1_name}={v1}, {param2_name}={v2}...")

                results = analyze_actual_input_correlations(
                    str(dir_name),
                    n_neurons_sample=n_neurons_sample,
                    seed=seed,
                    bin_size=bin_size,
                    total_time=total_time,
                    start_from=start_from,
                    single_stimulus=single_stimulus,
                    use_covariance=use_covariance,
                    use_currents=use_currents
                )

                r_feedforward[i, j] = results['feedforward']['r']
                r_excitatory[i, j] = results['excitatory']['r']
                r_inhibitory[i, j] = results['inhibitory']['r']
                r_exc_ff_cross[i, j] = results['exc_ff_cross']['r']
                r_inh_ff_cross[i, j] = results['inh_ff_cross']['r']
                r_feedforward_partial[i, j] = results['feedforward_partial']['r']
                r_excitatory_partial[i, j] = results['excitatory_partial']['r']
                r_inhibitory_partial[i, j] = results['inhibitory_partial']['r']
                r_exc_ff_cross_partial[i, j] = results['exc_ff_cross_partial']['r']
                r_inh_ff_cross_partial[i, j] = results['inh_ff_cross_partial']['r']
                r2_full[i, j] = results['full_r2']['r2']
                r2_same_only[i, j] = results['same_type_r2']['r2']
                if 'total' in results:
                    r_total[i, j] = results['total']['r']
                    r2_total[i, j] = results['total_r2']['r2']
                completed[i, j] = True

                if verbose:
                    print(f"    -> r_ff={r_feedforward[i,j]:.3f}, r_exc={r_excitatory[i,j]:.3f}, "
                          f"r_inh={r_inhibitory[i,j]:.3f}")
                    print(f"    -> r_exc×ff={r_exc_ff_cross[i,j]:.3f}, "
                          f"r_inh×ff={r_inh_ff_cross[i,j]:.3f}")
                    print(f"    -> partials: r_ff={r_feedforward_partial[i,j]:.3f}, "
                          f"r_exc={r_excitatory_partial[i,j]:.3f}, "
                          f"r_inh={r_inhibitory_partial[i,j]:.3f}")
                    print(f"    -> partials: r_exc×ff={r_exc_ff_cross_partial[i,j]:.3f}, "
                          f"r_inh×ff={r_inh_ff_cross_partial[i,j]:.3f}")
                    r2_line = (f"    -> R²: same-type={r2_same_only[i,j]:.3f}, "
                              f"full(+cross)={r2_full[i,j]:.3f}")
                    if 'total' in results:
                        print(f"    -> r_total={r_total[i,j]:.3f}")
                        r2_line += f", total-only={r2_total[i,j]:.3f}"
                    print(r2_line)

            except Exception as e:
                if verbose:
                    print(f"    -> ERROR: {e}")

            analyzed += 1

    result = {
        'param1_name': param1_name,
        'param1_values': param1_values,
        'param2_name': param2_name,
        'param2_values': param2_values,
        'r_feedforward': r_feedforward,
        'r_excitatory': r_excitatory,
        'r_inhibitory': r_inhibitory,
        'r_exc_ff_cross': r_exc_ff_cross,
        'r_inh_ff_cross': r_inh_ff_cross,
        'r_feedforward_partial': r_feedforward_partial,
        'r_excitatory_partial': r_excitatory_partial,
        'r_inhibitory_partial': r_inhibitory_partial,
        'r_exc_ff_cross_partial': r_exc_ff_cross_partial,
        'r_inh_ff_cross_partial': r_inh_ff_cross_partial,
        'r2_full': r2_full,
        'r2_same_only': r2_same_only,
        'completed': completed
    }
    if use_currents:
        result['r_total'] = r_total
        result['r2_total'] = r2_total
    return result


# =============================================================================
# Plotting
# =============================================================================

def plot_heatmaps(results: dict, fig_dir: str = "figures", use_covariance: bool = False):
    """
    Plot and save heatmaps for actual input correlation/covariance-noise correlation relationships.
    """
    fig_path = Path(fig_dir)
    fig_path.mkdir(exist_ok=True)

    param1_name = results['param1_name']
    param2_name = results['param2_name']
    param1_values = results['param1_values']
    param2_values = results['param2_values']

    mode = "Cov" if use_covariance else "Corr"
    prefix = "actual_input_cov" if use_covariance else "actual_input_corr"

    metrics = [
        ('r_feedforward', f'r(Noise, FF {mode})', 'RdBu_r'),
        ('r_excitatory', f'r(Noise, Exc {mode})', 'RdBu_r'),
        ('r_inhibitory', f'r(Noise, Inh {mode})', 'RdBu_r'),
        ('r_exc_ff_cross', f'r(Noise, Exc×FF {mode})', 'RdBu_r'),
        ('r_inh_ff_cross', f'r(Noise, Inh×FF {mode})', 'RdBu_r'),
    ]
    if 'r_total' in results:
        metrics.append(('r_total', f'r(Noise, Total {mode})', 'RdBu_r'))

    # Combined marginal figure
    n_marginal = len(metrics)
    fig, axes = plt.subplots(1, n_marginal, figsize=(5 * n_marginal, 5))

    all_r_values = np.concatenate([
        results[key].flatten() for key, _, _ in metrics
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

    plt.suptitle(f'Actual Input {mode} - Noise Correlation\n{param1_name} vs {param2_name}',
                 fontsize=14)
    plt.tight_layout()

    combined_path = fig_path / f'{prefix}_heatmaps.png'
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined heatmap: {combined_path}")

    # Partial correlations + R² figure
    partial_metrics = [
        ('r_feedforward_partial', f'r(Noise, FF {mode} | rest)', 'RdBu_r'),
        ('r_excitatory_partial', f'r(Noise, Exc {mode} | rest)', 'RdBu_r'),
        ('r_inhibitory_partial', f'r(Noise, Inh {mode} | rest)', 'RdBu_r'),
        ('r_exc_ff_cross_partial', f'r(Noise, Exc×FF {mode} | rest)', 'RdBu_r'),
        ('r_inh_ff_cross_partial', f'r(Noise, Inh×FF {mode} | rest)', 'RdBu_r'),
        ('r2_same_only', 'R² same-type only', 'viridis'),
        ('r2_full', 'R² full (+ cross)', 'viridis'),
    ]
    if 'r2_total' in results:
        partial_metrics.append(('r2_total', 'R² total input only', 'viridis'))

    partial_r_values = np.concatenate([
        results[key].flatten()
        for key, _, _ in partial_metrics
        if 'r2' not in key
    ])
    partial_r_values = partial_r_values[~np.isnan(partial_r_values)]
    if len(partial_r_values) > 0:
        vmax_partial = np.max(np.abs(partial_r_values)) * 1.1
        vmin_partial = -vmax_partial
    else:
        vmin_partial, vmax_partial = -1, 1

    fig_partial, axes_partial = plt.subplots(2, 4, figsize=(24, 10))
    axes_partial = axes_partial.flatten()

    for ax_idx, ax in enumerate(axes_partial):
        if ax_idx >= len(partial_metrics):
            ax.set_visible(False)
            continue

        key, title, cmap = partial_metrics[ax_idx]
        data = results[key]
        masked_data = np.ma.masked_invalid(data)

        if 'r2' in key:
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
        cbar.set_label('R²' if 'r2' in key else 'Partial r')

        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                if not np.isnan(data[i, j]):
                    val = data[i, j]
                    text = f"{val:.2f}"
                    if 'r2' in key:
                        color = 'white' if val > v_max * 0.5 else 'black'
                    else:
                        color = 'white' if abs(val) > vmax_partial * 0.6 else 'black'
                    ax.text(j, i, text, ha='center', va='center',
                            fontsize=8, color=color)

    plt.suptitle(f'Actual Input {mode} Partial Correlations & R²\n{param1_name} vs {param2_name}',
                 fontsize=14)
    plt.tight_layout()

    partial_path = fig_path / f'{prefix}_partial_heatmaps.png'
    plt.savefig(partial_path, dpi=150, bbox_inches='tight')
    print(f"Saved partial correlation heatmap: {partial_path}")

    # Individual heatmaps
    all_metrics = metrics + partial_metrics
    for key, title, cmap in all_metrics:
        if 'r2' in key:
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
        if 'r2' in key:
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
                    if 'r2' in key:
                        color = 'white' if val > v_max * 0.5 else 'black'
                    else:
                        color = 'white' if abs(val) > v_max * 0.6 else 'black'
                    ax.text(j, i, text, ha='center', va='center',
                            fontsize=8, color=color)

        plt.tight_layout()
        individual_path = fig_path / f'{prefix}_{key}_heatmap.png'
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
        description='Generate heatmaps of actual input covariance vs noise correlation relationships'
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
                        help='Time bin size in ms for output noise correlations (default: 1000)')
    parser.add_argument('--total_time', type=float, default=None,
                        help='Total simulation time in ms (default: read from params)')
    parser.add_argument('--start_from', type=float, default=50,
                        help='Discard spikes before this time in ms (default: 50)')
    parser.add_argument('--single_stimulus', action='store_true',
                        help='Use only stim0 (default: combine across all stimuli)')
    parser.add_argument('--use_covariance', action='store_true',
                        help='Use raw covariance instead of correlation for input pairwise metrics')
    parser.add_argument('--use_currents', action='store_true',
                        help='Use recorded currents instead of W @ spike_counts')

    args = parser.parse_args()

    stim_mode = "single stimulus (stim0)" if args.single_stimulus else "all stimuli combined"
    input_mode = "raw COVARIANCE" if args.use_covariance else "CORRELATION (normalized)"
    input_method = "RECORDED CURRENTS" if args.use_currents else "W @ spike_counts"
    print(f"Analyzing sweep: {args.sweep_dir}")
    print(f"Settings: n_neurons={args.n_neurons}, seed={args.seed}, bin_size={args.bin_size}ms")
    print(f"Stimulus mode: {stim_mode}")
    print(f"Input metric: {input_mode}")
    print(f"Input method: {input_method}")
    print()

    results = analyze_sweep_actual(
        args.sweep_dir,
        n_neurons_sample=args.n_neurons,
        seed=args.seed,
        verbose=True,
        bin_size=args.bin_size,
        total_time=args.total_time,
        start_from=args.start_from,
        single_stimulus=args.single_stimulus,
        use_covariance=args.use_covariance,
        use_currents=args.use_currents
    )

    n_completed = results['completed'].sum()
    n_total = results['completed'].size
    print(f"\nCompleted {n_completed}/{n_total} parameter points")

    if n_completed > 0:
        print("\nGenerating heatmaps...")
        plot_heatmaps(results, args.fig_dir, use_covariance=args.use_covariance)

        print("\n" + "="*70)
        print("SUMMARY: Mean values across completed parameter points")
        print("="*70)
        print("Marginal correlations (same-type):")
        for key in ['r_feedforward', 'r_excitatory', 'r_inhibitory']:
            data = results[key]
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                print(f"  {key:30s}: mean={np.mean(valid):+.3f}, "
                      f"range=[{np.min(valid):+.3f}, {np.max(valid):+.3f}]")
        if 'r_total' in results:
            print("Marginal correlation (total input):")
            data = results['r_total']
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                print(f"  {'r_total':30s}: mean={np.mean(valid):+.3f}, "
                      f"range=[{np.min(valid):+.3f}, {np.max(valid):+.3f}]")
        print("Marginal correlations (cross-type):")
        for key in ['r_exc_ff_cross', 'r_inh_ff_cross']:
            data = results[key]
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                print(f"  {key:30s}: mean={np.mean(valid):+.3f}, "
                      f"range=[{np.min(valid):+.3f}, {np.max(valid):+.3f}]")
        print("-"*70)
        print("Partial correlations (each controlling for all others):")
        for key in ['r_feedforward_partial', 'r_excitatory_partial',
                     'r_inhibitory_partial', 'r_exc_ff_cross_partial',
                     'r_inh_ff_cross_partial']:
            data = results[key]
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                print(f"  {key:30s}: mean={np.mean(valid):+.3f}, "
                      f"range=[{np.min(valid):+.3f}, {np.max(valid):+.3f}]")
        print("-"*70)
        print("Model R²:")
        r2_items = [('r2_same_only', 'Same-type only (FF+Exc+Inh)'),
                    ('r2_full', 'Full (+cross terms)')]
        if 'r2_total' in results:
            r2_items.append(('r2_total', 'Total input only'))
        for key, label in r2_items:
            data = results[key]
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                print(f"  {label:30s}: mean={np.mean(valid):.3f}, "
                      f"range=[{np.min(valid):.3f}, {np.max(valid):.3f}]")
        print("="*70)
        print("\nDone!")
    else:
        print("\nNo completed runs found - skipping heatmap generation")


if __name__ == '__main__':
    main()
