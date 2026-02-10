"""
Compute linear and quadratic d-prime measures for stimulus discriminability.

Linear d-prime quantifies how well stimuli can be distinguished based on mean
firing rate differences (what a linear classifier exploits).

Quadratic d-prime quantifies how well stimuli can be distinguished based on
covariance matrix differences (what QDA could exploit beyond linear).

This helps explain why QDA may or may not improve over linear classifiers.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from analysis import load_spikes_vectorized, load_params, get_n_stimuli
from corr_analyses import calculate_quadratic_d_prime


def load_responses_3d(directory: str, params: dict, n_stimuli: int = None,
                      duration: float = None, start_from: float = 50) -> np.ndarray:
    """
    Load spike count responses in (N_neurons, S_stimuli, T_trials) format.

    Args:
        directory: Path to simulation output directory
        params: Simulation parameters dict
        n_stimuli: Number of stimuli to load. If None, auto-detect from params/directory.
        duration: Duration in ms to use for spike counting (after start_from).
                  If None, uses full simulation time.
        start_from: Time in ms to start counting spikes (default: 50ms to skip transient)

    Returns:
        data: (n_neurons, n_stimuli, n_trials) array of spike counts
    """
    n_neurons = params['n_excite'] + params['n_inhib']
    n_trials = params['n_trials']
    if n_stimuli is None:
        n_stimuli = get_n_stimuli(params, directory)

    # Determine total_time based on duration argument
    if duration is None:
        total_time = 1000  # Default: use full run
    else:
        total_time = start_from + duration

    # bin_size = effective duration so we get one bin (spike count) per trial
    bin_size = total_time - start_from

    # Shape: (n_neurons, n_stimuli, n_trials)
    all_responses = np.zeros((n_neurons, n_stimuli, n_trials))

    for stim in range(n_stimuli):
        stim_dir = f"{directory}/stim{stim}"
        spikes = load_spikes_vectorized(
            stim_dir, n_trials, n_neurons,
            bin_size=bin_size, total_time=total_time, start_from=start_from
        )
        # spikes is (n_neurons, n_trials) since bin_size covers full duration
        all_responses[:, stim, :] = spikes

    return all_responses


def calculate_linear_d_prime(data: np.ndarray, shrinkage: float = 0.0) -> np.ndarray:
    """
    Calculate pairwise linear d-prime (Fisher's discriminant) between conditions.

    d'_linear = sqrt((μ1 - μ2)^T Σ_pooled^{-1} (μ1 - μ2))

    Parameters:
    -----------
    data : np.ndarray
        Shape (N, S, T) - N neurons, S stimuli, T trials
    shrinkage : float
        Shrinkage parameter for covariance regularization (0-1)

    Returns:
    --------
    d_prime_matrix : np.ndarray
        Shape (S, S) symmetric matrix of pairwise linear d-primes
    """
    N, S, T = data.shape

    # Compute per-class means: shape (N, S)
    means = np.mean(data, axis=2)

    # Compute per-class covariances and pool them
    covs = []
    for s in range(S):
        cov = np.cov(data[:, s, :], rowvar=True)  # (N, N)
        covs.append(cov)

    # Pooled covariance (average of per-class covariances)
    pooled_cov = np.mean(covs, axis=0)

    # Apply shrinkage
    if shrinkage > 0:
        avg_var = np.trace(pooled_cov) / N
        target = np.eye(N) * avg_var
        pooled_cov = (1 - shrinkage) * pooled_cov + shrinkage * target

    # Compute precision matrix
    if shrinkage > 0:
        pooled_prec = np.linalg.inv(pooled_cov)
    else:
        pooled_prec = np.linalg.pinv(pooled_cov)

    # Compute pairwise d-primes
    d_prime_matrix = np.zeros((S, S))

    for i in range(S):
        for j in range(i + 1, S):
            delta_mu = means[:, i] - means[:, j]  # (N,)

            # Mahalanobis distance squared
            d_squared = delta_mu @ pooled_prec @ delta_mu

            # d-prime is sqrt of this
            d_val = np.sqrt(max(0, d_squared))

            d_prime_matrix[i, j] = d_val
            d_prime_matrix[j, i] = d_val

    return d_prime_matrix


def calculate_quadratic_d_prime_centered(data: np.ndarray, shrinkage: float = 0.0) -> np.ndarray:
    """
    Calculate quadratic d-prime on per-class centered data.

    The quadratic d-prime function expects data to be centered per-class
    since it's measuring covariance differences, not mean differences.

    Parameters:
    -----------
    data : np.ndarray
        Shape (N, S, T) - N neurons, S stimuli, T trials
    shrinkage : float
        Shrinkage parameter for covariance regularization (0-1)

    Returns:
    --------
    d_prime_matrix : np.ndarray
        Shape (S, S) symmetric matrix of pairwise quadratic d-primes
    """
    N, S, T = data.shape

    # Center data per-class (subtract per-stimulus mean)
    centered_data = data.copy()
    for s in range(S):
        class_mean = np.mean(data[:, s, :], axis=1, keepdims=True)
        centered_data[:, s, :] = data[:, s, :] - class_mean

    return calculate_quadratic_d_prime(centered_data, shrinkage=shrinkage)


def compute_d_primes_for_neurons(data: np.ndarray, n_neurons: int,
                                  neuron_indices: np.ndarray,
                                  shrinkage: float = 0.0) -> dict:
    """
    Compute both d-prime measures for a subset of neurons.

    Parameters:
    -----------
    data : np.ndarray
        Full data, shape (N_total, S, T)
    n_neurons : int
        Number of neurons to use
    neuron_indices : np.ndarray
        Which neurons to select (length >= n_neurons)
    shrinkage : float
        Shrinkage for covariance regularization

    Returns:
    --------
    dict with linear_d_prime and quadratic_d_prime matrices
    """
    selected = neuron_indices[:n_neurons]
    data_subset = data[selected, :, :]

    linear_dp = calculate_linear_d_prime(data_subset, shrinkage=shrinkage)
    quadratic_dp = calculate_quadratic_d_prime_centered(data_subset, shrinkage=shrinkage)

    return {
        'linear': linear_dp,
        'quadratic': quadratic_dp
    }


def summarize_d_prime_matrix(dp_matrix: np.ndarray) -> dict:
    """Extract summary statistics from a d-prime matrix."""
    # Get upper triangle values (excluding diagonal)
    S = dp_matrix.shape[0]
    upper_tri = dp_matrix[np.triu_indices(S, k=1)]

    return {
        'mean': np.mean(upper_tri),
        'std': np.std(upper_tri),
        'min': np.min(upper_tri),
        'max': np.max(upper_tri)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compute linear and quadratic d-prime measures'
    )
    parser.add_argument('--data_dir', type=str,
                        default='outputs/longrun/input_time_300/',
                        help='Directory containing simulation output')
    parser.add_argument('--shrinkage', type=float, default=0.0,
                        help='Shrinkage parameter for covariance regularization (0-1)')
    parser.add_argument('--n_stimuli', type=int, default=None,
                        help='Number of stimuli to analyze (auto-detect if not specified)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for neuron ordering')
    parser.add_argument('--save_fig', action='store_true',
                        help='Save figure to figures/ directory')
    parser.add_argument('--duration', type=float, default=None,
                        help='Duration in ms to use for spike counting (after 50ms warmup). '
                             'Default: use full run (~950ms)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Neuron counts to test
    neuron_counts = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]

    print("Loading data...")
    params = load_params(args.data_dir)
    data = load_responses_3d(args.data_dir, params, n_stimuli=args.n_stimuli,
                             duration=args.duration)

    n_neurons_total, n_stimuli, n_trials = data.shape
    duration_str = f"{args.duration}ms" if args.duration else "full run (~950ms)"
    print(f"Data shape: {data.shape}")
    print(f"  {n_neurons_total} neurons")
    print(f"  {n_stimuli} stimuli")
    print(f"  {n_trials} trials per stimulus")
    print(f"  Duration: {duration_str}")
    print(f"  Shrinkage: {args.shrinkage}")

    # Random ordering of neurons
    neuron_order = np.random.permutation(n_neurons_total)

    # Filter neuron counts
    neuron_counts = [n for n in neuron_counts if n <= n_neurons_total]

    print(f"\nTesting {len(neuron_counts)} neuron counts: {neuron_counts}")
    print()

    # Storage for results
    results = []

    for n_neurons in neuron_counts:
        print(f"Computing d-primes with {n_neurons} neurons...", end=" ", flush=True)

        dp_result = compute_d_primes_for_neurons(
            data, n_neurons, neuron_order, shrinkage=args.shrinkage
        )

        linear_summary = summarize_d_prime_matrix(dp_result['linear'])
        quad_summary = summarize_d_prime_matrix(dp_result['quadratic'])

        results.append({
            'n_neurons': n_neurons,
            'linear_matrix': dp_result['linear'],
            'quadratic_matrix': dp_result['quadratic'],
            'linear_mean': linear_summary['mean'],
            'quadratic_mean': quad_summary['mean'],
            'linear_std': linear_summary['std'],
            'quadratic_std': quad_summary['std']
        })

        ratio = quad_summary['mean'] / linear_summary['mean'] if linear_summary['mean'] > 0 else 0
        print(f"Linear: {linear_summary['mean']:.3f}, "
              f"Quadratic: {quad_summary['mean']:.3f}, "
              f"Ratio: {ratio:.3f}")

    # Print detailed results for largest neuron count
    print("\n" + "=" * 80)
    print(f"DETAILED RESULTS (N = {results[-1]['n_neurons']} neurons)")
    print("=" * 80)

    print("\nLinear d-prime matrix (mean differences):")
    print(np.array2string(results[-1]['linear_matrix'], precision=3, suppress_small=True))

    print("\nQuadratic d-prime matrix (covariance differences):")
    print(np.array2string(results[-1]['quadratic_matrix'], precision=3, suppress_small=True))

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: d-prime vs Number of Neurons")
    print("=" * 80)
    print(f"{'N neurons':>10} {'Linear d':>12} {'Quad d':>12} {'Ratio Q/L':>12}")
    print("-" * 80)
    for r in results:
        ratio = r['quadratic_mean'] / r['linear_mean'] if r['linear_mean'] > 0 else 0
        print(f"{r['n_neurons']:>10} {r['linear_mean']:>12.3f} "
              f"{r['quadratic_mean']:>12.3f} {ratio:>12.3f}")
    print("=" * 80)

    # Interpretation
    final_ratio = results[-1]['quadratic_mean'] / results[-1]['linear_mean'] \
        if results[-1]['linear_mean'] > 0 else 0

    print("\nINTERPRETATION:")
    if final_ratio < 0.1:
        print("  Quadratic d' << Linear d': Covariance matrices are nearly identical.")
        print("  QDA cannot improve over linear classifiers because there's no")
        print("  stimulus-dependent covariance structure to exploit.")
    elif final_ratio < 0.5:
        print("  Quadratic d' < Linear d': Covariance differences exist but are weak.")
        print("  QDA may show marginal improvement, but linear methods dominate.")
    elif final_ratio < 1.0:
        print("  Quadratic d' ~ Linear d': Both mean and covariance carry information.")
        print("  QDA should show meaningful improvement over linear methods.")
    else:
        print("  Quadratic d' >= Linear d': Strong covariance differences exist.")
        print("  QDA should substantially outperform linear classifiers.")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    n_neurons_arr = np.array([r['n_neurons'] for r in results])
    linear_means = np.array([r['linear_mean'] for r in results])
    quad_means = np.array([r['quadratic_mean'] for r in results])
    ratios = quad_means / np.where(linear_means > 0, linear_means, 1)

    # Plot 1: d-primes vs neurons
    ax = axes[0]
    ax.plot(n_neurons_arr, linear_means, 'o-', label="Linear d'", markersize=6)
    ax.plot(n_neurons_arr, quad_means, 's-', label="Quadratic d'", markersize=6)
    ax.set_xscale('log')
    ax.set_xlabel('Number of neurons')
    ax.set_ylabel("d-prime (mean across pairs)")
    ax.set_title("d-prime vs Population Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Ratio vs neurons
    ax = axes[1]
    ax.plot(n_neurons_arr, ratios, 'o-', color='purple', markersize=6)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Number of neurons')
    ax.set_ylabel("Ratio (Quadratic / Linear)")
    ax.set_title("Relative Covariance Discriminability")
    ax.grid(True, alpha=0.3)

    # Plot 3: Heatmaps of d-prime matrices (largest N)
    ax = axes[2]
    # Show linear and quadratic side by side
    combined = np.zeros((n_stimuli, n_stimuli * 2 + 1))
    combined[:, :n_stimuli] = results[-1]['linear_matrix']
    combined[:, n_stimuli + 1:] = results[-1]['quadratic_matrix']
    combined[:, n_stimuli] = np.nan  # separator

    im = ax.imshow(combined, cmap='viridis', aspect='auto')
    ax.set_title(f"d-prime matrices (N={results[-1]['n_neurons']})\nLeft: Linear, Right: Quadratic")
    ax.set_xticks([n_stimuli // 2, n_stimuli + 1 + n_stimuli // 2])
    ax.set_xticklabels(['Linear', 'Quadratic'])
    ax.set_ylabel('Stimulus')
    plt.colorbar(im, ax=ax, label="d'")

    plt.tight_layout()

    if args.save_fig:
        fig_dir = Path("figures")
        fig_dir.mkdir(exist_ok=True)
        fig_path = fig_dir / 'd_prime_comparison.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {fig_path}")

    plt.show()


if __name__ == '__main__':
    main()
