"""
Linear decoding of stimulus identity from spiking activity.

Uses a linear SVM with cross-validated regularization to classify stimuli,
and plots decoding accuracy as a function of number of neurons.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from analysis import load_spikes_vectorized, load_params


def load_responses(directory: str, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Load spike count responses for all neurons across all stimuli/trials.

    Returns:
        X: (n_samples, n_neurons) array of spike counts
        y: (n_samples,) array of stimulus labels
    """
    n_neurons = params['n_excite'] + params['n_inhib']
    n_stimuli = params['n_stimuli']
    n_trials = params['n_trials']

    # Shape: (n_neurons, n_stimuli, n_trials)
    all_responses = np.zeros((n_neurons, n_stimuli, n_trials))

    for stim in range(n_stimuli):
        stim_dir = f"{directory}/stim{stim}"
        spikes = load_spikes_vectorized(
            stim_dir, n_trials, n_neurons,
            bin_size=1000, total_time=1000, start_from=50
        )
        # spikes is (n_neurons, n_trials) since bin_size = total_time
        all_responses[:, stim, :] = spikes

    # Reshape to (n_samples, n_neurons)
    # X[stim * n_trials + trial, :] = response to stimulus 'stim' on trial 'trial'
    X = all_responses.reshape(n_neurons, -1).T  # (n_stimuli * n_trials, n_neurons)

    # Create labels
    y = np.repeat(np.arange(n_stimuli), n_trials)

    return X, y


def decode_with_n_neurons(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          n_neurons: int, neuron_indices: np.ndarray,
                          C_values: np.ndarray, cv_folds: int = 5,
                          random_state: int = 42) -> dict:
    """
    Train and evaluate SVM decoder using a subset of neurons.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_neurons: Number of neurons to use
        neuron_indices: Which neurons to select (should have length >= n_neurons)
        C_values: Regularization values to try
        cv_folds: Number of cross-validation folds
        random_state: Random seed

    Returns:
        Dict with best_C, cv_accuracy, test_accuracy
    """
    # Select neurons
    selected = neuron_indices[:n_neurons]
    X_train_sub = X_train[:, selected]
    X_test_sub = X_test[:, selected]

    # Pipeline with scaling and SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(dual='auto', max_iter=5000, random_state=random_state))
    ])

    # Grid search for best C
    param_grid = {'svm__C': C_values}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train_sub, y_train)

    best_C = grid_search.best_params_['svm__C']
    cv_accuracy = grid_search.best_score_

    # Evaluate on test set
    test_accuracy = grid_search.score(X_test_sub, y_test)

    return {
        'best_C': best_C,
        'cv_accuracy': cv_accuracy,
        'test_accuracy': test_accuracy
    }


def main():
    # Configuration
    data_dir = "outputs/full_sweep/kappa_0/input_time_200"
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    seed = 42
    np.random.seed(seed)

    # Neuron counts to test
    neuron_counts = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

    # Regularization values to try (log-spaced)
    C_values = np.logspace(-4, 2, 13)

    # Cross-validation settings
    cv_folds = 5
    test_size = 0.2

    print("Loading data...")
    params = load_params(data_dir)
    X, y = load_responses(data_dir, params)

    n_samples, n_neurons_total = X.shape
    n_stimuli = params['n_stimuli']
    n_trials = params['n_trials']

    print(f"Data shape: {X.shape}")
    print(f"  {n_stimuli} stimuli, {n_trials} trials each")
    print(f"  {n_neurons_total} neurons total")
    print(f"  Chance accuracy: {1/n_stimuli:.1%}")

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    print(f"\nTrain set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")

    # Random ordering of neurons (will use first N for each condition)
    neuron_order = np.random.permutation(n_neurons_total)

    # Filter neuron counts to those <= total neurons
    neuron_counts = [n for n in neuron_counts if n <= n_neurons_total]

    print(f"\nTesting {len(neuron_counts)} neuron counts: {neuron_counts}")
    print(f"C values to search: {C_values}")
    print()

    # Run decoding for each neuron count
    results = []
    for n_neurons in neuron_counts:
        print(f"Decoding with {n_neurons} neurons...", end=" ", flush=True)

        result = decode_with_n_neurons(
            X_train, y_train, X_test, y_test,
            n_neurons, neuron_order, C_values,
            cv_folds=cv_folds, random_state=seed
        )
        result['n_neurons'] = n_neurons
        results.append(result)

        print(f"CV acc: {result['cv_accuracy']:.1%}, "
              f"Test acc: {result['test_accuracy']:.1%}, "
              f"best C: {result['best_C']:.2e}")

    # Extract arrays for plotting
    n_neurons_arr = np.array([r['n_neurons'] for r in results])
    cv_acc = np.array([r['cv_accuracy'] for r in results])
    test_acc = np.array([r['test_accuracy'] for r in results])
    best_C = np.array([r['best_C'] for r in results])

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy vs number of neurons
    ax = axes[0]
    ax.plot(n_neurons_arr, cv_acc * 100, 'o-', label='CV accuracy', markersize=8)
    ax.plot(n_neurons_arr, test_acc * 100, 's-', label='Test accuracy', markersize=8)
    ax.axhline(100 / n_stimuli, color='gray', linestyle='--', label='Chance')
    ax.set_xscale('log')
    ax.set_xlabel('Number of neurons')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Stimulus Decoding Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    # Best C vs number of neurons
    ax = axes[1]
    ax.plot(n_neurons_arr, best_C, 'o-', color='green', markersize=8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of neurons')
    ax.set_ylabel('Best C (regularization)')
    ax.set_title('Optimal Regularization Strength')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Linear SVM Decoding of {n_stimuli} Stimuli\n'
                 f'(kappa=0, input_time=200, {n_trials} trials/stimulus)',
                 fontsize=12)
    plt.tight_layout()

    fig_path = fig_dir / 'decoding_vs_neurons.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")

    # Print summary table
    print("\n" + "="*70)
    print("DECODING RESULTS SUMMARY")
    print("="*70)
    print(f"{'N neurons':>10} {'CV Acc':>10} {'Test Acc':>10} {'Best C':>12}")
    print("-"*70)
    for r in results:
        print(f"{r['n_neurons']:>10} {r['cv_accuracy']:>10.1%} "
              f"{r['test_accuracy']:>10.1%} {r['best_C']:>12.2e}")
    print("="*70)

    # Best result
    best_idx = np.argmax(test_acc)
    print(f"\nBest test accuracy: {test_acc[best_idx]:.1%} with {n_neurons_arr[best_idx]} neurons")

    plt.show()


if __name__ == '__main__':
    main()
