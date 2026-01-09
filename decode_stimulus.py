"""
Linear and quadratic decoding of stimulus identity from spiking activity.

Uses a linear SVM and QDA (Quadratic Discriminant Analysis) with cross-validated
regularization to classify stimuli, comparing decoding accuracy as a function
of number of neurons. QDA accounts for stimulus-specific covariance matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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
    #n_stimuli = params['n_stimuli']
    n_stimuli = 4
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


def decode_with_n_neurons_qda(X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               n_neurons: int, neuron_indices: np.ndarray,
                               shrinkage_values: np.ndarray, cv_folds: int = 5,
                               random_state: int = 42) -> dict:
    """
    Train and evaluate QDA decoder using a subset of neurons.

    QDA uses per-class covariance matrices, capturing stimulus-dependent
    covariance structure in the neural responses.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_neurons: Number of neurons to use
        neuron_indices: Which neurons to select (should have length >= n_neurons)
        shrinkage_values: Shrinkage parameter values to try (0-1)
        cv_folds: Number of cross-validation folds
        random_state: Random seed

    Returns:
        Dict with best_shrinkage, cv_accuracy, test_accuracy
    """
    # Select neurons
    selected = neuron_indices[:n_neurons]
    X_train_sub = X_train[:, selected]
    X_test_sub = X_test[:, selected]

    # Scale data (QDA benefits from standardized features)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sub)
    X_test_scaled = scaler.transform(X_test_sub)

    # Manual cross-validation over shrinkage
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    best_shrinkage = None
    best_cv_accuracy = -1

    for shrinkage in shrinkage_values:
        fold_accuracies = []
        for train_idx, val_idx in cv.split(X_train_scaled, y_train):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            qda = QuadraticDiscriminantAnalysis(solver='eigen', shrinkage=shrinkage)
            qda.fit(X_tr, y_tr)
            fold_accuracies.append(qda.score(X_val, y_val))

        mean_cv_acc = np.mean(fold_accuracies)
        if mean_cv_acc > best_cv_accuracy:
            best_cv_accuracy = mean_cv_acc
            best_shrinkage = shrinkage

    # Train final model with best shrinkage on full training set
    final_qda = QuadraticDiscriminantAnalysis(solver='eigen', shrinkage=best_shrinkage)
    final_qda.fit(X_train_scaled, y_train)
    test_accuracy = final_qda.score(X_test_scaled, y_test)

    return {
        'best_shrinkage': best_shrinkage,
        'cv_accuracy': best_cv_accuracy,
        'test_accuracy': test_accuracy
    }


def main():
    # Configuration
    data_dir = "outputs/longrun/input_time_300/"
    #data_dir = "outputs/input_sweep/input_rate_kappa_0/input_conn_kappa_0.2"
    #data_dir = "outputs/full_sweep/kappa_0.6/input_time_600"
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    seed = 42
    np.random.seed(seed)

    # Neuron counts to test
    neuron_counts = [5, 10, 20, 50, 100, 200, 500, 1000]#, 2000, 5000]

    # Regularization values to try (log-spaced for SVM)
    C_values = np.logspace(-4, 2, 13)

    # Shrinkage values to try for QDA (linear-spaced, 0-1)
    shrinkage_values = np.linspace(0, 0.6, 11)

    # Cross-validation settings
    cv_folds = 5
    test_size = 0.1

    print("Loading data...")
    params = load_params(data_dir)
    X, y = load_responses(data_dir, params)

    n_samples, n_neurons_total = X.shape
    #n_stimuli = params['n_stimuli']
    n_stimuli = 4
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
    print(f"SVM C values: {C_values}")
    print(f"QDA shrinkage values: {shrinkage_values}")
    print()

    # Run decoding for each neuron count
    svm_results = []
    qda_results = []
    for n_neurons in neuron_counts:
        print(f"Decoding with {n_neurons} neurons...")

        # Linear SVM
        print(f"  SVM: ", end="", flush=True)
        svm_result = decode_with_n_neurons(
            X_train, y_train, X_test, y_test,
            n_neurons, neuron_order, C_values,
            cv_folds=cv_folds, random_state=seed
        )
        svm_result['n_neurons'] = n_neurons
        svm_results.append(svm_result)
        print(f"CV acc: {svm_result['cv_accuracy']:.1%}, "
              f"Test acc: {svm_result['test_accuracy']:.1%}")

        # QDA
        print(f"  QDA: ", end="", flush=True)
        qda_result = decode_with_n_neurons_qda(
            X_train, y_train, X_test, y_test,
            n_neurons, neuron_order, shrinkage_values,
            cv_folds=cv_folds, random_state=seed
        )
        qda_result['n_neurons'] = n_neurons
        qda_results.append(qda_result)
        print(f"CV acc: {qda_result['cv_accuracy']:.1%}, "
              f"Test acc: {qda_result['test_accuracy']:.1%}")

    # Extract arrays for plotting
    n_neurons_arr = np.array([r['n_neurons'] for r in svm_results])
    svm_test_acc = np.array([r['test_accuracy'] for r in svm_results])
    qda_test_acc = np.array([r['test_accuracy'] for r in qda_results])

    # Plot results - single plot comparing both methods
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(n_neurons_arr, svm_test_acc * 100, 'o-', label='Linear SVM', markersize=8)
    ax.plot(n_neurons_arr, qda_test_acc * 100, 's-', label='QDA', markersize=8)
    ax.axhline(100 / n_stimuli, color='gray', linestyle='--', label='Chance')
    ax.set_xscale('log')
    ax.set_xlabel('Number of neurons')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'Stimulus Decoding: Linear SVM vs QDA\n'
                 f'({n_stimuli} stimuli, {n_trials} trials/stimulus)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()

    fig_path = fig_dir / 'decoding_vs_neurons.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")

    # Print summary table
    print("\n" + "="*80)
    print("DECODING RESULTS SUMMARY")
    print("="*80)
    print(f"{'N neurons':>10} {'SVM Test':>12} {'QDA Test':>12} {'Difference':>12}")
    print("-"*80)
    for svm_r, qda_r in zip(svm_results, qda_results):
        diff = qda_r['test_accuracy'] - svm_r['test_accuracy']
        print(f"{svm_r['n_neurons']:>10} {svm_r['test_accuracy']:>12.1%} "
              f"{qda_r['test_accuracy']:>12.1%} {diff:>+12.1%}")
    print("="*80)

    # Best results
    svm_best_idx = np.argmax(svm_test_acc)
    qda_best_idx = np.argmax(qda_test_acc)
    print(f"\nBest SVM: {svm_test_acc[svm_best_idx]:.1%} with {n_neurons_arr[svm_best_idx]} neurons")
    print(f"Best QDA: {qda_test_acc[qda_best_idx]:.1%} with {n_neurons_arr[qda_best_idx]} neurons")

    plt.show()


if __name__ == '__main__':
    main()
