"""
Linear and quadratic decoding of stimulus identity from spiking activity.

Uses a linear SVM and QDA (Quadratic Discriminant Analysis) with cross-validated
regularization to classify stimuli, comparing decoding accuracy as a function
of number of neurons (default) or principal components (--use_pca flag).

QDA accounts for stimulus-specific covariance matrices, which may be more
apparent in PC space where estimation noise is reduced.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from analysis import load_spikes_vectorized, load_params, get_n_stimuli


def load_responses(directory: str, params: dict, n_stimuli: int = None,
                   duration: float = None, start_from: float = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Load spike count responses for all neurons across all stimuli/trials.

    Args:
        directory: Path to simulation output directory
        params: Simulation parameters dict
        n_stimuli: Number of stimuli to load. If None, auto-detect from params/directory.
        duration: Duration in ms to use for spike counting (after start_from).
                  If None, uses full simulation time.
        start_from: Time in ms to start counting spikes (default: 50ms to skip transient)

    Returns:
        X: (n_samples, n_neurons) array of spike counts
        y: (n_samples,) array of stimulus labels
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

    # Reshape to (n_samples, n_neurons)
    # X[stim * n_trials + trial, :] = response to stimulus 'stim' on trial 'trial'
    X = all_responses.reshape(n_neurons, -1).T  # (n_stimuli * n_trials, n_neurons)

    # Create labels
    y = np.repeat(np.arange(n_stimuli), n_trials)

    return X, y


def decode_with_n_features(X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           n_features: int, feature_indices: np.ndarray,
                           C_values: np.ndarray, cv_folds: int = 5,
                           random_state: int = 42) -> dict:
    """
    Train and evaluate SVM decoder using a subset of features (neurons or PCs).

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_features: Number of features to use
        feature_indices: Which features to select (should have length >= n_features)
        C_values: Regularization values to try
        cv_folds: Number of cross-validation folds
        random_state: Random seed

    Returns:
        Dict with best_C, cv_accuracy, test_accuracy
    """
    # Select features
    selected = feature_indices[:n_features]
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


def decode_with_n_features_qda(X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                n_features: int, feature_indices: np.ndarray,
                                shrinkage_values: np.ndarray, cv_folds: int = 5,
                                random_state: int = 42,
                                covariance_only: bool = False) -> dict:
    """
    Train and evaluate QDA decoder using a subset of features (neurons or PCs).

    QDA uses per-class covariance matrices, capturing stimulus-dependent
    covariance structure in the neural responses.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_features: Number of features to use
        feature_indices: Which features to select (should have length >= n_features)
        shrinkage_values: Shrinkage parameter values to try (0-1)
        cv_folds: Number of cross-validation folds
        random_state: Random seed
        covariance_only: If True, ignore class mean differences and classify
                         based only on covariance structure

    Returns:
        Dict with best_shrinkage, cv_accuracy, test_accuracy
    """
    # Select features
    selected = feature_indices[:n_features]
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
            if covariance_only:
                # Set all class means to zero (global mean of scaled data)
                # This forces classification to use only covariance differences
                qda.means_ = np.zeros_like(qda.means_)
            fold_accuracies.append(qda.score(X_val, y_val))

        mean_cv_acc = np.mean(fold_accuracies)
        if mean_cv_acc > best_cv_accuracy:
            best_cv_accuracy = mean_cv_acc
            best_shrinkage = shrinkage

    # Train final model with best shrinkage on full training set
    final_qda = QuadraticDiscriminantAnalysis(solver='eigen', shrinkage=best_shrinkage)
    final_qda.fit(X_train_scaled, y_train)
    if covariance_only:
        # Set all class means to zero (global mean of scaled data)
        # This forces classification to use only covariance differences
        final_qda.means_ = np.zeros_like(final_qda.means_)
    test_accuracy = final_qda.score(X_test_scaled, y_test)

    return {
        'best_shrinkage': best_shrinkage,
        'cv_accuracy': best_cv_accuracy,
        'test_accuracy': test_accuracy
    }


def main():
    parser = argparse.ArgumentParser(
        description='Decode stimulus identity using linear SVM and QDA'
    )
    parser.add_argument('--data_dir', type=str,
                        default='outputs/longrun/input_time_300/',
                        help='Directory containing simulation output')
    parser.add_argument('--use_pca', action='store_true',
                        help='Sweep over PCs instead of neurons')
    parser.add_argument('--n_stimuli', type=int, default=None,
                        help='Number of stimuli to analyze (auto-detect if not specified)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_fig', action='store_true',
                        help='Save figure to figures/ directory')
    parser.add_argument('--diagnostics', action='store_true',
                        help='Run diagnostic checks for data leakage')
    parser.add_argument('--duration', type=float, default=None,
                        help='Duration in ms to use for spike counting (after 50ms warmup). '
                             'Default: use full run (~950ms)')
    parser.add_argument('--covariance_only', action='store_true',
                        help='QDA uses only covariance differences, ignoring class mean differences')
    args = parser.parse_args()

    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    np.random.seed(args.seed)

    # Feature counts to test (will be filtered based on available features)
    feature_counts = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]

    # Regularization values to try (log-spaced for SVM)
    C_values = np.logspace(-4, 2, 13)

    # Shrinkage values to try for QDA (linear-spaced, 0-1)
    shrinkage_values = np.linspace(0.1, 1.0, 10)

    # Cross-validation settings
    cv_folds = 5
    test_size = 0.1

    print("Loading data...")
    params = load_params(args.data_dir)
    X, y = load_responses(args.data_dir, params, n_stimuli=args.n_stimuli,
                          duration=args.duration)

    n_samples, n_neurons_total = X.shape
    n_stimuli = args.n_stimuli if args.n_stimuli is not None else get_n_stimuli(params, args.data_dir)
    n_trials = params['n_trials']

    duration_str = f"{args.duration}ms" if args.duration else "full run (~950ms)"
    print(f"Data shape: {X.shape}")
    print(f"  {n_stimuli} stimuli, {n_trials} trials each")
    print(f"  {n_neurons_total} neurons total")
    print(f"  Duration: {duration_str}")
    print(f"  Chance accuracy: {1/n_stimuli:.1%}")

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=args.seed
    )
    print(f"\nTrain set: {len(y_train)} samples")
    print(f"Test set: {len(y_test)} samples")

    # Diagnostic checks
    if args.diagnostics:
        print("\n" + "="*60)
        print("DIAGNOSTIC CHECKS")
        print("="*60)

        # Check class distributions
        print("\nClass distributions:")
        for label in np.unique(y_train):
            n_train = np.sum(y_train == label)
            n_test = np.sum(y_test == label)
            print(f"  Class {label}: {n_train} train, {n_test} test")

        # Check for duplicate samples between train and test
        # (This would indicate a serious bug)
        print("\nChecking for train/test overlap...")
        n_overlap = 0
        for i in range(len(X_test)):
            for j in range(len(X_train)):
                if np.allclose(X_test[i], X_train[j]):
                    n_overlap += 1
                    break
        print(f"  Samples in test that match train: {n_overlap}/{len(X_test)}")
        if n_overlap > 0:
            print("  WARNING: Data leakage detected!")

        # Shuffled label sanity check
        print("\nShuffled label sanity check (should be near chance)...")
        y_train_shuffled = np.random.permutation(y_train)
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[:, :100])  # Use first 100 neurons
        X_test_scaled = scaler.transform(X_test[:, :100])
        clf = LinearSVC(dual='auto', max_iter=5000, C=1.0)
        clf.fit(X_train_scaled, y_train_shuffled)
        shuffled_acc = clf.score(X_test_scaled, y_test)
        print(f"  Accuracy with shuffled labels: {shuffled_acc:.1%} (chance={1/n_stimuli:.1%})")

        # Real label check
        clf_real = LinearSVC(dual='auto', max_iter=5000, C=1.0)
        clf_real.fit(X_train_scaled, y_train)
        real_acc = clf_real.score(X_test_scaled, y_test)
        print(f"  Accuracy with real labels: {real_acc:.1%}")

        print("="*60 + "\n")

    # PCA mode: fit on training data, transform both
    if args.use_pca:
        print("\nPCA MODE: Sweeping over principal components")

        # Determine max PCs we can extract (limited by min of n_samples, n_features)
        max_components = min(X_train.shape[0], X_train.shape[1])
        # Also limit to max we'll actually test
        max_needed = max(f for f in feature_counts if f <= max_components)
        n_components = min(max_components, max_needed)

        print(f"Fitting PCA on training set ({n_components} components)...")
        pca = PCA(n_components=n_components)
        X_train_transformed = pca.fit_transform(X_train)
        X_test_transformed = pca.transform(X_test)

        # Report variance explained
        cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
        print(f"Variance explained by top PCs:")
        for n in [10, 50, 100, 200, 500]:
            if n <= len(cumvar):
                print(f"  {n} PCs: {cumvar[n-1]:.1f}%")
        if len(cumvar) > 0:
            print(f"  {len(cumvar)} PCs: {cumvar[-1]:.1f}%")

        # PCA-specific diagnostics
        if args.diagnostics:
            print("\n" + "-"*40)
            print("PCA-SPECIFIC DIAGNOSTICS")
            print("-"*40)

            # Check shapes
            print(f"\nShapes after PCA:")
            print(f"  X_train_transformed: {X_train_transformed.shape}")
            print(f"  X_test_transformed: {X_test_transformed.shape}")

            # Verify PCA was fit only on train by checking reconstruction
            # If test data was included, reconstruction error would be suspiciously low
            train_recon = pca.inverse_transform(X_train_transformed)
            test_recon = pca.inverse_transform(X_test_transformed)
            train_recon_err = np.mean((X_train - train_recon)**2)
            test_recon_err = np.mean((X_test - test_recon)**2)
            print(f"\nReconstruction MSE (should be similar if no leakage):")
            print(f"  Train: {train_recon_err:.4f}")
            print(f"  Test:  {test_recon_err:.4f}")
            print(f"  Ratio (test/train): {test_recon_err/train_recon_err:.2f}")

            # Shuffled label check in PC space
            print("\nShuffled label check in PC space (10 PCs)...")
            y_train_shuffled = np.random.permutation(y_train)
            scaler = StandardScaler()
            X_tr_pc = scaler.fit_transform(X_train_transformed[:, :10])
            X_te_pc = scaler.transform(X_test_transformed[:, :10])
            clf = LinearSVC(dual='auto', max_iter=5000, C=1.0)
            clf.fit(X_tr_pc, y_train_shuffled)
            shuffled_acc = clf.score(X_te_pc, y_test)
            print(f"  Shuffled labels: {shuffled_acc:.1%}")

            clf_real = LinearSVC(dual='auto', max_iter=5000, C=1.0)
            clf_real.fit(X_tr_pc, y_train)
            real_acc = clf_real.score(X_te_pc, y_test)
            print(f"  Real labels: {real_acc:.1%}")

            # Check if test samples project "too well" onto training PCs
            # by comparing variance of projections
            train_var = np.var(X_train_transformed, axis=0)
            test_var = np.var(X_test_transformed, axis=0)
            print(f"\nVariance of PC projections (first 10 PCs):")
            print(f"  Train: {train_var[:10]}")
            print(f"  Test:  {test_var[:10]}")
            print(f"  Ratio (test/train): {test_var[:10]/train_var[:10]}")

            # Check between-class vs within-class variance in PC space
            print(f"\nBetween-class vs within-class variance (first 10 PCs):")
            for pc_idx in range(min(10, X_train_transformed.shape[1])):
                pc_vals = X_train_transformed[:, pc_idx]
                class_means = [np.mean(pc_vals[y_train == c]) for c in range(n_stimuli)]
                grand_mean = np.mean(pc_vals)
                between_var = np.var(class_means)
                within_var = np.mean([np.var(pc_vals[y_train == c]) for c in range(n_stimuli)])
                ratio = between_var / within_var if within_var > 0 else float('inf')
                print(f"  PC{pc_idx}: between={between_var:.3f}, within={within_var:.3f}, ratio={ratio:.3f}")

            print("-"*40 + "\n")

        # Use transformed data
        X_train_use = X_train_transformed
        X_test_use = X_test_transformed

        # Sequential indices (PCs are already ordered by variance)
        feature_order = np.arange(n_components)
        n_features_total = n_components
        feature_label = "PCs"

    else:
        print("\nNEURON MODE: Sweeping over random neuron subsets")
        X_train_use = X_train
        X_test_use = X_test

        # Random ordering of neurons
        feature_order = np.random.permutation(n_neurons_total)
        n_features_total = n_neurons_total
        feature_label = "neurons"

    # Filter feature counts to those <= total features
    feature_counts = [n for n in feature_counts if n <= n_features_total]

    print(f"\nTesting {len(feature_counts)} {feature_label} counts: {feature_counts}")
    print(f"SVM C values: {C_values}")
    print(f"QDA shrinkage values: {shrinkage_values}")
    print()

    # Run decoding for each feature count
    svm_results = []
    qda_results = []
    for n_features in feature_counts:
        print(f"Decoding with {n_features} {feature_label}...")

        # Linear SVM
        print(f"  SVM: ", end="", flush=True)
        svm_result = decode_with_n_features(
            X_train_use, y_train, X_test_use, y_test,
            n_features, feature_order, C_values,
            cv_folds=cv_folds, random_state=args.seed
        )
        svm_result['n_features'] = n_features
        svm_results.append(svm_result)
        print(f"CV acc: {svm_result['cv_accuracy']:.1%}, "
              f"Test acc: {svm_result['test_accuracy']:.1%}")

        # QDA
        qda_label = "QDA (cov-only)" if args.covariance_only else "QDA"
        print(f"  {qda_label}: ", end="", flush=True)
        qda_result = decode_with_n_features_qda(
            X_train_use, y_train, X_test_use, y_test,
            n_features, feature_order, shrinkage_values,
            cv_folds=cv_folds, random_state=args.seed,
            covariance_only=args.covariance_only
        )
        qda_result['n_features'] = n_features
        qda_results.append(qda_result)
        print(f"CV acc: {qda_result['cv_accuracy']:.1%}, "
              f"Test acc: {qda_result['test_accuracy']:.1%}")

    # Extract arrays for plotting
    n_features_arr = np.array([r['n_features'] for r in svm_results])
    svm_test_acc = np.array([r['test_accuracy'] for r in svm_results])
    qda_test_acc = np.array([r['test_accuracy'] for r in qda_results])

    # Plot results - single plot comparing both methods
    fig, ax = plt.subplots(figsize=(8, 6))

    qda_plot_label = "QDA (cov-only)" if args.covariance_only else "QDA"
    ax.plot(n_features_arr, svm_test_acc * 100, 'o-', label='Linear SVM', markersize=8)
    ax.plot(n_features_arr, qda_test_acc * 100, 's-', label=qda_plot_label, markersize=8)
    ax.axhline(100 / n_stimuli, color='gray', linestyle='--', label='Chance')
    ax.set_xscale('log')
    ax.set_xlabel(f'Number of {feature_label}')
    ax.set_ylabel('Test Accuracy (%)')
    mode_str = "PC space" if args.use_pca else "neuron space"
    qda_title_str = "QDA (covariance-only)" if args.covariance_only else "QDA"
    ax.set_title(f'Stimulus Decoding: Linear SVM vs {qda_title_str} ({mode_str})\n'
                 f'({n_stimuli} stimuli, {n_trials} trials/stimulus)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()

    cov_suffix = '_covonly' if args.covariance_only else ''
    fig_name = f'decoding_vs_pcs{cov_suffix}.png' if args.use_pca else f'decoding_vs_neurons{cov_suffix}.png'
    fig_path = fig_dir / fig_name
    if args.save_fig:
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {fig_path}")

    # Print summary table
    print("\n" + "="*80)
    qda_mode_str = "COVARIANCE-ONLY " if args.covariance_only else ""
    print(f"DECODING RESULTS SUMMARY ({feature_label.upper()}, {qda_mode_str}QDA)")
    print("="*80)
    header_label = "N PCs" if args.use_pca else "N neurons"
    qda_header = "QDA(cov)" if args.covariance_only else "QDA Test"
    print(f"{header_label:>10} {'SVM Test':>12} {qda_header:>12} {'Difference':>12}")
    print("-"*80)
    for svm_r, qda_r in zip(svm_results, qda_results):
        diff = qda_r['test_accuracy'] - svm_r['test_accuracy']
        print(f"{svm_r['n_features']:>10} {svm_r['test_accuracy']:>12.1%} "
              f"{qda_r['test_accuracy']:>12.1%} {diff:>+12.1%}")
    print("="*80)

    # Best results
    svm_best_idx = np.argmax(svm_test_acc)
    qda_best_idx = np.argmax(qda_test_acc)
    qda_best_label = "QDA (cov-only)" if args.covariance_only else "QDA"
    print(f"\nBest SVM: {svm_test_acc[svm_best_idx]:.1%} with {n_features_arr[svm_best_idx]} {feature_label}")
    print(f"Best {qda_best_label}: {qda_test_acc[qda_best_idx]:.1%} with {n_features_arr[qda_best_idx]} {feature_label}")

    # If in PCA mode, report variance explained at best points
    if args.use_pca:
        cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
        svm_best_n = n_features_arr[svm_best_idx]
        qda_best_n = n_features_arr[qda_best_idx]
        print(f"\nVariance explained at best SVM ({svm_best_n} PCs): {cumvar[svm_best_n-1]:.1f}%")
        print(f"Variance explained at best {qda_best_label} ({qda_best_n} PCs): {cumvar[qda_best_n-1]:.1f}%")

    plt.show()


if __name__ == '__main__':
    main()
