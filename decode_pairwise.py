"""
Pairwise decoding of stimulus identity as a function of angular separation.

For each separation j = 1..N/2, trains linear SVM and QDA to discriminate
stimulus 0 vs stimulus j (binary classification). Plots accuracy vs angular
separation to reveal the resolution limit of each decoder type.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from analysis import load_params, get_n_stimuli
from decode_stimulus import load_responses


def load_stimulus_centers(data_dir, n_stimuli, params):
    """Load 1D stimulus centers in radians from saved data.

    Tries (in order):
    1. stimulus_center.npy files in each stim directory
    2. _stimulus_centers from params.json
    3. Falls back to assuming uniform [0, pi) spacing
    """
    data_dir = Path(data_dir)

    # Try loading from per-stimulus files
    centers = []
    for s in range(n_stimuli):
        sc_file = data_dir / f'stim{s}' / 'stimulus_center.npy'
        if sc_file.exists():
            c = np.load(sc_file)
            centers.append(float(c[0]) if c.ndim > 0 else float(c))
        else:
            break
    if len(centers) == n_stimuli:
        return np.array(centers)

    # Try from params
    if '_stimulus_centers' in params:
        sc = params['_stimulus_centers']
        return np.array([c[0] if isinstance(c, list) else c for c in sc])

    # Fallback: uniform spacing over [0, pi)
    return np.linspace(0, np.pi, n_stimuli, endpoint=False)


def decode_pair_svm(X_train, y_train, X_test, y_test,
                    C_values, cv_folds=5, random_state=42):
    """Binary SVM decoding with CV'd regularization."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', LinearSVC(dual='auto', max_iter=5000, random_state=random_state))
    ])
    grid = GridSearchCV(
        pipeline, {'svm__C': C_values},
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        scoring='accuracy', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.score(X_test, y_test)


def decode_pair_qda(X_train, y_train, X_test, y_test,
                    shrinkage_values, cv_folds=5, random_state=42):
    """Binary QDA decoding with CV'd shrinkage."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    best_shrinkage = shrinkage_values[0]
    best_cv_acc = -1

    for shrinkage in shrinkage_values:
        fold_accs = []
        for train_idx, val_idx in cv.split(X_train_s, y_train):
            qda = QuadraticDiscriminantAnalysis(solver='eigen', shrinkage=shrinkage)
            qda.fit(X_train_s[train_idx], y_train[train_idx])
            fold_accs.append(qda.score(X_train_s[val_idx], y_train[val_idx]))
        mean_acc = np.mean(fold_accs)
        if mean_acc > best_cv_acc:
            best_cv_acc = mean_acc
            best_shrinkage = shrinkage

    final_qda = QuadraticDiscriminantAnalysis(solver='eigen', shrinkage=best_shrinkage)
    final_qda.fit(X_train_s, y_train)
    return final_qda.score(X_test_s, y_test)


def main():
    parser = argparse.ArgumentParser(
        description='Pairwise decoding accuracy vs angular separation on the stimulus ring'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to param point directory')
    parser.add_argument('--n_neurons', type=int, required=True,
                        help='Number of neurons to use for decoding')
    parser.add_argument('--n_stimuli', type=int, default=None,
                        help='Number of stimuli (auto-detect if omitted)')
    parser.add_argument('--duration', type=float, default=None,
                        help='Spike counting window in ms after 50ms warmup')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_fig', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    # Load all data once
    print("Loading data...")
    params = load_params(args.data_dir)
    X, y = load_responses(args.data_dir, params, n_stimuli=args.n_stimuli,
                          duration=args.duration)

    n_samples, n_neurons_total = X.shape
    n_stimuli = args.n_stimuli if args.n_stimuli else get_n_stimuli(params, args.data_dir)
    n_trials = params['n_trials']

    # Load actual stimulus centers (radians) for computing real separations
    stimulus_centers_rad = load_stimulus_centers(args.data_dir, n_stimuli, params)

    print(f"  {n_stimuli} stimuli, {n_trials} trials each, {n_neurons_total} neurons total")
    if stimulus_centers_rad is not None:
        span_deg = np.rad2deg(stimulus_centers_rad[-1] - stimulus_centers_rad[0])
        print(f"  Stimulus range: {span_deg:.1f} degrees")

    # Select random neuron subset
    assert args.n_neurons <= n_neurons_total, \
        f"Requested {args.n_neurons} neurons but only {n_neurons_total} available"
    neuron_idx = np.random.choice(n_neurons_total, args.n_neurons, replace=False)
    neuron_idx.sort()
    X = X[:, neuron_idx]
    print(f"  Using {args.n_neurons} randomly selected neurons")

    # Decoding hyperparameters
    C_values = np.logspace(-4, 2, 13)
    shrinkage_values = np.linspace(0.1, 1.0, 10)
    test_size = 0.2
    cv_folds = 5

    # Pairwise decoding loop: decode stimulus 0 vs all others, sorted by separation
    # Compute actual angular separations from stimulus centers
    ref_center = stimulus_centers_rad[0]
    other_indices = np.arange(1, n_stimuli)
    other_seps_rad = np.abs(stimulus_centers_rad[1:] - ref_center)
    # Sort by separation
    sort_order = np.argsort(other_seps_rad)
    other_indices = other_indices[sort_order]
    other_seps_deg = np.rad2deg(other_seps_rad[sort_order])

    n_pairs = len(other_indices)
    separations_deg = other_seps_deg
    svm_accs = np.zeros(n_pairs)
    qda_accs = np.zeros(n_pairs)

    print(f"\nDecoding stimulus 0 vs stimulus j, for j in 1..{n_stimuli-1} (sorted by separation)")
    print(f"{'j':>4} {'sep (deg)':>10} {'SVM':>8} {'QDA':>8}")
    print("-" * 34)

    for i, j in enumerate(other_indices):
        sep_deg = separations_deg[i]

        # Subset to binary problem
        mask = (y == 0) | (y == j)
        X_pair = X[mask]
        y_pair = y[mask]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_pair, y_pair, test_size=test_size, stratify=y_pair, random_state=args.seed
        )

        # SVM
        svm_acc = decode_pair_svm(X_train, y_train, X_test, y_test,
                                  C_values, cv_folds=cv_folds, random_state=args.seed)
        svm_accs[i] = svm_acc

        # QDA
        qda_acc = decode_pair_qda(X_train, y_train, X_test, y_test,
                                  shrinkage_values, cv_folds=cv_folds, random_state=args.seed)
        qda_accs[i] = qda_acc

        print(f"{j:>4} {sep_deg:>10.1f} {svm_acc:>7.1%} {qda_acc:>7.1%}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(separations_deg, svm_accs * 100, 'o-', label='Linear SVM', markersize=5)
    ax.plot(separations_deg, qda_accs * 100, 's-', label='QDA', markersize=5)
    ax.axhline(50, color='gray', linestyle='--', label='Chance')
    ax.set_xlabel('Angular separation (degrees)')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title(f'Pairwise decoding: stimulus 0 vs stimulus j\n'
                 f'({args.n_neurons} neurons, {n_stimuli} stimuli, {n_trials} trials)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([40, 105])

    plt.tight_layout()

    if args.save_fig:
        dur_str = f"{args.duration}ms" if args.duration else "full"
        fig_path = fig_dir / f"pairwise_decoding_{args.n_neurons}neurons_{dur_str}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {fig_path}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
