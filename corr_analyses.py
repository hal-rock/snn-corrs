### 101325 script for corr-based analyses, variance decomposition and RSA stuff
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial

import numpy as np

# gemini's code for 'quadratic d prime' to quantify
# how well you should be able to distinguish trials based on covariance alone
# note this should be used with centered data... per-class centered
def calculate_quadratic_d_prime(data, shrinkage=0.0):
    """
    Calculates the pairwise Quadratic d-prime (d'_Q) between conditions.
    
    Parameters:
    -----------
    data : np.ndarray
        Shape (N, S, T).
        N = Dimension of the data (features/neurons/sensors).
        S = Number of conditions/classes.
        T = Number of samples/trials per condition.
        
    shrinkage : float, optional (default=0.0)
        A value between 0 and 1.
        0.0 = No shrinkage (Empirical Covariance).
        1.0 = Full shrinkage (Diagonal matrix with mean variance).
        
    Returns:
    --------
    d_prime_matrix : np.ndarray
        Shape (S, S). A symmetric matrix where entry [i, j] is the 
        quadratic d-prime between condition i and condition j.
    """
    N, S, T = data.shape
    
    # 1. Estimate Covariance Matrices for each condition
    covs = []
    for s in range(S):
        # Extract data for condition s: shape (N, T)
        # N features (rows), T trials (cols) matches rowvar=True default
        condition_data = data[:, s, :]
        
        sigma = np.cov(condition_data, rowvar=True)
        
        # Apply Shrinkage (Ridge-like regularization)
        if shrinkage > 0:
            avg_var = np.trace(sigma) / N
            target = np.eye(N) * avg_var
            sigma = (1 - shrinkage) * sigma + shrinkage * target
            
        covs.append(sigma)
        
    covs = np.array(covs)
    
    # 2. Pre-compute Precision Matrices (Inverses)
    precs = []
    for sigma in covs:
        if shrinkage > 0:
            precs.append(np.linalg.inv(sigma))
        else:
            precs.append(np.linalg.pinv(sigma))
            
    d_prime_matrix = np.zeros((S, S))

    # 3. Compute Pairwise d'_Q
    for i in range(S):
        for j in range(i + 1, S):
            Sigma_i = covs[i]
            Sigma_j = covs[j]
            Prec_i = precs[i]
            Prec_j = precs[j]
            
            # The Decision Matrix M (Difference of Precision Matrices)
            M = Prec_j - Prec_i
            
            # Intermediate matrix products
            Sigma_i_M = Sigma_i @ M
            Sigma_j_M = Sigma_j @ M
            
            # --- Numerator (Signal) ---
            # Difference in expected "energy"
            mu_energy_i = np.trace(Sigma_i_M)
            mu_energy_j = np.trace(Sigma_j_M)
            signal = np.abs(mu_energy_i - mu_energy_j)
            
            # --- Denominator (Noise) ---
            # Variance of the quadratic form
            var_energy_i = 2 * np.trace(Sigma_i_M @ Sigma_i_M)
            var_energy_j = 2 * np.trace(Sigma_j_M @ Sigma_j_M)
            
            total_noise = np.sqrt(var_energy_i + var_energy_j)
            
            if total_noise < 1e-12:
                d_val = 0.0
            else:
                d_val = signal / total_noise
            
            d_prime_matrix[i, j] = d_val
            d_prime_matrix[j, i] = d_val
            
    return d_prime_matrix

# AI slop for quantifying "usefulness" of splitting stimuli by corrs
def calculate_icc(data):
    """
    Calculates the Intraclass Correlation Coefficient (ICC) for a given dataset.

    This function computes ICC(1) based on a one-way random-effects ANOVA.
    It assumes data is in a matrix format where columns represent different
    groups/targets and rows represent different raters/measurements.

    Args:
        data (np.ndarray): A 2D numpy array of shape (n_measurements, n_groups).
                           In your case, this would be (12, ~40000).

    Returns:
        float: The Intraclass Correlation Coefficient (ICC).
    """
    # Get the dimensions of the data
    # n = number of measurements (raters), k = number of groups (targets)
    n, k = data.shape

    # Calculate the mean per group (along columns) and the grand mean
    group_means = np.mean(data, axis=0)
    grand_mean = np.mean(data)

    # Calculate Sum of Squares Between (SSB) and Sum of Squares Within (SSW)
    ss_between = n * np.sum((group_means - grand_mean)**2)
    ss_within = np.sum((data - group_means)**2)

    # Calculate Mean Square Between (MSB) and Mean Square Within (MSW)
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (k * (n - 1))

    # Calculate the ICC
    icc = (ms_between - ms_within) / (ms_between + (n - 1) * ms_within)

    return icc

# AI slop for adjusting the ICC with explainable variance
import numpy as np

def calculate_explainable_variance_proportion(data, measurement_noise_variance):
    """
    Calculates the proportion of explainable variance attributable to group differences.

    This function partitions the total variance into three components:
    1. True Between-Group Variance
    2. True Within-Group Variance
    3. Measurement Noise Variance (provided by the user)

    It then returns the ratio of Between-Group Variance to the sum of the two "true"
    (explainable) variance components.

    Args:
        data (np.ndarray): A 2D numpy array of shape (n_measurements, n_groups).
        measurement_noise_variance (float): The user's estimate of the average
                                            variance of the measurement noise.

    Returns:
        float: The proportion of explainable variance explained by the grouping.
    """
    # Get the dimensions of the data
    n, k = data.shape

    # --- Step 1: Get MS_Between and MS_Within from ANOVA ---
    group_means = np.mean(data, axis=0)
    grand_mean = np.mean(data)
    
    ss_between = n * np.sum((group_means - grand_mean)**2)
    ss_within = np.sum((data - group_means)**2)
    
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (k * (n - 1))

    # --- Step 2: Calculate the three variance components ---
    
    # Measurement noise variance is provided directly
    v_noise = measurement_noise_variance
    
    # True within-group variance is what's left after subtracting noise
    # We cap it at 0 in case the noise model is overestimated.
    v_true_within = max(0, ms_within - v_noise)
    
    # True between-group variance
    v_between = (ms_between - ms_within) / n
    # Cap at 0 in case of sampling error where MS_Within > MS_Between
    v_between = max(0, v_between)
    print(ms_within)
    print(v_noise)


    # --- Step 3: Calculate the final proportion ---
    
    # The total "explainable" variance is the sum of the true (non-noise) parts
    total_explainable_variance = v_between + v_true_within
    
    if total_explainable_variance == 0:
        return 0.0

    proportion = v_between / total_explainable_variance
    
    return proportion, v_between, total_explainable_variance - v_between

# AI slop for bootstrapping 'measurement noise' of corrs
def bootstrap_correlation_variance_grouped(
    original_data, 
    num_total_pairs_to_sample, 
    num_bootstrap_samples=1000
):
    """
    Estimates measurement noise variance from grouped data via balanced bootstrapping.

    This function takes a 3D dataset of shape (N, C, K), where C is the number of
    groups. It samples a total number of pairs evenly distributed across the C
    groups, performs a bootstrap analysis on each, and returns the average variance.

    Args:
        original_data (np.ndarray): A 3D array of shape (N, C, K), where:
                                    N = number of variables to correlate
                                    C = number of groups (e.g., 12)
                                    K = number of data points per variable (e.g., ~100)
        num_total_pairs_to_sample (int): The total number of random variable pairs
                                         to sample across all groups. This number
                                         MUST be divisible by C.
        num_bootstrap_samples (int): The number of bootstrap resamples for each pair.

    Returns:
        float: The average variance of the Fisher z-transformed correlation
               coefficients, serving as the measurement noise estimate.
    """
    if original_data.ndim != 3:
        raise ValueError(f"original_data must be a 3D array of shape (N, C, K), but got {original_data.ndim} dimensions.")
    
    num_vars, num_groups, num_points = original_data.shape
    
    if num_total_pairs_to_sample % num_groups != 0:
        raise ValueError(f"num_total_pairs_to_sample ({num_total_pairs_to_sample}) must be divisible by the number of groups C ({num_groups}).")
        
    if num_vars < 2:
        raise ValueError("original_data must have at least 2 variables (dim 0).")

    pairs_per_group = num_total_pairs_to_sample // num_groups
    all_variances = []
    all_corrs = [] # record actual correlations
    
    variable_indices = np.arange(num_vars)

    # Loop through each of the C groups
    for group_idx in range(num_groups):
        group_data = original_data[:, group_idx, :] # Shape is now (N, K)
        
        # Sample the required number of pairs within this group
        for _ in range(pairs_per_group):
            idx1, idx2 = np.random.choice(variable_indices, 2, replace=False)
            var1_data = group_data[idx1]
            var2_data = group_data[idx2]
            all_corrs.append(np.arctanh(pearsonr(var1_data, var2_data)[0]))
            
            # --- Vectorized Bootstrap (core logic is unchanged) ---
            bootstrap_indices = np.random.randint(
                0, num_points, size=(num_bootstrap_samples, num_points)
            )
            x_resampled = var1_data[bootstrap_indices]
            y_resampled = var2_data[bootstrap_indices]
            
            x_mean = x_resampled.mean(axis=1, keepdims=True)
            y_mean = y_resampled.mean(axis=1, keepdims=True)
            x_centered = x_resampled - x_mean
            y_centered = y_resampled - y_mean
            
            numerator = np.sum(x_centered * y_centered, axis=1)
            denominator = np.sqrt(np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1))
            correlations = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator != 0))
            
            correlations = np.clip(correlations, -1 + 1e-9, 1 - 1e-9)
            z_transformed_corrs = np.arctanh(correlations)
            
            variance_in_z_space = np.var(z_transformed_corrs, ddof=1)
            all_variances.append(variance_in_z_space)

    return all_variances, all_corrs #np.mean(all_variances)

def bootstrap_worker(boot_idx, group_data, var_idx1, var_idx2, K, seed):
    """
    (Internal subfunction) Performs one bootstrap resample and correlation calculation.
    This function is called in parallel by the main function.
    """
    rng = np.random.default_rng(seed)
    # 1. Create one bootstrap resample of the data
    resample_indices = rng.integers(0, K, size=K)
    resampled_data = group_data[:, resample_indices]

    # 2. Select the data for every pair
    x_resampled = resampled_data[var_idx1, :]
    y_resampled = resampled_data[var_idx2, :]
    
    # 3. Perform vectorized correlation over all pairs for this one sample
    x_mean = x_resampled.mean(axis=1, keepdims=True)
    y_mean = y_resampled.mean(axis=1, keepdims=True)
    x_centered = x_resampled - x_mean
    y_centered = y_resampled - y_mean
    
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(
        np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1)
    )
    
    # Return a (P,) array of correlations for this bootstrap sample
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator != 0))


def bootstrap_all_pairs_variance(original_data, num_bootstrap_samples=1000):
    """
    Calculates bootstrap variance for all pairs, parallelized over bootstrap samples.
    
    Args:
        original_data (np.ndarray): 3D array of shape (N, C, K).
        num_bootstrap_samples (int): The number of bootstrap resamples.

    Returns:
        np.ndarray: 2D array of shape (C, P) with the variance for each pair,
                    where P = N*(N-1)/2.
    """
    if original_data.ndim != 3:
        raise ValueError("original_data must be a 3D array of shape (N, C, K).")
    
    N, C, K = original_data.shape
    
    if N < 2:
        raise ValueError("original_data must have at least 2 variables (dim 0).")

    # Use tril_indices as requested to get all unique pairs of variables
    var_idx1, var_idx2 = np.tril_indices(N, k=-1)
    num_pairs = len(var_idx1)
    
    all_group_variances = []

    # Generate independent seeds for each worker call across all groups
    ss = np.random.SeedSequence()
    all_seeds = ss.spawn(C * num_bootstrap_samples)

    # Outer loop over groups (serial, as requested due to memory constraints)
    for group_idx in range(C):
        print(f"Processing group {group_idx + 1}/{C}...")
        group_data = original_data[:, group_idx, :]
        group_seeds = all_seeds[group_idx * num_bootstrap_samples:(group_idx + 1) * num_bootstrap_samples]

        # Build argument tuples with unique seeds for each call
        args = [(i, group_data, var_idx1, var_idx2, K, seed)
                for i, seed in enumerate(group_seeds)]

        # Create a pool and map the work across cores
        with multiprocessing.Pool() as pool:
            results = pool.starmap(bootstrap_worker, args)

        # 4. Consolidate results and find variance
        # Shape: (B, P)
        correlations_matrix = np.array(results)

        correlations_matrix = np.clip(correlations_matrix, -1 + 1e-9, 1 - 1e-9)
        z_transformed_corrs = np.arctanh(correlations_matrix)

        # Calculate variance along axis 0 (the bootstrap sample axis)
        # Shape: (P,)
        variances = np.var(z_transformed_corrs, axis=0, ddof=1)
        all_group_variances.append(variances)

    return np.array(all_group_variances)


def splithalf_worker(split_idx, group_data, var_idx1, var_idx2, K, seed):
    """
    (Internal subfunction) Performs one random split-half and returns the difference
    in Fisher-z correlations between halves, for all pairs. Shape: (P,).
    """
    rng = np.random.default_rng(seed)
    # Random permutation, split into two non-overlapping halves
    perm = rng.permutation(K)
    half = K // 2
    idx_a, idx_b = perm[:half], perm[half:2*half]

    data_a = group_data[:, idx_a]
    data_b = group_data[:, idx_b]

    def _vectorized_corr(data, i1, i2):
        x = data[i1, :]
        y = data[i2, :]
        x_c = x - x.mean(axis=1, keepdims=True)
        y_c = y - y.mean(axis=1, keepdims=True)
        num = np.sum(x_c * y_c, axis=1)
        den = np.sqrt(np.sum(x_c**2, axis=1) * np.sum(y_c**2, axis=1))
        return np.divide(num, den, out=np.zeros_like(num), where=(den != 0))

    r_a = _vectorized_corr(data_a, var_idx1, var_idx2)
    r_b = _vectorized_corr(data_b, var_idx1, var_idx2)

    r_a = np.clip(r_a, -1 + 1e-9, 1 - 1e-9)
    r_b = np.clip(r_b, -1 + 1e-9, 1 - 1e-9)

    return np.arctanh(r_a) - np.arctanh(r_b)


def splithalf_all_pairs_variance(original_data, num_splits=1000):
    """
    Estimates correlation measurement noise variance via split-half trial splitting.

    For each random split, trials are partitioned into two independent halves, correlations
    are computed from each half, and the squared Fisher-z difference is used to estimate
    variance. Since the halves are independent, Var(z_A - z_B) = Var(z_A) + Var(z_B),
    and under ~1/n scaling of Fisher-z variance, Var(z_full) ≈ mean(d²) / 4.

    Args:
        original_data (np.ndarray): 3D array of shape (N, C, K).
                                    N: number of variables (cells)
                                    C: number of groups (stimuli)
                                    K: number of observations (trials)
        num_splits (int): Number of random split-halves to average over.

    Returns:
        np.ndarray: 2D array of shape (C, P) with the estimated variance of the
                    Fisher-z correlation for each pair using the full K trials,
                    where P = N*(N-1)/2.
    """
    if original_data.ndim != 3:
        raise ValueError("original_data must be a 3D array of shape (N, C, K).")

    N, C, K = original_data.shape

    if N < 2:
        raise ValueError("original_data must have at least 2 variables (dim 0).")
    if K < 4:
        raise ValueError("Need at least 4 trials to split into halves of size >= 2.")

    var_idx1, var_idx2 = np.tril_indices(N, k=-1)

    all_group_variances = []

    # Generate independent seeds for each worker call across all groups
    ss = np.random.SeedSequence()
    all_seeds = ss.spawn(C * num_splits)

    for group_idx in range(C):
        print(f"Processing group {group_idx + 1}/{C}...")
        group_data = original_data[:, group_idx, :]
        group_seeds = all_seeds[group_idx * num_splits:(group_idx + 1) * num_splits]

        args = [(i, group_data, var_idx1, var_idx2, K, seed)
                for i, seed in enumerate(group_seeds)]

        with multiprocessing.Pool() as pool:
            results = pool.starmap(splithalf_worker, args)

        # Shape: (num_splits, P)
        diffs = np.array(results)
        # Var(z_full) ≈ mean(d²) / 4, using mean(d²) since E[d]=0 by construction
        variances = np.mean(diffs**2, axis=0) / 4.0
        all_group_variances.append(variances)

    return np.array(all_group_variances)


def analytical_correlation_variance(original_data):
    """
    Returns the Gaussian-theory analytical variance of Fisher-z transformed correlations.

    Under bivariate Gaussian assumptions, Var(arctanh(r)) ≈ 1/(K-3).
    This is expected to be an underestimate for non-Gaussian neural data.

    Args:
        original_data (np.ndarray): 3D array of shape (N, C, K).

    Returns:
        np.ndarray: 2D array of shape (C, P) filled with 1/(K-3),
                    where P = N*(N-1)/2.
    """
    if original_data.ndim != 3:
        raise ValueError("original_data must be a 3D array of shape (N, C, K).")

    N, C, K = original_data.shape
    num_pairs = N * (N - 1) // 2
    return np.full((C, num_pairs), 1.0 / (K - 3))


# make paired bar plot things for off and on diagonals
# AI slop for that
def create_bar_scatter_plot(ax, data, title, y_label):
    """
    Creates a bar-and-scatter plot on a given matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): The subplot axis to draw on.
        data (list of tuples): The data, e.g., [(val1_a, val1_b), (val2_a, val2_b), ...].
        title (str): The title for the subplot.
        y_label (str): The label for the y-axis.
    """
    # 1. Prepare the data by separating tuples into two lists
    group1_vals = np.array([item[0] for item in data])
    group2_vals = np.array([item[1] for item in data])
    
    # 2. Calculate statistics: mean and standard error of the mean (SEM)
    mean1, mean2 = np.mean(group1_vals), np.mean(group2_vals)
    # SEM = standard deviation / sqrt(number of samples)
    sem1 = np.std(group1_vals, ddof=1) / np.sqrt(len(group1_vals))
    sem2 = np.std(group2_vals, ddof=1) / np.sqrt(len(group2_vals))

    # 3. Plot the bars with error bars
    # The 'yerr' argument automatically adds the error bars. 'capsize' adds the caps.
    ax.bar(
        x=[0, 1], 
        height=[mean1, mean2], 
        yerr=[sem1, sem2], 
        capsize=10,
        color=['skyblue', 'lightcoral'],
        alpha=0.6, # Make bars semi-transparent
        edgecolor='black'
    )

    # 4. Plot the individual data points and connecting lines
    # We loop through each pair of points to draw the scatter dots and the line between them.
    for val1, val2 in zip(group1_vals, group2_vals):
        ax.plot(
            [0, 1], # x-coordinates for the two points
            [val1, val2], # y-coordinates for the two points
            'o-', # 'o' for dots, '-' for the line
            color='gray',
            alpha=0.8,
            linewidth=1.5,
            markersize=5
        )

    # 5. Customize the plot's appearance
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['off-diagonal', 'rest of RSA matrix'], fontsize=12)
    
    # Set y-axis to start at 0 and have some padding at the top
    max_val = np.max(np.concatenate((group1_vals, group2_vals)))
    min_val = ax.get_ylim()[0]
    ax.set_ylim(min_val, max_val * 1.15)
    
    # Add a grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)

def get_off_diag(rsa_mat):
    '''off-diagonal values (including the rollover one) for grating RSA matrices
    and the rest of them
    returns just one tri for the off-diagonal but all for the rest, which is stupid, so just returns means
    '''
    dims = rsa_mat.shape[0]
    idxs = np.eye(dims, dtype=bool)
    off = np.roll(idxs, 1, axis=1)
    opp = np.roll(idxs, -1, axis=1)
    rest = ~np.logical_or(off, opp)
    np.fill_diagonal(rest, False)
    
    return np.mean(rsa_mat[off]), np.mean(rsa_mat[rest])


# bootstrap shit for partial corrs, partialling out mean
import numpy as np
import multiprocessing
from functools import partial

def bootstrap_partial_worker(boot_idx, group_data, var_idx1, var_idx2, K, seed):
    """
    (Internal subfunction) Performs one bootstrap resample and partial correlation calculation.

    For each pair of variables (x, y), it partials out the mean of all other variables (z).
    This function is called in parallel by the main function.
    """
    rng = np.random.default_rng(seed)
    # 1. Create one bootstrap resample of the data
    resample_indices = rng.integers(0, K, size=K)
    resampled_data = group_data[:, resample_indices]

    N = group_data.shape[0]
    num_pairs = len(var_idx1)
    partial_correlations = np.zeros(num_pairs)
    all_indices = np.arange(N)

    # 2. Loop over all pairs since the 'z' variable is unique for each pair
    for p_idx in range(num_pairs):
        i = var_idx1[p_idx]
        j = var_idx2[p_idx]

        # Select the data for the pair (x, y)
        x = resampled_data[i, :]
        y = resampled_data[j, :]

        # Define z as the mean of all OTHER N-2 variables
        other_vars_mask = (all_indices != i) & (all_indices != j)
        if np.sum(other_vars_mask) > 0:
            z = np.mean(resampled_data[other_vars_mask, :], axis=0)
        else: # Handle case where N=2, so there are no other variables
            z = np.zeros_like(x)

        # Stack x, y, and z to compute the correlation matrix
        # Note: np.corrcoef is equivalent to manually calculating Pearson's r
        data_matrix = np.vstack([x, y, z])
        
        # Guard against zero-variance slices, which would cause corrcoef to fail
        if np.any(np.std(data_matrix, axis=1) < 1e-9):
            partial_correlations[p_idx] = 0.0
            continue

        corr_matrix = np.corrcoef(data_matrix)
        r_xy = corr_matrix[0, 1]
        r_xz = corr_matrix[0, 2]
        r_yz = corr_matrix[1, 2]

        # 3. Use the standard formula for partial correlation
        numerator = r_xy - (r_xz * r_yz)
        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        
        # Avoid division by zero if a variable is perfectly correlated with z
        if denominator > 1e-9:
            partial_correlations[p_idx] = numerator / denominator
        else:
            partial_correlations[p_idx] = 0.0
            
    return partial_correlations


def bootstrap_all_pairs_partial_variance(original_data, num_bootstrap_samples=1000):
    """
    Calculates bootstrap variance for all pairs' partial correlations, 
    parallelized over bootstrap samples.
    
    Args:
        original_data (np.ndarray): 3D array of shape (N, C, K).
                                    N: number of variables
                                    C: number of conditions/groups
                                    K: number of observations
        num_bootstrap_samples (int): The number of bootstrap resamples.

    Returns:
        np.ndarray: 2D array of shape (C, P) with the partial correlation variance 
                    for each pair, where P = N*(N-1)/2.
    """
    if original_data.ndim != 3:
        raise ValueError("original_data must be a 3D array of shape (N, C, K).")
    
    N, C, K = original_data.shape
    
    if N < 2:
        raise ValueError("original_data must have at least 2 variables (dim 0).")

    # Get indices for all unique pairs of variables
    var_idx1, var_idx2 = np.tril_indices(N, k=-1)
    
    all_group_variances = []

    # Generate independent seeds for each worker call across all groups
    ss = np.random.SeedSequence()
    all_seeds = ss.spawn(C * num_bootstrap_samples)

    # Outer loop over groups (conditions) is serial to manage memory
    for group_idx in range(C):
        print(f"Processing group {group_idx + 1}/{C}...")
        group_data = original_data[:, group_idx, :]
        group_seeds = all_seeds[group_idx * num_bootstrap_samples:(group_idx + 1) * num_bootstrap_samples]

        args = [(i, group_data, var_idx1, var_idx2, K, seed)
                for i, seed in enumerate(group_seeds)]

        # Create a pool of workers and map the task across available cores
        with multiprocessing.Pool() as pool:
            results = pool.starmap(bootstrap_partial_worker, args)

        # 4. Consolidate results and find variance
        # Shape: (B, P) -> (num_bootstrap_samples, num_pairs)
        correlations_matrix = np.array(results)
        
        # Apply Fisher's z-transformation to stabilize the variance
        # Clip values to avoid infinity at +/- 1
        correlations_matrix = np.clip(correlations_matrix, -1 + 1e-9, 1 - 1e-9)
        z_transformed_corrs = np.arctanh(correlations_matrix)
        
        # Calculate variance along axis 0 (the bootstrap sample axis)
        # Shape: (P,)
        variances = np.var(z_transformed_corrs, axis=0, ddof=1)
        all_group_variances.append(variances)

    return np.array(all_group_variances)


# AI slop function for pairwise geometric means
def compute_pairwise_gmean(arr):
    """
    Computes the pairwise geometric mean for an M x N array.

    The geometric mean of two numbers (a, b) is sqrt(a * b).

    This function computes this for every pair of rows (i, j) in the
    input array, for each column k.

    The result G[i, j, k] = sqrt(arr[i, k] * arr[j, k]).

    Args:
        arr (np.ndarray): The input array of shape (M, N).
                          Assumes all elements are non-negative.

    Returns:
        np.ndarray: The output array of shape (M, M, N).
    """
    # 1. Reshape the array into two broadcastable forms.

    # Shape becomes (M, 1, N).
    # This treats each M-row as a separate entity to be broadcast.
    a1 = arr[:, np.newaxis, :]

    # Shape becomes (1, M, N).
    # This treats each M-row (now as columns) as a separate entity.
    a2 = arr[np.newaxis, :, :]

    # 2. Multiply them.
    # Due to broadcasting, NumPy multiplies (M, 1, N) * (1, M, N)
    # The result is an (M, M, N) array.
    # The element at [i, j, k] is a1[i, 0, k] * a2[0, j, k]
    # which is exactly arr[i, k] * arr[j, k].
    products = a1 * a2

    # 3. Compute the square root to get the geometric mean.
    # This is an element-wise operation.
    geometric_means = np.sqrt(products)

    return geometric_means
