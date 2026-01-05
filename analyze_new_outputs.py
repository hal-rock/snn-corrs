import numpy as np
import matplotlib.pyplot as plt
import csv
from corr_analyses import *

# AI slop to process tarek's poorly saved data
def desparsify_and_concatenate(sparse_data, total_time, bin_size, start_from=50):
    """
    Processes sparse event data into a single dense, concatenated matrix.

    This function takes a list of simulation runs, where each run contains
    sparse event times for multiple elements. It "desparsifies" each run into
    a dense binary matrix (elements x time_bins) and then concatenates these
    matrices horizontally.

    Args:
        sparse_data (list): A list of simulation runs. The expected structure is
            a list of lists of lists: (num_runs, num_elements, variable_num_events).
            For your case, this would be a 100-element list, where each element
            is a 5000-element list of event time lists.
        total_time (float): The total simulation time for a single run. This value
            defines the upper bound of the time axis.
        bin_size (float): The duration of a single time bin. The number of bins
            is calculated as ceil(total_time / bin_size).

    Returns:
        numpy.ndarray: A single dense binary matrix of shape
                       (num_elements, num_runs * num_time_bins),
                       with a memory-efficient dtype of np.uint8.
    """
    if not sparse_data:
        return np.array([])

    # Infer dimensions from the input data structure
    num_runs = len(sparse_data)
    # Assume all runs have the same number of elements based on the first run
    num_elements = len(sparse_data[0]) if num_runs > 0 else 0
    if num_elements == 0:
        return np.array([[] for _ in range(num_runs)])

    # Calculate the number of time bins required to cover the total time.
    # Using np.ceil ensures that events occurring near total_time are included.
    num_bins = int(np.ceil((total_time  - start_from) / bin_size))

    # This list will hold the dense matrix for each individual run
    all_dense_matrices = []

    # Process each run separately
    for run_idx in range(num_runs):
        run_data = sparse_data[run_idx]
        
        # Initialize a dense matrix of zeros for the current run.
        # Using np.uint8 is memory-efficient for binary (0/1) data.
        dense_run_matrix = np.zeros((num_elements, num_bins), dtype=np.uint8)
        
        # Iterate through each element (e.g., neuron or sensor) in the run
        for element_idx in range(num_elements):
            event_times = run_data[element_idx]
            
            # If there are no events for this element, skip to the next
            if len(event_times) == 0:
                continue
            
            # Convert event times to a NumPy array for fast, vectorized operations
            event_times_arr = np.array(event_times) - start_from
            event_times_arr = event_times_arr[event_times_arr > 0]
            
            # Calculate the corresponding bin index for each event time
            bin_indices = np.floor(event_times_arr / bin_size).astype(int)
            
            # Filter out indices that are out of the valid range [0, num_bins - 1].
            # This handles cases where an event time might be >= total_time.
            valid_indices = bin_indices[bin_indices < num_bins]

            # count instead of just add
            vals, counts = np.unique(valid_indices, return_counts=True)
            
            # Use the valid bin indices to set the corresponding positions to 1.
            # This is a form of advanced indexing and is very efficient.
            dense_run_matrix[element_idx, vals] = counts
            
        all_dense_matrices.append(dense_run_matrix)

    # Concatenate all the dense matrices along the time axis (axis=1)
    # This stacks them horizontally to create the final desired shape.
    final_matrix = np.concatenate(all_dense_matrices, axis=1)
    
    return final_matrix

def load_sesh(dir, n_trials=100):
    mats = []
    # tarek's loading code for this horrible CSV format
    for trial in range(n_trials):
        spike_trains_file = f'{dir}/trial{trial}/spike_monitor0.csv'
        with open(spike_trains_file, 'r', newline='') as csvf:
            readc = csv.reader(csvf, delimiter=',')
            mat = []
            for row in readc:
                row = [r.replace(" ms", '') for r in row]
                for b in range(0, len(row)):
                    if row[b] == '':
                        row[b] = None
                    elif 'u' in row[b]:
                        row[b] = row[b].replace(' us', '')
                        row[b] = float(row[b])
                        row[b] = row[b] / 1000
                    elif 's' in row[b]:
                        row[b] = row[b].replace(' s', '')
                        row[b] = float(row[b])
                        row[b] = row[b] * 1000
                    else:
                        row[b] = float(row[b])
                mat.append(row)
            mats.append(mat)
    
    bins = desparsify_and_concatenate(mats, total_time=1000, bin_size=1000) # 10 ms bins for now
    return bins

def load_fast(dir, n_trials=100, n_neurons=5000, bin_size=1000): # need n specified unforch
    mats = []
    # tarek's loading code for this horrible CSV format
    for trial in range(n_trials):
        spike_trains_file = f'{dir}/trial{trial}/spike_monitor0.npz'
        data = np.load(spike_trains_file)
        idxs = data['idxs']
        times = data['times']
        data.close()
        # need to process into stupider format for desparsify func unless I fix that
        # each entry in mat needs to be a list of neurons
        mat = [times[idxs == i] for i in range(n_neurons)] # I think?
        mats.append(mat)

    bins = desparsify_and_concatenate(mats, total_time=1000, bin_size=bin_size) # 10 ms bins for now
    return bins


def load_fast_vectorized(dir, n_trials=100, n_neurons=5000, bin_size=1000,
                         total_time=1000, start_from=50):
    """
    Vectorized loading - O(n_spikes) instead of O(n_neurons * n_spikes) per trial.
    Goes directly from sparse (idxs, times) to binned matrix without intermediate format.
    """
    n_bins = int(np.ceil((total_time - start_from) / bin_size))

    # Pre-allocate output: (neurons, trials * bins)
    result = np.zeros((n_neurons, n_trials * n_bins), dtype=np.uint8)

    for trial in range(n_trials):
        data = np.load(f'{dir}/trial{trial}/spike_monitor0.npz')
        idxs = data['idxs']
        times = data['times']
        data.close()

        # Filter to times after start_from
        mask = times > start_from
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

        # Count spikes with unbuffered addition - single pass through data
        np.add.at(result, (idxs, col_indices), 1)

    return result


if __name__ == '__main__':
    vsis = [0]
    bin_size = 1000

    n_inputs = 3000
    n_e = 4000

    n_stim = 3
    n_neurons = 250
    n_indices = np.random.choice(5000, n_neurons, replace=False)
    n_trials = 100
    n_pairs = (n_neurons * (n_neurons - 1)) // 2
    flat_corrs = np.zeros((n_stim, n_pairs))
    all_resps = np.zeros((n_neurons, n_stim, n_trials))
    all_inpts = np.zeros((n_neurons, n_stim, n_trials))
    all_inpt_corrs = np.zeros((n_stim, n_pairs))
    all_recs = np.zeros((n_neurons, n_stim, n_trials))
    all_rec_corrs = np.zeros((n_stim, n_pairs))
    save_dir = 'kappa_cont_test'
    for stim in range(n_stim):
        dir_name = f'{save_dir}/stim{stim}/'#trial{n}
        bins = load_fast(dir_name, n_trials=n_trials, bin_size=bin_size)#, n_neurons=n_neurons)
        # now these are in time so I can look at that
        print(bins.shape)

        # plt.plot(bins[:4000].mean(0)[:5*(1000//bin_size)])
        # plt.show()
        # bins = bins.reshape(bins.shape[0], n_trials, 1+(1000-50)//bin_size).sum(2)

        some_corrs = np.corrcoef(bins[n_indices]) # first 500 neurons without loss of generality
        some_corrs[np.isnan(some_corrs)] = 0
        trils = np.tril_indices(n_neurons, k=-1)
        some_flat = some_corrs[trils]
        some_resps = bins[n_indices]
        all_resps[:, stim] = some_resps
        flat_corrs[stim] = some_flat
        # also loadin inputs I guess
        inpt_spikes = np.load(f'{save_dir}/stim{stim}/input_spikes.npz')
        inpt_spikes = [v for v in inpt_spikes.values()]
        inpt_times = np.load(f'{save_dir}/stim{stim}/input_times.npz')
        inpt_times = [v for v in inpt_times.values()]
        counts = [[np.sum(times == i) for i in range(n_inputs)] for idxs, times in zip(inpt_spikes, inpt_times)]
        counts = np.array(counts)[:n_trials]
        #bins = desparsify_and_concatenate(mats, total_time=1000, bin_size=1000)
        # load in input connectivity
        conns = np.load(f'{save_dir}/conn_mats.npz')
        inpt_conn = conns['inpt'][:, n_indices]
        # get input to each cell
        inpt_sums = counts.dot(inpt_conn)
        inpt_corrs = np.corrcoef(inpt_sums.T)
        all_inpts[:, stim] = inpt_sums.T
        all_inpt_corrs[stim] = inpt_corrs[trils]
        # get recurrent inputs too
        # this is why we need bins from all neurons despite that being slow :(
        recur_conn = conns['recurrent']
        # make the negative weights negative
        recur_conn[:n_e, n_e:] *= -1/(3.3 * 2.26) # roughly right
        recur_conn[n_e:, n_e:] *= -1/(3.3 * 2.26) 
        recur_inpts = recur_conn.dot(bins)[n_indices] # I think that's the right order
        all_recs[:, stim] = recur_inpts
        rec_corrs = np.corrcoef(recur_inpts)
        rec_corrs[np.isnan(rec_corrs)] = 0
        all_rec_corrs[stim] = rec_corrs[trils]


    # dir_name = f'many_trials/stim0/'
    # bins = load_fast(dir_name, n_trials=n_stim*n_trials)
    # bins = bins[:n_neurons].reshape(n_neurons, n_stim, n_trials)
    # all_resps[...] = bins
    # for s in range(n_stim):
    #     some_corrs = np.corrcoef(bins[:, s])
    #     trils = np.tril_indices(n_neurons, k=-1)
    #     flat_corrs[s] = some_corrs[trils]
    
    # do basic ICC first for fun
    icc = calculate_icc(np.arctanh(flat_corrs))
    print(f'ICC of corrs across stimuli is {icc}')
    all_noises = bootstrap_all_pairs_variance(all_resps)
    corr_noise_var = np.mean(all_noises)
    z_corrs = np.arctanh(flat_corrs)
    snrs = np.power(z_corrs, 2) / all_noises

    avg_snrs = snrs.mean(0)
    avg_corrs = z_corrs.mean(0)
    print(f'estimate of correlation estimate noise variance: {corr_noise_var}')
    # want a sense of that vs overall corr variance
    print(f'compared to overall correlation variance estimate: {np.var(np.arctanh(flat_corrs))}')
    print(f'so fraction of unexplainable variance is {corr_noise_var / np.var(np.arctanh(flat_corrs))} :(')
    explainable_var, pair_var, leftover = calculate_explainable_variance_proportion(np.arctanh(flat_corrs), corr_noise_var)
    print(f'explainable ICC of corrs across stimuli is {explainable_var}')

    # before SNR shit, do this for inputs of each type
    inpt_noises = bootstrap_all_pairs_variance(all_inpts)
    inpt_noise_var = np.mean(inpt_noises)
    print(f'estimate of input correlation noise variance: {inpt_noise_var}')
    print(f'compared to overall input correlation variance estimate: {np.var(np.arctanh(all_inpt_corrs))}')
    print(f'so fraction of unexplainable variance is {inpt_noise_var / np.var(np.arctanh(all_inpt_corrs))} :(')
    explainable_var, pair_var, leftover = calculate_explainable_variance_proportion(np.arctanh(all_inpt_corrs), inpt_noise_var)
    print(f'explainable ICC of corrs across stimuli is {explainable_var}')

    # likewise for recurrent inputs.. maybe need to treat E and I differently later...
    rec_noises = bootstrap_all_pairs_variance(all_recs)
    rec_noise_var = np.mean(rec_noises)
    print(f'estimate of input correlation noise variance: {rec_noise_var}')
    print(f'compared to overall input correlation variance estimate: {np.var(np.arctanh(all_rec_corrs))}')
    print(f'so fraction of unexplainable variance is {rec_noise_var / np.var(np.arctanh(all_rec_corrs))} :(')
    explainable_var, pair_var, leftover = calculate_explainable_variance_proportion(np.arctanh(all_rec_corrs), rec_noise_var)
    print(f'explainable ICC of corrs across stimuli is {explainable_var}')




    # restrict to high SNR pairs I guess
    snrs = (z_corrs ** 2) / all_noises
    snr_thresh = 3
    good_idxs = snrs.mean(0) > snr_thresh
    print(f'number of pairs over {snr_thresh} SNR for some stimulus: {np.mean(good_idxs)}')
    g_flat_corrs = flat_corrs[:, good_idxs]
    g_z_corrs = z_corrs[:, good_idxs]
    g_all_noises = all_noises[:, good_idxs]
    g_corr_noise_var = np.mean(g_all_noises)
    print(f'estimate of correlation estimate noise variance: {g_corr_noise_var}')
    # want a sense of that vs overall corr variance
    print(f'compared to overall correlation variance estimate: {np.var(np.arctanh(g_flat_corrs))}')
    print(f'so fraction of unexplainable variance is {g_corr_noise_var / np.var(np.arctanh(g_flat_corrs))} :(')
    g_explainable_var, g_pair_var, g_leftover = calculate_explainable_variance_proportion(np.arctanh(g_flat_corrs), g_corr_noise_var)
    print(f'explainable ICC of corrs across stimuli is {g_explainable_var}')


    # now estimate variances

    # for vsi in vsis:
    #     bins = load_sesh(f'fast_test/')#/trial{trial}')
    #     e_traces = bins[:4000]
    #     i_traces = bins[4000:]
    #     e_rate = e_traces.mean() * 100
    #     e_rates.append(e_rate)
    #     i_rate = i_traces.mean() * 100
    #     i_rates.append(i_rate)

    #     # # check out PSTH at each reversal
    #     plt.figure()
    #     plt.plot(e_traces.mean(0) * 100, label='E')
    #     #plt.plot(i_rates.mean(0) * 100, label='I')
    #     plt.show()

    #     # get correlations
    #     corrs = np.corrcoef(bins)
    #     ee_corrs = corrs[:4000, :4000]
    #     ee_corrs = ee_corrs[np.tril_indices(4000, k=-1)]
    #     ei_corrs = corrs[:4000, 4000:].flatten()
    #     ee_means.append(np.nanmean(ee_corrs))
    #     ei_means.append(np.nanmean(ei_corrs))
    #     plt.figure()
    #     plt.hist(ee_corrs, bins=100)
    #     plt.show()