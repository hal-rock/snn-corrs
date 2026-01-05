import random
import numpy as np
from os import path, makedirs
from scipy import stats
import math
import multiprocessing
from functools import partial

N_THREADS = multiprocessing.cpu_count() - 2 # bit of wiggle room

def generate_trial_spikes(i, rate_vec, input_time):
    '''generate one trial of spikes'''
    n_neurons = len(rate_vec)
    # assume 1ms resolution
    spikes_dense = np.random.rand(n_neurons, input_time) < (rate_vec / 1000).reshape(-1, 1)
    spike_times = [np.nonzero(neu)[0] for neu in spikes_dense]

    # return indices, times parallel arrays, which brian wants
    spike_idxs = np.concatenate([[i] * len(spike_times[i]) for i in range(n_neurons)])
    spike_times = np.concatenate(spike_times)

    return spike_idxs, spike_times


def generate_input_spikes(rate_vec, n_trials, input_time, output_file):
    '''
    generate poisson spikes from a vector of rates (Hz), for a set number of trials/time in ms, and save them to output_file

    each trial is a list of arrays, total result is then a list of lists of arrays, nesting trial -> neuron -> spike times

    deal with the connectivity and rate vector generation elsewhere

    this is just continuous input, for now
    '''
    gen_func = partial(generate_trial_spikes, rate_vec=rate_vec, input_time=int(input_time))

    pool = multiprocessing.Pool(N_THREADS)
    outputs = list(pool.map(gen_func, list(range(n_trials))))
    idxs = [o[0] for o in outputs]
    times = [o[1] for o in outputs] # unwrap index and time arrays
    pool.close()

    #np.save(output_file, trials, allow_pickle=True) # disable for now, doesn't like list of lists
    return idxs, times #trials # return and/or save?
    

class Sike(object):
    def __init__(self, n_input_units, n_targets, n_neurons, prob_conn, rate_mean, rate_std, input_time, output_file,
                 multiplier, cyclical=False, separate_files=False):
        pass

    def generate_inputs(self):
        try:
            bins = int(self.input_time * 1000)   # convert to ms because each bin will be 1 ms
            neuron_indices = []
            spike_times = []
            adj_mat = numpy.zeros((self.n_input_units, self.n_neurons))
            small_mat = numpy.zeros((self.n_input_units, self.n_targets))

            for i in range(0, self.n_input_units):
                # create spike times
                # bin_vals = [random.random() for x in range(0, (bins-1))]
                # spikes = [x+1 for x in range(0, (bins-1)) if bin_vals[x] <= self.rate_mean/1000]
                # rep_ind = [i for x in spikes]

                if not self.cyclical_flag:
                    bin_vals = [random.random() for x in range(0, bins-1)]
                    spikes = [x+1 for x in range(0, bins-1) if bin_vals[x] <= (numpy.random.lognormal(self.rate_mean, self.rate_std))/1000]  # (self.rate_mean * numpy.random.uniform(0, 1))/1000]
                    rep_ind = [i for x in spikes]

                if self.cyclical_flag:
                    fr_max = numpy.random.lognormal(self.rate_mean, self.rate_std)
                    fr_vals = [(fr_max / 2) * math.sin(2 * math.pi * 0.004 * t) + (fr_max / 2) for t in range(0, bins - 1)]
                    bin_vals = [random.random() for x in range(0, bins - 1)]
                    spikes = [x + 1 for x in range(0, bins - 1) if bin_vals[x] <= fr_vals[x] / 1000]
                    rep_ind = [i for x in spikes]

                # add to the lists
                neuron_indices.extend(rep_ind)
                spike_times.extend(spikes)

                # CODE to favor input to neurons with low out degrees
                '''cutoff = stats.norm.ppf(self.prob_conn)
                out_degrees = [sum(y) for y in self.conn_mat[0:len(self.conn_mat[0]) - (len(self.conn_mat[0]) - self.n_targets), 0:len(self.conn_mat[0]) - (len(self.conn_mat[0]) - self.n_targets)]]#inclusding and excluding inhibitory?
                zscores = stats.mstats.zscore(out_degrees)
                #out_degrees = [self.prob_conn * stats.norm.sf(a) for a in zscores]
                #cur_row_as_list = [1 if numpy.random.uniform(0, 1) <= x else 0 for x in out_degrees]
                cur_row_as_list = [1 if x < cutoff else 0 for x in zscores]
                adj_mat[i,:] = numpy.array(cur_row_as_list)'''''

                # create one row in the adjacency matrix
                # target_vals = [random.random() for x in range(0, self.n_targets)]
                # cur_row_as_list = [1 if x <= self.prob_conn else 0 for x in target_vals]
                # adj_mat[i, :] = numpy.array(cur_row_as_list)
                target_vals = numpy.random.rand(self.n_targets, 1)
                cur_row_as_list = [numpy.random.lognormal(-.64, .51)*self.multiplier if x <= self.prob_conn else 0 for x in target_vals]  # -.00005
                small_mat[i, :] = cur_row_as_list

            '''# flatten adjacency
            adj_mat = numpy.reshape(adj_mat, (1, self.n_input_units*self.n_targets)).tolist()
            # adj_mat = ",".join(str(i) for i in adj_mat)

            if not self.write_to_file(neuron_indices, spike_times, adj_mat):
                raise Exception()

            return True'''
            # flatten adjacency
            pool_1 = []
            if self.n_neurons != self.n_targets:
                target_indices = numpy.random.choice(self.n_neurons, self.n_targets, replace=False)
                curr = 0
                for c in target_indices:
                    adj_mat[:, c] = small_mat[:, curr]
                    curr += 1
            else:
                adj_mat = small_mat
                for aa in range(0, self.n_neurons):
                    if any(adj_mat[ee][aa] != 0 for ee in range(0, self.n_input_units)):
                        pool_1.append(aa)

            adj_mat = numpy.reshape(adj_mat, (1, self.n_input_units*self.n_neurons)).tolist()

            if not self.write_to_file(neuron_indices, spike_times, adj_mat):
                raise Exception()

            return [neuron_indices, spike_times, adj_mat]

        except Exception as e:
            print(e)
            return False

    def write_to_file(self, indices, spike_times, adj_mat):
        try:
            # write everything to a file
            # create directory if it doesn't exist

            (output_dir, output_f) = path.split(self.output_file)
            if not path.exists(output_dir):
                makedirs(output_dir)

            if self.separate_flag:
                # create three separate files - create names
                indices_filename = self.output_file.replace('.csv', '_indices.csv')
                spikes_filename = self.output_file.replace('.csv', '_spike_times.csv')
                adj_filename = self.output_file.replace('.csv', '_adj_mat.csv')

                with open(indices_filename, 'w', newline='') as f:
                    writerf = csv.writer(f, delimiter=',')
                    writerf.writerow(indices)

                with open(spikes_filename, 'w', newline='') as f:
                    writerf = csv.writer(f, delimiter=',')
                    writerf.writerow(spike_times)

                with open(adj_filename, 'w', newline='') as f:
                    writerf = csv.writer(f, delimiter=',')
                    writerf.writerow(adj_mat)
            else:
                with open(self.output_file, 'w', newline='') as f:
                    writef = csv.writer(f, delimiter=',')
                    writef.writerow(indices)
                    writef.writerow(spike_times)
                    writef.writerow(adj_mat)
            return True

        except Exception as e:
            return False


if __name__ == '__main__':
    try:
        num_trials = 1  # using different conn mat + may or may not use different input for each conn mat
        num_sims_per = 5  # using different initial voltages for each conn mat
        n_excite = 4000
        n_inhib = 1000

        n_input = 3000
        n_target = n_excite + n_inhib

        input_time = 1.050

        pei_w_multiplier = 1
        pei_p = 0.100

        rate_mean = 2.8
        rate_std = 0.3

        input_dictionary = {}
        for n in range(0, num_trials):
            poisson_input_loc = 'Inputs/sample_cyclical.csv'
            poisson_generator = PoissonInputGenerator(n_input, n_target, n_target, pei_p, rate_mean, rate_std,
                                                      input_time, poisson_input_loc, pei_w_multiplier, cyclical=True)
            input_dictionary[n] = poisson_generator.generate_inputs()
        numpy.save('Inputs/sample_cyclical.npy', input_dictionary)

    except Exception as e:
        print(e)
