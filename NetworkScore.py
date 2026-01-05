from brian2 import *
import numpy
import csv
from os.path import exists, isfile
import elephant
import neo
import quantities as pq
from scipy import stats


class NetworkScoreGenerator(object):
    def __init__(self, n_neurons, n_inhib, spike_trains_file, run_time, stim_in, dt, output_files, num_intervals=1):

        self.n_neurons = n_neurons
        self.n_inhib = n_inhib
        self.spike_trains_file = spike_trains_file

        if not exists(spike_trains_file) or not isfile(spike_trains_file):
            raise Exception('Invalid config spike_trains_file path!')

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

        # self.something = volt_stuff
        self.spike_trains = mat
        self.stim_in = stim_in  # stimulus input period
        self.run_time = run_time
        self.dt = dt  # specified in network simulator, i think currently .1
        self.output_files = output_files
        self.num_intervals = num_intervals

    def run_scores(self):
        try:

            self.last_spiked = self.last_spike()
            self.e_rate = self.rate_est(0, self.n_neurons - self.n_inhib, self.stim_in, self.last_spiked)
            self.i_rate = self.rate_est(self.n_neurons - self.n_inhib, self.n_neurons, self.stim_in, self.last_spiked)

            sco = [self.e_rate, self.i_rate, self.last_spiked]

            return sco

        except Exception as e:
            print(e)
            return False

    def run_synch_scores(self):
        try:
            last_e_synch = self.van_rossum()
            # fano = self.fano_factor()

            return [last_e_synch]  # [mean(fano)]

        except Exception as e:
            print(e)
            return False

    def rate_est(self, low, high, start, end):
        """
        Calculate firing rate
        Use once for e and once for i; if overall rate, start = self.stim_in, end = self.run_time
        :param low: lowest neuron index
        :param high: Highest neuron index
        :param start: Start time
        :param end: End time
        :return: Firing rate
        """
        try:
            num_spikes = 0
            max_values = numpy.zeros(high - low)

            for i in range(low, high):
                post_stim = [j for j in self.spike_trains[i] if (j > start) and (j <= end)]  # EDIT THIS LINE
                if not post_stim:
                    max_values[i - low] = 0
                else:
                    max_values[i - low] = numpy.amax(post_stim)
                num_spikes = num_spikes + len(post_stim)

            max_values = numpy.amax(max_values) - self.stim_in  # length of stimulus input period
            spikes_per = num_spikes / (high - low)
            rate = 1000 * spikes_per / max_values

            return rate

        except Exception as e:
            print(e)
            return False

    def last_spike(self):
        """
        Determine the length of the sustained activity based on the time of the last excitatory spike
        If there is more than 150 ms of excitatory inactivity, discard all spikes after them
        """
        try:
            e_spikes = self.spike_trains[0:int(0.8*self.n_neurons)]
            decision = True
            interval_length = 150
            flat_spikes = [item for sublist in e_spikes for item in sublist]
            if len(flat_spikes) != 0:
                flat_spikes.sort()
                max_max = flat_spikes[-1]
                stop_at = max_max
                qq = 0
                while decision and qq < (len(flat_spikes)-1):
                    if flat_spikes[qq+1] - flat_spikes[qq] > interval_length:
                        decision = False
                        stop_at = flat_spikes[qq]
                    qq += 1
                return stop_at
            else:
                return 0
        except Exception as e:
            print(e)
            return False

    def van_rossum(self):
        """
        Calculate synchrony of excitatory neurons during the last 150 ms of activity using the Van Rossum distance
        """
        try:
            last_e_spikes = self.spike_trains[0:int(0.8*self.n_neurons)]
            if self.last_spiked > 150:
                only_after = int(round(self.last_spiked-150))
                for mmm in range(0, int(0.8*self.n_neurons)):
                    last_e_spikes[mmm] = list(filter(lambda a: a > only_after, last_e_spikes[mmm]))
            spik = []
            for i in last_e_spikes:
                spik.append(neo.SpikeTrain(i, units='ms', t_stop=self.run_time))
            tau = 10 * pq.ms
            dist = elephant.spike_train_dissimilarity.van_rossum_distance(spik, tau, sort=False)
            print(mean(dist))
            return mean(dist)

        except Exception as e:
            print(e)
            return False

    def fano_factor(self):
        """
        Calculate value similar to the Fano factor over excitatory neurons to estimate synchrony quickly
        Compute over self.num_intervals intervals
        Use 10ms bins in each interval
        During each bin, calculate the variance of the number of spikes per neurons divided by the mean of the number of spikes per neuron
        The Fano factor during one interval is equal to the mean of the values calculated for each bin in it
        Often will use mean of intervals but divided in case we want to look at change in synchrony
        :return list of self.num_intervals elements (usually consider the last one only or the mean)
        """
        try:

            len_bins = 10
            n_fano = [0]*self.num_intervals
            len_interval = self.last_spiked/self.num_intervals
            n_bins = int(round(len_interval/len_bins))
            for m in range(0, self.num_intervals):
                fano_all = [0] * n_bins
                for i in range(0, n_bins):
                    spikes_per_neuron = [0]*(self.n_neurons-self.n_inhib)
                    for j in range(0, self.n_neurons-self.n_inhib):
                        for k in self.spike_trains[j]:
                            if len_interval*m + len_bins*i <= k < len_interval*m + len_bins*(i+1):
                                spikes_per_neuron[j] += 1
                    fano_all[i] = var(spikes_per_neuron)/mean(spikes_per_neuron)
                n_fano[m] = mean(fano_all)

            return n_fano

        except Exception as e:
            print(e)
            return False

    def spike_count(self):
        """
        Spike counts during 25ms bins to see if there is a prolonged period of decreased spike count that might indicate later truncation.
        Conducts a one sided z-test to see if there is a significant decrease during any of the periods
        :return: list of spike counts of length (last_spiked-input_length)/10, time of decrease, spike count when it decreased, flag (=True if decreased)
        """
        try:
            length = 25  # ms
            spike_counts = [0]*int(round((self.last_spiked-self.stim_in)/length))
            for i in range(0, len(spike_counts)):
                spikes_per_neuron = [0] * self.n_neurons
                for j in range(0, self.n_neurons):
                    for k in self.spike_trains[j]:
                        if self.stim_in + length * i <= k < self.stim_in + length * (i + 1):
                            spikes_per_neuron[j] += 1
                spike_counts[i] = sum(spikes_per_neuron)
            time_of_decrease = []
            count_at_decrease = []
            a = np.array(spike_counts)
            b = stats.zscore(a)
            c = stats.norm.cdf(b)
            flag = False
            for i in range(0, len(c)):
                if c[i] < 0.05 and a[i] < np.average(a):
                    flag = True
                    print('Between ' + str(self.stim_in + (i * length)) + 'ms and ' + str(self.stim_in + ((i+1) * length)) +
                          'ms there was a significant decrease in excitatory spike counts resulting in only ' + str(a[i]) + ' spikes.')
                    time_of_decrease.append(self.stim_in + (i * length))
                    count_at_decrease.append(a[i])
            return [spike_counts, time_of_decrease, count_at_decrease, flag]
        except Exception as e:
            print(e)
            return False
