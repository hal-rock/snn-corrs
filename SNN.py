from brian2 import *
import numpy
import logging
import csv
from logging.handlers import RotatingFileHandler
from os.path import exists, isfile
import sys
import time
import PoissonInputGenerator as poisson
import ConnMatGenerator as connmat
import NetworkScore as scorer
import os
import shutil
import pickle
import pandas as pd
import sklearn

csv.field_size_limit(2**30)


class NetworkSimulator(object):

    def __init__(self, p_log_path, p_log_level, simulation_time):

        # Initiate logger
        self.logger = logging.getLogger(__name__)
        self.set_logging(p_log_path, p_log_level)
        numpy.set_printoptions(threshold=sys.maxsize)

        # Define conn_mat and network
        self.conn_mat = None
        self.network = None

        self.simulation_time = simulation_time

        self.e_eqs = '''
            dvm/dt = (gL*(EL - vm) + gse*(Vse-vm) + gsi*(Vsi-vm)+ gP*(Vse-vm) + gL*DeltaT*exp((vm - VT)/DeltaT) - w)/C : volt
            dw/dt = (a*(vm - EL) - w)/tauw : amp
            dgse/dt=-gse/taue : siemens
            dgsi/dt=-gsi/taui : siemens
            dgP/dt=-gP/taup : siemens
            '''

        self.i_eqs = '''
            dvm/dt = (gL*(EL - vm) + gse*(Vse-vm) + gsi*(Vsi-vm)+ gP*(Vse-vm) + gL*DeltaT*exp((vm - VT)/DeltaT) - w)/C : volt
            dw/dt = (a*(vm - EL) - w)/tauw : amp
            dgse/dt=-gse/taue : siemens
            dgsi/dt=-gsi/taui : siemens
            dgP/dt=-gP/taup : siemens
            '''

    def set_logging(self, log_path, log_level):

        self.logger.setLevel(log_level)

        # Formatter object
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Handler objects
        file_handler = RotatingFileHandler(log_path, maxBytes=100000000, backupCount=10)
        file_handler.formatter = formatter
        file_handler.setLevel(log_level)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.formatter = formatter
        console_handler.setLevel(log_level)
        self.logger.addHandler(console_handler)

    def run_simulation(self, n_inputs, conn_mat_file, output_dir, input_file, run_num, n_neurons):
        try:
            self.n_input_units = n_inputs
            self.output_dir = output_dir
            self.conn_mat_file = conn_mat_file
            self.inputs_file_loc = input_file

            if not self.load_connectivity_mat():
                raise Exception('Failed to load connectivity matrix.')

            n_neurons = numpy.shape(self.conn_mat)[0]

            if not self.construct_network(n_neurons):
                raise Exception('Failed to construct model.')

            if not self.load_and_define_inputs(n_neurons):
                raise Exception('Failed to construct inputs.')

            if not self.define_monitors(n_neurons):
                raise Exception('Failed to define monitors.')

            self.network.run(self.simulation_time * second,
                             namespace={'C': self.C, 'gL': self.gL, 'Vsi': self.Vsi, 'Vse': self.Vse, 'EL': self.EL,
                                        'Vr': self.Vr, 'VT': self.VT, 'DeltaT': self.DeltaT,
                                        'Vcut': self.VT + 5 * self.DeltaT, 'taue': self.taue, 'taui': self.taui,
                                        'tauw': self.tauw, 'taup': self.taup, 'a': self.a, 'b': self.b,
                                        'threshold': 'vm >Vcut', 'reset': "vm=EL;w+=b", 'refractory': 2 * ms,
                                        'e_neurons': self.e_neurons, 'i_neurons': self.i_neurons,
                                        'i_spike_monitor': self.i_spike_monitor,
                                        'e_spike_monitor': self.e_spike_monitor})

            writing = self.write_to_file(run_num, n_neurons)
            if not writing:
                raise Exception('Failed to write file...')

            return writing

        except Exception as e:
            self.logger.error(e)
            return False

    def construct_network(self, n_nrn):
        try:
            self.logger.info('Building model...')

            self.C = .281 * nF
            self.gL = 30 * nS
            self.EL = -70.6 * mV
            self.Vr = -70.6 * mV
            self.VT = -50.4 * mV
            self.Vse = 0 * mV
            self.Vsi = -75 * mV
            self.DeltaT = 2 * mV
            self.Vcut = self.VT + 5 * self.DeltaT
            self.taue = 10 * ms
            self.taui = 3 * ms
            self.taup = 10 * ms
            self.tauw, self.a, self.b = 144 * ms, 4 * nS, 80.5 * pA

            # Define neuron groups; rq: i=no firing rate adaptation in reset and in equations
            self.e_neurons = NeuronGroup(n_nrn - (n_nrn // 5), model=self.e_eqs, threshold='vm>Vcut',
                                         reset="vm=EL;w+=b", refractory=2 * ms, method='euler')
            self.i_neurons = NeuronGroup((n_nrn // 5), model=self.i_eqs, threshold='vm>Vcut', reset="vm=EL;w+=b",
                                         refractory=2 * ms, method='euler')

            # Define initial membrane potential
            self.e_initial_volt = numpy.random.normal(-65, 5, n_nrn - (n_nrn // 5))
            self.i_initial_volt = numpy.random.normal(-65, 5, (n_nrn // 5))
            self.e_neurons.vm = self.e_initial_volt * mV
            self.i_neurons.vm = self.i_initial_volt * mV

            # Define synapses
            ee_model_synapses = Synapses(self.e_neurons, self.e_neurons,
                                         model='''weights : siemens''', on_pre='''gse+=weights''')
            ii_model_synapses = Synapses(self.i_neurons, self.i_neurons,
                                         model='''weights : siemens''', on_pre='''gsi+=weights''')
            ie_model_synapses = Synapses(self.i_neurons, self.e_neurons,
                                         model='''weights : siemens''', on_pre='''gsi+=weights''')
            ei_model_synapses = Synapses(self.e_neurons, self.i_neurons,
                                         model='''weights : siemens''', on_pre='''gse+=weights''')

            # Create synapses
            ee_model_synapses.connect(True)
            ii_model_synapses.connect(True)
            ie_model_synapses.connect(True)
            ei_model_synapses.connect(True)

            # Determine synapses weight from connectivity matrix
            self.conn_mat = self.conn_mat * nsiemens
            ee_model_synapses.weights = numpy.reshape(
                self.conn_mat[0:n_nrn - (n_nrn // 5), 0:n_nrn - (n_nrn // 5)],
                (1, pow(n_nrn - (n_nrn // 5), 2)))
            ii_model_synapses.weights = numpy.reshape(
                self.conn_mat[n_nrn - (n_nrn // 5):n_nrn, n_nrn - (n_nrn // 5):n_nrn],
                (1, pow((n_nrn // 5), 2)))
            ei_model_synapses.weights = numpy.reshape(
                self.conn_mat[0:n_nrn - (n_nrn // 5), n_nrn - (n_nrn // 5):n_nrn],
                (1, (n_nrn // 5) * (n_nrn - (n_nrn // 5))))
            ie_model_synapses.weights = numpy.reshape(
                self.conn_mat[n_nrn - (n_nrn // 5):n_nrn, 0:n_nrn - (n_nrn // 5)],
                (1, (n_nrn // 5) * (n_nrn - (n_nrn // 5))))

            # Pack everything into a Network object
            self.network = Network(self.e_neurons, self.i_neurons, ee_model_synapses,
                                   ii_model_synapses, ie_model_synapses, ei_model_synapses)
            print('Done building network')

            return True

        except Exception as e:
            self.logger.error(e)
            return False

    def load_connectivity_mat(self):
        try:
            self.logger.info('Loading connectivity matrix...')

            if not exists(self.conn_mat_file) or not isfile(self.conn_mat_file):
                raise Exception('Invalid conn_mat_file path!')

            with open(self.conn_mat_file, 'r', newline='') as csvf:
                readc = csv.reader(csvf, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
                mat = []
                for row in readc:
                    mat.append(row)

            self.conn_mat = mat

            return True

        except Exception as e:
            self.logger.error(e)
            return False

    def load_and_define_inputs(self, n_nrn):
        try:
            self.logger.info('Loading inputs and constructing input synapses...')

            n_excite = int(4*n_nrn/5)
            n_inhib = int(n_nrn/5)

            [poisson_inds, input_spikes, flat_adj] = self.inputs_file_loc
            flat_adj = flat_adj[0]
            print("Done reading input spikes")
            input_spikes = input_spikes * ms
            input_group = SpikeGeneratorGroup(self.n_input_units, poisson_inds, input_spikes)
            input_synapses_i = Synapses(input_group, self.i_neurons,
                                        model='''weights : siemens''', on_pre='''gP += weights''')
            input_synapses_e = Synapses(input_group, self.e_neurons,
                                        model='''weights : siemens''', on_pre='''gP += weights''')
            input_synapses_i.connect(True)
            input_synapses_e.connect(True)
            temp1 = flat_adj[:self.n_input_units * n_excite]
            input_synapses_e.weights = temp1 * nS
            temp2 = flat_adj[self.n_input_units * n_excite:self.n_input_units * n_nrn]
            input_synapses_i.weights = temp2 * nS

            # Add to network - check that self.network is not None before
            if not self.network:
                raise Exception('Network object hasn\'t been created yet!')
            self.network.add(input_group)
            self.network.add(input_synapses_i)
            self.network.add(input_group)
            self.network.add(input_synapses_e)

            return True

        except Exception as e:
            self.logger.error(e)
            return False

    def define_monitors(self, n_nrn):
        try:
            self.logger.info('Defining Monitors...')

            self.e_spike_monitor = SpikeMonitor(self.e_neurons)
            self.network.add(self.e_spike_monitor)
            self.i_spike_monitor = SpikeMonitor(self.i_neurons)
            self.network.add(self.i_spike_monitor)

            return True

        except Exception as e:
            self.logger.error(e)
            return False

    def write_to_file(self, run_num, n_neurons):
        try:
            self.logger.info('Writing Files...')

            with open(self.output_dir + '/spike_monitor' + run_num + '.csv', 'w', newline='') as f:
                writerf = csv.writer(f, delimiter=',')

                e_spike_trains = self.e_spike_monitor.spike_trains()
                for i in e_spike_trains:
                    writerf.writerow(e_spike_trains[i])

                i_spike_trains = self.i_spike_monitor.spike_trains()
                for r in i_spike_trains:
                    writerf.writerow(i_spike_trains[r])

            return True

        except Exception as e:
            self.logger.error(e)
            return False


def run_sims(ii_p, ie_p, ei_p, ee_p, n_excite, n_inhib, directory_name, num_sims_per, poisson_input_loc,
             n_input, input_length, w_mu, w_sd, simulation_time):
    try:
        n_neurons = n_excite + n_inhib

        start = time.time()

        engine = NetworkSimulator('Logs/current_log.log', 'DEBUG', simulation_time)
        engine.logger.info('Generating connectivity matrix...' + str(ee_p) + '/' + str(ei_p) + '/' + str(ie_p) + '/' +
                           str(ii_p) + '/' + str(n_neurons))

        connmat_generator = connmat.ConnectivityMatrixGenerator(n_excite, n_inhib, ii_p, ei_p, ie_p, ee_p,
                                                                w_mu, w_sd, directory_name)

        ret = [0, 0, 0, 0]
        synch = 0

        if not connmat_generator.run_generator():
            raise Exception('Failed to generate connectivity matrix.')

        for i in range(0, num_sims_per):
            engine.logger.info('RUNNING SIMULATION...')

            result = engine.run_simulation(n_input, directory_name + '/weight_mat.csv',
                                           directory_name, poisson_input_loc, str(i), n_neurons)

            if not result:
                raise Exception('Failed to run simulation.')

            engine.logger.info('Scoring network...')

            if input_length == simulation_time:
                input_length = 0

            network_scorer = scorer.NetworkScoreGenerator(n_excite+n_inhib, n_inhib,
                                                          directory_name + '/spike_monitor' + str(i) + '.csv',
                                                          simulation_time*1000, input_length*1000, .1,
                                                          directory_name + '/scoring.csv', num_intervals=5)

            ret = network_scorer.run_scores()
            print(ret)
            synch = network_scorer.run_synch_scores()
            print(synch)
            if input_length != 0:
                if ret[2] > 1040:
                    with open('brief_input_classifier/SVM_PT_scaler_with_sklearn_0.22.1.pkl', 'rb') as scl:
                        scl_loaded =pickle.load(scl)
                    with open('brief_input_classifier/SVM_classifier_with_sklearn_0.22.1.pkl', 'rb') as fid:
                        clf_loaded = pickle.load(fid)
                    X_test = pd.DataFrame([[ret[0], ret[1], synch[0]]], columns=['e_fr', 'i_fr', 'synch'])
                    X_test_scaled = scl_loaded.transform(X_test)
                    X_test_scaled = pd.DataFrame(X_test_scaled, columns=['e_fr_scaled', 'i_fr_scaled', 'synch_scaled'])
                    y_predict = clf_loaded.predict(X_test_scaled)
                    print('classification done')
                    if y_predict == 0:
                        ret[2] = 0
            print(ret[2])

            with open(directory_name + '/scoring.csv', 'a') as f:
                writerf = csv.writer(f, delimiter=',')
                writerf.writerow(ret+synch)

            end = time.time()
            print(end - start)

            engine.logger.info('Done')

        return ret + synch

    except Exception as e:
        engine.logger.error(e)
        print(e)
        return False


def run_on_existing(n_excite, n_inhib, weight_mat, directory_name, num_sims_per,
                    poisson_input_loc, n_input, input_length, simulation_time):

    try:

        for num in range(0, num_sims_per):  # num_sims_per is number of simulations on given conn_mat

            n_neurons = n_excite + n_inhib

            engine.logger.info('RUNNING SIMULATION...')

            if not engine.run_simulation(n_input, weight_mat, directory_name, poisson_input_loc, str(num), n_neurons):
                raise Exception('Failed to run simulation.')

            if input_length == simulation_time:
                input_length = 0

            network_scorer = scorer.NetworkScoreGenerator(n_excite + n_inhib, n_inhib,
                                                          directory_name + '/spike_monitor' + str(i) + '.csv',
                                                          simulation_time * 1000, input_length * 1000, .1,
                                                          directory_name + '/scoring.csv', num_intervals=5)

            ret = network_scorer.run_scores()
            print(ret)
            synch = network_scorer.run_synch_scores()
            print(synch)
            if input_length != 0:
                if ret[2] > 1040:
                    with open('brief_input_classifier/SVM_PT_scaler_with_sklearn_0.22.1.pkl', 'rb') as scl:
                        scl_loaded = pickle.load(scl)
                    with open('brief_input_classifier/SVM_classifier_with_sklearn_0.22.1.pkl', 'rb') as fid:
                        clf_loaded = pickle.load(fid)
                    X_test = pd.DataFrame([[ret[0], ret[1], synch[0]]], columns=['e_fr', 'i_fr', 'synch'])
                    X_test_scaled = scl_loaded.transform(X_test)
                    X_test_scaled = pd.DataFrame(X_test_scaled,
                                                 columns=['e_fr_scaled', 'i_fr_scaled', 'synch_scaled'])
                    y_predict = clf_loaded.predict(X_test_scaled)
                    print('classification done')
                    if y_predict == 0:
                        ret[2] = 0
            print(ret[2])

            with open(directory_name + '/scoring.csv', 'a') as f:
                writerf = csv.writer(f, delimiter=',')
                writerf.writerow(ret + synch)

            engine.logger.info('Done')

        return ret + synch

    except Exception as e:
        engine.logger.error(e)
        print(e)
        return False


if __name__ == '__main__':
    try:

        num_trials = 1  # using different conn mat + may or may not use different input for each conn mat
        num_sims_per = 5  # using different initial voltages for each conn mat
        n_excite = 4000
        n_inhib = 1000

        n_input = 3000
        n_target = n_excite + n_inhib
        input_time = 0.300

        pei_w_multiplier = [1]
        pei_p = [0.100]
        p_ee = numpy.arange(0.10, 0.30 + 0.02, 0.02)
        p_ei = numpy.arange(0.10, 0.30 + 0.02, 0.02)
        p_ie = numpy.arange(0.10, 0.30 + 0.02, 0.02)
        p_ii = numpy.linspace(0.20, 0.40, 11)

        w_mu = -0.64
        w_sd = 0.51

        rate_mean = 2.8
        rate_std = 0.3

        score = numpy.zeros((len(pei_w_multiplier) * len(pei_p) * len(p_ee) * len(p_ei) * len(p_ie) * len(p_ii) * num_trials * num_sims_per, 11))
        count = -1

        simulation_time = 1.05
        engine = NetworkSimulator('Logs/current_log.log', 'DEBUG', simulation_time)

        for p in pei_p:
            for m in pei_w_multiplier:

                input_dictionary = {}
                for n in range(0, num_trials):
                    print("Generating Poisson Input")
                    poisson_input_loc = 'output/poisson_inputs_' + str(n) + '.csv'
                    poisson_generator = poisson.PoissonInputGenerator(n_input, n_target, n_excite + n_inhib, p,
                                                                      rate_mean, rate_std, input_time,
                                                                      poisson_input_loc, m, cyclical=False)
                    input_dictionary[n] = poisson_generator.generate_inputs()

                # Can also use saved inputs e.g.
                # input_dictionary = np.load('Inputs/sample_brief.npy', allow_pickle=True).item()

                for ee_p in p_ee:
                    for ei_p in p_ei:
                        for ie_p in p_ie:
                            for ii_p in p_ii:

                                directory_name = 'grid_search_random_short_input/pee_' + str(round(ee_p, 4)) +\
                                                 '_pei_' + str(round(ei_p, 4)) + '_pie_' + str(round(ie_p, 4)) +\
                                                 '_pii_' + str(round(ii_p, 4)) + '_simulation_time_' +\
                                                 str(simulation_time)

                                if not os.path.exists(directory_name):
                                    os.mkdir(directory_name)

                                # To use different inputs for each parameter combination, generate them here

                                for i in range(0, num_trials):
                                    results = run_sims(ii_p, ie_p, ei_p, ee_p, n_excite, n_inhib, directory_name,
                                                       num_sims_per, input_dictionary[i], n_input, input_time,
                                                       w_mu, w_sd, simulation_time)

                                    with open(directory_name + '/scoring.csv', 'r', newline='') as f:
                                        read_f = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
                                        for j in range(0, num_sims_per):
                                            count += 1
                                            line = next(read_f)
                                            score[count][0] = line[0]
                                            score[count][1] = line[1]
                                            score[count][2] = line[2]
                                            score[count][3] = line[6]
                                            score[count][4] = p
                                            score[count][5] = m
                                            score[count][6] = ee_p
                                            score[count][7] = ei_p
                                            score[count][8] = ie_p
                                            score[count][9] = ii_p
                                            score[count][10] = j
                                            numpy.save('grid_search_random_brief_input/SummaryScores', score)

                                    shutil.rmtree(directory_name)

        numpy.save('grid_search_random_brief_input/SummaryScores', score)

    except Exception as e:
        print(e)
