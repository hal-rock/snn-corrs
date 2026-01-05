from brian2 import *
import numpy as np
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
import matplotlib.pyplot as plt


# my new faster functions
from input_generation import generate_input_spikes
# connmat contains the connectivity functions

csv.field_size_limit(2**30)


class NetworkSimulator(object):

    def __init__(self, p_log_path, p_log_level, simulation_time, n_excite, n_inhib, Vsi=-75):

        # Initiate logger
        #self.logger = logging.getLogger(__name__)
        #self.set_logging(p_log_path, p_log_level)
        np.set_printoptions(threshold=sys.maxsize)

        # Define conn_mat and network
        self.conn_mat = None
        self.network = None

        # hack for sweeping thru inhibitory reversal potential
        self.Vsi = Vsi * mV

        self.simulation_time = simulation_time
        self.n_excite = n_excite
        self.n_inhib = n_inhib

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

    # def set_logging(self, log_path, log_level):

    #     self.logger.setLevel(log_level)

    #     # Formatter object
    #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    #     # Handler objects
    #     file_handler = RotatingFileHandler(log_path, maxBytes=100000000, backupCount=10)
    #     file_handler.formatter = formatter
    #     file_handler.setLevel(log_level)
    #     self.logger.addHandler(file_handler)

    #     console_handler = logging.StreamHandler()
    #     console_handler.formatter = formatter
    #     console_handler.setLevel(log_level)
    #     self.logger.addHandler(console_handler)

    def run_simulation(self, conn_mat, output_dir, inputs, input_conn, run_num, n_neurons):
        try:
            self.output_dir = output_dir
            self.conn_mat = conn_mat
            #self.inputs_file_loc = input_file

            # if not self.load_connectivity_mat():
            #     raise Exception('Failed to load connectivity matrix.')

            n_neurons = np.shape(self.conn_mat)[0]

            if not self.construct_network(n_neurons):
                raise Exception('Failed to construct model.')

            if not self.load_and_define_inputs(input_conn, inputs):
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
                                        'e_spike_monitor': self.e_spike_monitor})#,
                                        #'v_monitor': self.v_monitor}) # addition -- necessary?
            # print(np.mean(self.v_monitor.vm))
            # plt.figure()
            # plt.hist(np.array(self.v_monitor.vm).flatten(), bins=100)
            # plt.show()


            writing = self.write_to_file_fast(run_num)#, n_neurons)
            if not writing:
                raise Exception('Failed to write file...')

            return writing

        except Exception as e:
            #self.logger.error(e)
            return False

    def construct_network(self, n_nrn):
        try:
            #self.logger.info('Building model...')

            self.C = .281 * nF
            self.gL = 30 * nS
            self.EL = -70.6 * mV
            self.Vr = -70.6 * mV
            self.VT = -50.4 * mV
            self.Vse = 0 * mV
            #self.Vsi = -75 * mV
            self.DeltaT = 2 * mV
            self.Vcut = self.VT + 5 * self.DeltaT
            self.taue = 10 * ms
            self.taui = 3 * ms # 7
            self.taup = 10 * ms
            self.tauw, self.a, self.b = 144 * ms, 4 * nS, 80.5 * pA

            # Define neuron groups; rq: i=no firing rate adaptation in reset and in equations
            self.e_neurons = NeuronGroup(self.n_excite, model=self.e_eqs, threshold='vm>Vcut',
                                         reset="vm=EL;w+=b", refractory=2 * ms, method='euler')
            self.i_neurons = NeuronGroup(self.n_inhib, model=self.i_eqs, threshold='vm>Vcut', reset="vm=EL;w+=b",
                                         refractory=2 * ms, method='euler')

            # Define initial membrane potential
            self.e_initial_volt = np.random.normal(-65, 5, self.n_excite)
            self.i_initial_volt = np.random.normal(-65, 5, self.n_inhib)
            self.e_neurons.vm = self.e_initial_volt * mV
            self.i_neurons.vm = self.i_initial_volt * mV

            # Define synapses
            ee_model_synapses = Synapses(self.e_neurons, self.e_neurons,
                                         model='''weights : siemens''', on_pre='''gse+=weights''')
            ii_model_synapses = Synapses(self.i_neurons, self.i_neurons,
                                         model='''weights : siemens''', on_pre='''gsi+=weights''')
            ie_model_synapses = Synapses(self.e_neurons, self.i_neurons,
                                         model='''weights : siemens''', on_pre='''gse+=weights''')
            ei_model_synapses = Synapses(self.i_neurons, self.e_neurons,
                                         model='''weights : siemens''', on_pre='''gsi+=weights''')

            # Create synapses
            # first slice up the matrix
            ee_slice = self.conn_mat[:self.n_excite, :self.n_excite]
            ie_slice = self.conn_mat[self.n_excite:, :self.n_excite]
            ei_slice = self.conn_mat[:self.n_excite, self.n_excite:]
            ii_slice = self.conn_mat[self.n_excite:, self.n_excite:]
            ee_model_synapses.connect(i=np.where(ee_slice > 0)[1], j=np.where(ee_slice > 0)[0])
            ie_model_synapses.connect(i=np.where(ie_slice > 0)[1], j=np.where(ie_slice > 0)[0])
            ei_model_synapses.connect(i=np.where(ei_slice > 0)[1], j=np.where(ei_slice > 0)[0])
            ii_model_synapses.connect(i=np.where(ii_slice > 0)[1], j=np.where(ii_slice > 0)[0])


            # Determine synapses weight from connectivity matrix
            #self.conn_mat = self.conn_mat * nsiemens
            ee_model_synapses.weights = ee_slice[ee_slice > 0] * nS
            ie_model_synapses.weights = ie_slice[ie_slice > 0] * nS
            ei_model_synapses.weights = ei_slice[ei_slice > 0] * nS
            ii_model_synapses.weights = ii_slice[ii_slice > 0] * nS

            # Pack everything into a Network object
            self.network = Network(self.e_neurons, self.i_neurons, ee_model_synapses,
                                   ii_model_synapses, ie_model_synapses, ei_model_synapses)
            #print(self.e_neurons.gse[200])

            print('Done building network')

            return True

        except Exception as e:
            #self.logger.error(e)
            return False

    def load_connectivity_mat(self):
        try:
            #self.logger.info('Loading connectivity matrix...')

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
            #self.logger.error(e)
            return False

    def load_and_define_inputs(self, input_conn, inputs):
        try:
            #self.logger.info('Loading inputs and constructing input synapses...')

            #[poisson_inds, input_spikes, flat_adj] = self.inputs_file_loc
            #flat_adj = flat_adj[0]
            print("Done reading input spikes")
            poisson_inds, input_spikes = inputs # packed indices + times in a tuple
            input_spikes = input_spikes * ms
            input_group = SpikeGeneratorGroup(input_conn.shape[0], poisson_inds, input_spikes)
            input_synapses_i = Synapses(input_group, self.i_neurons,
                                        model='''weights : siemens''', on_pre='''gP += weights''')
            input_synapses_e = Synapses(input_group, self.e_neurons,
                                        model='''weights : siemens''', on_pre='''gP += weights''')
            input_synapses_i.connect(i=np.nonzero(input_conn[:, self.n_excite:])[0], j=np.nonzero(input_conn[:, self.n_excite:])[1])
            input_synapses_e.connect(i=np.nonzero(input_conn[:, :self.n_excite])[0], j=np.nonzero(input_conn[:, :self.n_excite])[1])
            input_synapses_i.weights = input_conn[:, self.n_excite:][input_conn[:, self.n_excite:] > 0] * nS
            input_synapses_e.weights = input_conn[:, :self.n_excite][input_conn[:, :self.n_excite] > 0] * nS

            # Add to network - check that self.network is not None before
            if not self.network:
                raise Exception('Network object hasn\'t been created yet!')
            self.network.add(input_group)
            self.network.add(input_synapses_i)
            self.network.add(input_group)
            self.network.add(input_synapses_e)

            return True

        except Exception as e:
            #self.logger.error(e)
            return False

    def define_monitors(self, n_nrn):
        self.e_spike_monitor = SpikeMonitor(self.e_neurons)
        self.network.add(self.e_spike_monitor)
        self.i_spike_monitor = SpikeMonitor(self.i_neurons)
        self.network.add(self.i_spike_monitor)
        # rand_indices = np.random.randint(0, 4000, 20)
        # self.v_monitor = StateMonitor(source=self.e_neurons, variables='vm', record=rand_indices)
        # self.network.add(self.v_monitor)

        return True


    def write_to_file(self, run_num, n_neurons):
        try:
            #self.logger.info('Writing Files...')

            with open(self.output_dir + '/spike_monitor' + str(run_num) + '.csv', 'w', newline='') as f:
                writerf = csv.writer(f, delimiter=',')

                e_spike_trains = self.e_spike_monitor.spike_trains()
                for i in e_spike_trains:
                    writerf.writerow(e_spike_trains[i])

                i_spike_trains = self.i_spike_monitor.spike_trains()
                for r in i_spike_trains:
                    writerf.writerow(i_spike_trains[r])

            return True

        except Exception as e:
            #self.logger.error(e)
            return False

    def write_to_file_fast(self, run_num):
        fname = f'{self.output_dir}/spike_monitor{run_num}.npz'
        # put E and I idxs/times together first
        idxs = np.concatenate([self.e_spike_monitor.i, self.i_spike_monitor.i + self.n_excite])
        times = np.concatenate([self.e_spike_monitor.t / ms, self.i_spike_monitor.t / ms])
        np.savez(fname, idxs=idxs, times=times)
        return True


def run_sims(input_conn, inputs, recur_conn, n_excite, n_inhib, directory_name, num_sims_per,
             simulation_time, Vsi=-75):
    n_neurons = n_excite + n_inhib

    start = time.time()

    engine = NetworkSimulator('Logs/current_log.log', 'DEBUG', simulation_time, n_excite=n_excite, n_inhib=n_inhib, Vsi=Vsi)

    for i in range(0, num_sims_per):
        #engine.logger.info('RUNNING SIMULATION...')

        result = engine.run_simulation(recur_conn, directory_name, inputs, input_conn, i, n_neurons)

        if not result:
            raise Exception('Failed to run simulation.')

        end = time.time()
        print(end - start)

        #engine.logger.info('Done')

    return 0 #ret + synch




if __name__ == '__main__':
    # try:

    n_trials = 200  # using different conn mat + may or may not use different input for each conn mat
    num_sims_per = 1  # using different initial voltages for each conn mat
    n_excite = 4000
    n_inhib = 1000
    n_cells = n_excite + n_inhib

    n_input = 3000
    n_target = n_excite + n_inhib
    #input_time = 0.300
    #input_time = 1.0 #0.1 #1.0 #0.1 # 1.0 # continuous input to start

    pei_w_multiplier = 1
    pei_p = 0.100
    # individual connectivity values from... something
    ee_p = 0.28 # back to orig numbers now
    ei_p = 0.24 
    ie_p = 0.28 # fixing ei, ie mixup
    ii_p = 0.2 
    # ranges searched over in tarek's example script
    # p_ee = np.arange(0.10, 0.30 + 0.02, 0.02)
    # p_ei = np.arange(0.10, 0.30 + 0.02, 0.02)
    # p_ie = np.arange(0.10, 0.30 + 0.02, 0.02)
    # p_ii = np.linspace(0.20, 0.40, 11)

    w_mu = -0.64
    w_sd = 0.51

    rate_mean = 2.8
    rate_std = 0.3

    simulation_time = 1.05
    # engine line taken from here
    #engine = NetworkSimulator('Logs/current_log.log', 'DEBUG', simulation_time, n_excite, n_inhib) # seemingly not actually used?

    # to start, I want to run multiple trials (input spike and voltage draws) from one stimulus (input rate vector)
    # all of these will be on the same connectivity matrix (input and recurrent), so fix those here
    input_conn = connmat.gen_input_conn(n_input, n_cells, pei_p, w_mu, w_sd) # are mu/sigma same as recurrent correct?
    #recur_conn = connmat.gen_recurrent_conn(n_excite, n_inhib, ii_p, ei_p, ie_p, ee_p, w_mu, w_sd, None, i_multiplier=7)

    for input_time in np.arange(0.1, 1.1, 0.1):
        for kappa in np.arange(0.0, 2.2, 0.2):
            recur_conn = connmat.gen_ring_conn(n_excite, n_inhib, ii_p, ei_p, ie_p, ee_p, w_mu, w_sd, kappa, None, i_multiplier=10)
            save_dir = f'/mnt/eight1/tarek_runs/inpt_t{input_time}/kappa{kappa}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.savez(f'{save_dir}/conn_mats.npz', inpt=input_conn, recurrent=recur_conn)
            # for each trial, I want to generate its own input spikes, on the fly I guess
            # sike actually that's batched
            n_stimuli = 3
            for s in range(0, n_stimuli): # stimuli are diff rate vectors for each one, but connectivity is the same
                input_rates = np.random.lognormal(mean = rate_mean, sigma=rate_std, size=n_input) # fixed input rates
                inpt_times, inpt_spikes = generate_input_spikes(input_rates, n_trials, input_time*1000, output_file=None)
                if not os.path.exists(f'{save_dir}/stim{s}/'):
                    os.makedirs(f'{save_dir}/stim{s}/')
                np.savez(f'{save_dir}/stim{s}/input_times.npz', *inpt_times) # this runs and probably saves things correctly :P
                np.savez(f'{save_dir}/stim{s}/input_spikes.npz', *inpt_spikes)

                for n in range(n_trials):
                    print(f'running stim {s} trial {n}')
                    # ditching the "sims per" notation cause it's confusing. could be used for diff voltages but same input spikes (but largely idc abt that)
                    # ideally, would compile the network once, before this loop, and run it the same for all trials (and later for stimuli)
                    # but don't prematurely optimize
                    directory_name = f'{save_dir}/stim{s}/trial{n}'
                    if not os.path.exists(directory_name):
                        os.makedirs(directory_name)
                    results = run_sims(input_conn, (inpt_times[n], inpt_spikes[n]), recur_conn, n_excite, n_inhib, directory_name,
                                        num_sims_per, simulation_time)

    # except Exception as e:
    #     print(e)
