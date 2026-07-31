"""
Parallelized version of fastSNN.py using Brian2 standalone mode.
Runs multiple independent simulations across CPU cores.
"""
from brian2 import *
import numpy as np
import os
import shutil
import time
import multiprocessing as mp
from functools import partial

# Local imports
from input_generation import generate_input_spikes, generate_rate_vector
import ConnMatGenerator as connmat


def run_single_trial(trial_args, config):
    """
    Run a single simulation trial. Designed to be called from a worker process.

    Args:
        trial_args: tuple of (trial_id, input_times, input_spikes)
        config: dict with simulation configuration (connectivity, params, etc.)

    Returns:
        tuple of (trial_id, success)
    """
    trial_id, inpt_times, inpt_spikes = trial_args

    # Unpack config
    input_conn = config['input_conn']
    recur_conn = config['recur_conn']
    n_excite = config['n_excite']
    n_inhib = config['n_inhib']
    simulation_time = config['simulation_time']
    output_dir = config['output_dir']
    Vsi = config.get('Vsi', -75)

    # Get process-specific directory for standalone compilation
    pid = os.getpid()
    standalone_dir = f'/tmp/brian_standalone_{pid}'

    # Set device for this process (only on first call)
    if not hasattr(run_single_trial, '_device_set'):
        set_device('cpp_standalone', directory=standalone_dir, build_on_run=True)
        run_single_trial._device_set = True
    #set_device('cpp_standalone', directory=standalone_dir)#, build_on_run=True)
    #run_single_trial._device_set = True

    try:
        # Build and run the network
        result = _build_and_run_network(
            input_conn=input_conn,
            recur_conn=recur_conn,
            inputs=(inpt_times, inpt_spikes),
            n_excite=n_excite,
            n_inhib=n_inhib,
            simulation_time=simulation_time,
            output_dir=output_dir,
            trial_id=trial_id,
            Vsi=Vsi
        )

        # Reinitialize device for next run in this process
        device.reinit()
        device.activate(directory=standalone_dir, build_on_run=True)

        return (trial_id, result, pid)

    except Exception as e:
        print(f"Trial {trial_id} failed: {e}")
        import traceback
        traceback.print_exc()
        device.reinit()
        device.activate(directory=standalone_dir, build_on_run=True)
        return (trial_id, False, pid)


def _build_and_run_network(input_conn, recur_conn, inputs, n_excite, n_inhib,
                           simulation_time, output_dir, trial_id, Vsi=-75):
    """
    Build and run a single simulation. Internal function.
    """
    # Neuron parameters - these will be passed via namespace
    C = .281 * nF
    gL = 30 * nS
    EL = -70.6 * mV
    Vr = -70.6 * mV
    VT = -50.4 * mV
    Vse = 0 * mV
    Vsi_val = Vsi * mV
    DeltaT = 2 * mV
    Vcut = VT + 5 * DeltaT
    taue = 10 * ms
    taui = 3 * ms
    taup = 10 * ms
    tauw, a, b = 144 * ms, 4 * nS, 80.5 * pA

    # Equations - use explicit namespace for all constants
    eqs = '''
        dvm/dt = (gL*(EL - vm) + gse*(Vse-vm) + gsi*(Vsi-vm)+ gP*(Vse-vm) + gL*DeltaT*exp((vm - VT)/DeltaT) - w)/C : volt
        dw/dt = (a*(vm - EL) - w)/tauw : amp
        dgse/dt=-gse/taue : siemens
        dgsi/dt=-gsi/taui : siemens
        dgP/dt=-gP/taup : siemens
    '''

    # Namespace for equations - must pass all constants explicitly
    namespace = {
        'C': C, 'gL': gL, 'EL': EL, 'Vr': Vr, 'VT': VT, 'Vse': Vse,
        'Vsi': Vsi_val, 'DeltaT': DeltaT, 'Vcut': Vcut,
        'taue': taue, 'taui': taui, 'taup': taup, 'tauw': tauw,
        'a': a, 'b': b
    }

    # Create neuron groups
    e_neurons = NeuronGroup(n_excite, model=eqs, threshold='vm>Vcut',
                            reset="vm=EL;w+=b", refractory=2*ms, method='euler',
                            namespace=namespace)
    i_neurons = NeuronGroup(n_inhib, model=eqs, threshold='vm>Vcut',
                            reset="vm=EL;w+=b", refractory=2*ms, method='euler',
                            namespace=namespace)

    # Initial membrane potentials - use string-based initialization for standalone
    e_neurons.vm = 'rand()*10*mV - 70*mV'  # ~U(-70, -60) mV, roughly matches N(-65, 5)
    i_neurons.vm = 'rand()*10*mV - 70*mV'

    # Slice connectivity matrix
    ee_slice = recur_conn[:n_excite, :n_excite]
    ie_slice = recur_conn[n_excite:, :n_excite]
    ei_slice = recur_conn[:n_excite, n_excite:]
    ii_slice = recur_conn[n_excite:, n_excite:]

    # Create synapses
    ee_syn = Synapses(e_neurons, e_neurons, model='weights : siemens', on_pre='gse+=weights')
    ii_syn = Synapses(i_neurons, i_neurons, model='weights : siemens', on_pre='gsi+=weights')
    ie_syn = Synapses(e_neurons, i_neurons, model='weights : siemens', on_pre='gse+=weights')
    ei_syn = Synapses(i_neurons, e_neurons, model='weights : siemens', on_pre='gsi+=weights')

    ee_syn.connect(i=np.where(ee_slice > 0)[1], j=np.where(ee_slice > 0)[0])
    ie_syn.connect(i=np.where(ie_slice > 0)[1], j=np.where(ie_slice > 0)[0])
    ei_syn.connect(i=np.where(ei_slice > 0)[1], j=np.where(ei_slice > 0)[0])
    ii_syn.connect(i=np.where(ii_slice > 0)[1], j=np.where(ii_slice > 0)[0])

    ee_syn.weights = ee_slice[ee_slice > 0] * nS
    ie_syn.weights = ie_slice[ie_slice > 0] * nS
    ei_syn.weights = ei_slice[ei_slice > 0] * nS
    ii_syn.weights = ii_slice[ii_slice > 0] * nS

    # Input spikes
    poisson_inds, input_spikes = inputs
    input_spikes_with_units = input_spikes * ms
    input_group = SpikeGeneratorGroup(input_conn.shape[0], poisson_inds, input_spikes_with_units)

    input_syn_i = Synapses(input_group, i_neurons, model='weights : siemens', on_pre='gP += weights')
    input_syn_e = Synapses(input_group, e_neurons, model='weights : siemens', on_pre='gP += weights')

    input_syn_i.connect(i=np.nonzero(input_conn[:, n_excite:])[0],
                        j=np.nonzero(input_conn[:, n_excite:])[1])
    input_syn_e.connect(i=np.nonzero(input_conn[:, :n_excite])[0],
                        j=np.nonzero(input_conn[:, :n_excite])[1])

    input_syn_i.weights = input_conn[:, n_excite:][input_conn[:, n_excite:] > 0] * nS
    input_syn_e.weights = input_conn[:, :n_excite][input_conn[:, :n_excite] > 0] * nS

    # Monitors
    e_spike_mon = SpikeMonitor(e_neurons)
    i_spike_mon = SpikeMonitor(i_neurons)

    # Run simulation
    run(simulation_time * second)

    # Save results
    trial_dir = f'{output_dir}/trial{trial_id}'
    os.makedirs(trial_dir, exist_ok=True)

    idxs = np.concatenate([np.array(e_spike_mon.i), np.array(i_spike_mon.i) + n_excite])
    times = np.concatenate([np.array(e_spike_mon.t/ms), np.array(i_spike_mon.t/ms)])
    np.savez(f'{trial_dir}/spike_monitor0.npz', idxs=idxs, times=times)

    return True


def run_trials_parallel(input_conn, recur_conn, input_times_list, input_spikes_list,
                        n_excite, n_inhib, simulation_time, output_dir,
                        n_workers=None, Vsi=-75):
    """
    Run multiple trials in parallel.

    Args:
        input_conn: Input connectivity matrix
        recur_conn: Recurrent connectivity matrix
        input_times_list: List of input spike index arrays, one per trial
        input_spikes_list: List of input spike time arrays, one per trial
        n_excite: Number of excitatory neurons
        n_inhib: Number of inhibitory neurons
        simulation_time: Simulation duration in seconds
        output_dir: Base output directory
        n_workers: Number of parallel workers (default: CPU count)
        Vsi: Inhibitory reversal potential

    Returns:
        List of (trial_id, success) tuples
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    n_trials = len(input_times_list)
    print(f"Running {n_trials} trials with {n_workers} workers...")

    # Package configuration (shared across all trials)
    config = {
        'input_conn': input_conn,
        'recur_conn': recur_conn,
        'n_excite': n_excite,
        'n_inhib': n_inhib,
        'simulation_time': simulation_time,
        'output_dir': output_dir,
        'Vsi': Vsi,
    }

    # Package trial arguments
    trial_args = [
        (trial_id, input_times_list[trial_id], input_spikes_list[trial_id])
        for trial_id in range(n_trials)
    ]

    # Create worker function with config baked in
    worker_fn = partial(run_single_trial, config=config)

    # Run in parallel
    start_time = time.time()

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(worker_fn, trial_args)

    # cleanup temp directories
    pids = np.unique([r[2] for r in results])
    for pid in pids:
        dir_name = f'/tmp/brian_standalone_{pid}'
        shutil.rmtree(dir_name)

    elapsed = time.time() - start_time
    successful = sum(1 for _, success, _ in results if success)
    print(f"Completed {successful}/{n_trials} trials in {elapsed:.1f}s "
          f"({elapsed/n_trials:.2f}s per trial, {n_trials/elapsed:.1f} trials/s)")

    return results


def run_stimulus_batch(input_conn, recur_conn, input_rates, n_trials, input_time_ms,
                       n_excite, n_inhib, simulation_time, output_dir,
                       n_workers=None, Vsi=-75):
    """
    Generate input spikes for a stimulus and run all trials in parallel.

    Args:
        input_conn: Input connectivity matrix
        recur_conn: Recurrent connectivity matrix
        input_rates: Array of input firing rates
        n_trials: Number of trials to run
        input_time_ms: Duration of input in milliseconds
        n_excite, n_inhib: Neuron counts
        simulation_time: Total simulation time in seconds
        output_dir: Output directory for this stimulus
        n_workers: Number of parallel workers
        Vsi: Inhibitory reversal potential

    Returns:
        List of results
    """
    # Generate input spikes for all trials
    print(f"Generating input spikes for {n_trials} trials...")
    inpt_times, inpt_spikes = generate_input_spikes(
        input_rates, n_trials, input_time_ms, output_file=None
    )

    # Save input spikes
    os.makedirs(output_dir, exist_ok=True)
    np.savez(f'{output_dir}/input_times.npz', *inpt_times)
    np.savez(f'{output_dir}/input_spikes.npz', *inpt_spikes)

    # Run trials in parallel
    results = run_trials_parallel(
        input_conn=input_conn,
        recur_conn=recur_conn,
        input_times_list=inpt_times,
        input_spikes_list=inpt_spikes,
        n_excite=n_excite,
        n_inhib=n_inhib,
        simulation_time=simulation_time,
        output_dir=output_dir,
        n_workers=n_workers,
        Vsi=Vsi
    )

    return results


if __name__ == '__main__':
    # Configuration - same as original fastSNN.py
    n_trials = 200
    n_excite = 4000
    n_inhib = 1000
    n_cells = n_excite + n_inhib

    n_input = 3000
    pei_w_multiplier = 1
    pei_p = 0.100

    ee_p = 0.28
    ei_p = 0.24
    ie_p = 0.28
    ii_p = 0.2

    w_mu = -0.64
    w_sd = 0.51

    rate_mean = 2.8
    rate_std = 0.3

    simulation_time = 1.05
    n_stimuli = 3

    # Input structure parameters
    input_rate_kappa = 0.0   # Von Mises kappa for input rate bump (0 = no structure)
    input_conn_kappa = 0.0   # Von Mises kappa for input connectivity (0 = random)

    # Number of parallel workers (set to number of CPU cores, or less if memory constrained)
    n_workers = min(mp.cpu_count(), 16)  # cap at 16 to avoid memory issues

    print(f"Using {n_workers} parallel workers")

    # Generate input connectivity (shared across all runs)
    if input_conn_kappa > 1e-5:
        input_conn = connmat.gen_ring_input_conn(n_input, n_cells, pei_p, w_mu, w_sd, input_conn_kappa, n_e=n_excite)
    else:
        input_conn = connmat.gen_input_conn(n_input, n_cells, pei_p, w_mu, w_sd)

    # Compute bump centers evenly distributed across 0 to pi for each stimulus
    bump_centers = np.linspace(0, np.pi, n_stimuli, endpoint=False)

    # Parameter sweep
    for input_time in np.arange(0.1, 1.1, 0.1):
        for kappa in np.arange(0.0, 2.2, 0.2):
            print(f"\n{'='*60}")
            print(f"Running input_time={input_time:.1f}, kappa={kappa:.1f}")
            print('='*60)

            # Generate recurrent connectivity for this kappa
            recur_conn = connmat.gen_ring_conn(
                n_excite, n_inhib, ii_p, ei_p, ie_p, ee_p,
                w_mu, w_sd, kappa, None, i_multiplier=10
            )

            save_dir = f'/mnt/eight1/tarek_runs/inpt_t{input_time}/kappa{kappa}'
            os.makedirs(save_dir, exist_ok=True)
            np.savez(f'{save_dir}/conn_mats.npz', inpt=input_conn, recurrent=recur_conn)

            # Run each stimulus
            for s in range(n_stimuli):
                print(f"\nStimulus {s}/{n_stimuli}")

                # Generate input rates with optional bump structure
                input_rates = generate_rate_vector(
                    n_input,
                    rate_mu=rate_mean,
                    rate_sigma=rate_std,
                    kappa=input_rate_kappa,
                    bump_center=bump_centers[s]
                )

                # Run all trials for this stimulus in parallel
                stim_dir = f'{save_dir}/stim{s}'
                results = run_stimulus_batch(
                    input_conn=input_conn,
                    recur_conn=recur_conn,
                    input_rates=input_rates,
                    n_trials=n_trials,
                    input_time_ms=input_time * 1000,
                    n_excite=n_excite,
                    n_inhib=n_inhib,
                    simulation_time=simulation_time,
                    output_dir=stim_dir,
                    n_workers=n_workers,
                    Vsi=-75
                )
