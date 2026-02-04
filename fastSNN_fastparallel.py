"""
Optimized parallel SNN simulation using runtime mode.

Key optimization: Parallelization at the parameter-point level, not trial level.
Each worker handles all trials for one (kappa, input_time) combination.

This avoids the massive overhead of the original fastSNN_parallel.py which
tried to use standalone mode but recompiled for every trial.

After testing, we found that:
- Brian2's run_args doesn't support changing SpikeGeneratorGroup spikes
- Runtime mode with Cython is fast enough (~10-20s per 1s simulation on 5000 neurons)
- The main speedup comes from proper parallelization, not standalone mode
"""
from brian2 import *
import numpy as np
import os
import time
import multiprocessing as mp
from pathlib import Path

# Local imports
from input_generation import generate_input_spikes, generate_rate_vector
import ConnMatGenerator as connmat

# Use runtime mode with Cython for good performance
prefs.codegen.target = 'cython'

# Suppress Brian2 namespace resolution warnings (expected when passing I_inj/Vsi per trial)
BrianLogger.suppress_hierarchy('brian2.groups.group.Group.resolve')


def run_trials_parallel(input_conn, recur_conn, input_times_list, input_spikes_list,
                        n_excite, n_inhib, simulation_time, output_dir,
                        n_workers=None, Vsi=-75, I_values=None):
    """
    Run multiple trials in parallel using runtime mode.

    Compatible interface with fastSNN_parallel.run_trials_parallel() so this
    can be used as a drop-in replacement in run_sweep.py.

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
        I_values: Array of injected currents (pA), one per trial (same for E and I)

    Returns:
        List of (trial_id, success, pid) tuples
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    n_trials = len(input_times_list)
    print(f"Running {n_trials} trials with {n_workers} workers (runtime mode)...")

    # Default to zero current if not provided
    if I_values is None:
        I_values = np.zeros(n_trials)

    # Package trial arguments - each trial gets its own I_value
    trial_args = [
        (trial_id, input_times_list[trial_id], input_spikes_list[trial_id],
         input_conn, recur_conn, n_excite, n_inhib, simulation_time, output_dir, Vsi, I_values[trial_id])
        for trial_id in range(n_trials)
    ]

    start_time = time.time()

    # Use spawn context for clean processes
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_run_trial_worker, trial_args)

    elapsed = time.time() - start_time
    successful = sum(1 for _, success, _ in results if success)
    print(f"Completed {successful}/{n_trials} trials in {elapsed:.1f}s "
          f"({elapsed/n_trials:.2f}s per trial, {n_trials/elapsed:.1f} trials/s)")

    return results


def _run_trial_worker(args):
    """Worker function for run_trials_parallel."""
    (trial_id, inpt_times, inpt_spikes, input_conn, recur_conn,
     n_excite, n_inhib, simulation_time, output_dir, Vsi, I_inj) = args

    pid = os.getpid()
    trial_dir = f'{output_dir}/trial{trial_id}'

    try:
        run_single_trial(
            n_excite=n_excite,
            n_inhib=n_inhib,
            input_conn=input_conn,
            recur_conn=recur_conn,
            inputs=(inpt_times, inpt_spikes),
            simulation_time=simulation_time,
            trial_dir=trial_dir,
            Vsi=Vsi,
            I_inj=I_inj
        )
        return (trial_id, True, pid)
    except Exception as e:
        print(f"Trial {trial_id} failed: {e}")
        return (trial_id, False, pid)


def run_single_trial(n_excite, n_inhib, input_conn, recur_conn, inputs,
                     simulation_time, trial_dir, Vsi=-75, I_inj=0.0):
    """
    Run a single simulation trial in runtime mode.

    Args:
        n_excite, n_inhib: Neuron counts
        input_conn: Input connectivity matrix
        recur_conn: Recurrent connectivity matrix
        inputs: tuple of (spike_indices, spike_times) in ms
        simulation_time: Total simulation time in seconds
        trial_dir: Output directory for this trial
        Vsi: Inhibitory reversal potential
        I_inj: Injected current for all neurons (pA), same for E and I

    Returns:
        True if successful
    """
    start_scope()  # Fresh Brian2 scope for each trial

    # Neuron parameters
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

    # Injected current with units (same for E and I)
    I_inj_val = I_inj * pA

    eqs = '''
        dvm/dt = (gL*(EL - vm) + gse*(Vse-vm) + gsi*(Vsi-vm)+ gP*(Vse-vm) + gL*DeltaT*exp((vm - VT)/DeltaT) - w + I_inj)/C : volt
        dw/dt = (a*(vm - EL) - w)/tauw : amp
        dgse/dt=-gse/taue : siemens
        dgsi/dt=-gsi/taui : siemens
        dgP/dt=-gP/taup : siemens
        I_inj : amp
    '''

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

    # Set injected current (same for both populations)
    e_neurons.I_inj = I_inj_val
    i_neurons.I_inj = I_inj_val

    # Initial membrane potentials
    e_neurons.vm = 'rand()*10*mV - 70*mV'
    i_neurons.vm = 'rand()*10*mV - 70*mV'

    # Recurrent synapses
    ee_slice = recur_conn[:n_excite, :n_excite]
    ie_slice = recur_conn[n_excite:, :n_excite]
    ei_slice = recur_conn[:n_excite, n_excite:]
    ii_slice = recur_conn[n_excite:, n_excite:]

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
    os.makedirs(trial_dir, exist_ok=True)
    idxs = np.concatenate([np.array(e_spike_mon.i), np.array(i_spike_mon.i) + n_excite])
    times = np.concatenate([np.array(e_spike_mon.t/ms), np.array(i_spike_mon.t/ms)])
    np.savez(f'{trial_dir}/spike_monitor0.npz', idxs=idxs, times=times)

    return True


def run_parameter_point(param_args):
    """
    Run all stimuli and trials for a single parameter point.

    This runs sequentially within a single process - parallelization
    happens at the parameter-point level, not trial level.
    """
    # Unpack parameters
    kappa = param_args['kappa']
    kappa_e = param_args.get('kappa_e')  # kappa for E->E and E->I connections
    kappa_i = param_args.get('kappa_i')  # kappa for I->I and I->E connections
    input_time = param_args['input_time']
    input_conn = param_args['input_conn']
    n_excite = param_args['n_excite']
    n_inhib = param_args['n_inhib']
    n_input = input_conn.shape[0]
    n_stimuli = param_args['n_stimuli']
    n_trials = param_args['n_trials']
    simulation_time = param_args['simulation_time']
    rate_mean = param_args['rate_mean']
    rate_std = param_args['rate_std']
    save_dir = param_args['save_dir']
    Vsi = param_args.get('Vsi', -75)
    I_mu = param_args.get('I_mu', 0.0)
    I_sigma = param_args.get('I_sigma', 0.0)

    # Connection params
    ee_p = param_args['ee_p']
    ei_p = param_args['ei_p']
    ie_p = param_args['ie_p']
    ii_p = param_args['ii_p']
    w_mu = param_args['w_mu']
    w_sd = param_args['w_sd']
    i_multiplier = param_args.get('i_multiplier', 10)

    pid = os.getpid()

    results = {
        'kappa': kappa,
        'input_time': input_time,
        'pid': pid,
        'total_time': 0,
        'n_trials_completed': 0,
        'success': False
    }

    try:
        start_total = time.time()

        # Generate recurrent connectivity for this kappa
        kappa_e_str = f", kappa_e={kappa_e:.1f}" if kappa_e is not None else ""
        kappa_i_str = f", kappa_i={kappa_i:.1f}" if kappa_i is not None else ""
        print(f"[PID {pid}] Generating connectivity for kappa={kappa:.1f}{kappa_e_str}{kappa_i_str}...")
        recur_conn = connmat.gen_ring_conn(
            n_excite, n_inhib, ii_p, ei_p, ie_p, ee_p,
            w_mu, w_sd, kappa, None, i_multiplier=i_multiplier,
            kappa_e=kappa_e, kappa_i=kappa_i
        )

        # Save connectivity
        os.makedirs(save_dir, exist_ok=True)
        np.savez(f'{save_dir}/conn_mats.npz', inpt=input_conn, recurrent=recur_conn)

        # Pre-generate all input spikes and per-trial currents
        print(f"[PID {pid}] Generating input spikes for {n_stimuli} stimuli x {n_trials} trials...")
        bump_centers = np.linspace(0, np.pi, n_stimuli, endpoint=False)

        all_inputs = []  # List of (stim_idx, trial_idx, stim_dir, indices, times, I_inj)

        for s in range(n_stimuli):
            input_rates = generate_rate_vector(
                n_input, rate_mu=rate_mean, rate_sigma=rate_std,
                kappa=0.0, bump_center=bump_centers[s]
            )
            inpt_times, inpt_spikes = generate_input_spikes(
                input_rates, n_trials, input_time * 1000, output_file=None
            )

            # Generate per-trial injected currents (independent for each stimulus)
            if I_sigma > 0:
                I_values = np.random.normal(I_mu, I_sigma, n_trials)
            else:
                I_values = np.full(n_trials, I_mu)

            stim_dir = f'{save_dir}/stim{s}'
            os.makedirs(stim_dir, exist_ok=True)
            np.savez(f'{stim_dir}/input_times.npz', *inpt_times)
            np.savez(f'{stim_dir}/input_spikes.npz', *inpt_spikes)
            np.save(f'{stim_dir}/I_values.npy', I_values)

            for t in range(n_trials):
                all_inputs.append((s, t, stim_dir, inpt_times[t], inpt_spikes[t], I_values[t]))

        # Run all trials
        print(f"[PID {pid}] Running {len(all_inputs)} trials for kappa={kappa:.1f}, input_time={input_time:.1f}...")
        run_start = time.time()
        trials_completed = 0

        for s, t, stim_dir, trial_indices, trial_times, I_inj in all_inputs:
            trial_dir = f'{stim_dir}/trial{t}'

            try:
                run_single_trial(
                    n_excite=n_excite,
                    n_inhib=n_inhib,
                    input_conn=input_conn,
                    recur_conn=recur_conn,
                    inputs=(trial_indices, trial_times),
                    simulation_time=simulation_time,
                    trial_dir=trial_dir,
                    Vsi=Vsi,
                    I_inj=I_inj
                )
                trials_completed += 1

                if trials_completed % 25 == 0:
                    elapsed = time.time() - run_start
                    rate = trials_completed / elapsed
                    remaining = len(all_inputs) - trials_completed
                    eta = remaining / rate if rate > 0 else 0
                    print(f"[PID {pid}] {trials_completed}/{len(all_inputs)} trials "
                          f"({rate:.2f}/s, ETA {eta:.0f}s)")

            except Exception as e:
                print(f"[PID {pid}] Trial {s}/{t} failed: {e}")

        results['total_time'] = time.time() - start_total
        results['n_trials_completed'] = trials_completed
        results['success'] = trials_completed == len(all_inputs)

        print(f"[PID {pid}] Finished kappa={kappa:.1f}, input_time={input_time:.1f}: "
              f"{trials_completed}/{len(all_inputs)} trials in {results['total_time']:.1f}s")

    except Exception as e:
        print(f"[PID {pid}] Error for kappa={kappa:.1f}, input_time={input_time:.1f}: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)

    return results


def run_sweep_parallel(config, n_workers=None):
    """
    Run a parameter sweep with parallelization at the parameter-point level.

    Each worker handles one full parameter point (all stimuli and trials).
    This is more efficient than parallelizing at the trial level because:
    1. No repeated connectivity generation
    2. Cython compilation happens once per process
    3. Better memory locality
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    # Generate parameter combinations
    kappa_values = config.get('kappa_values', np.arange(0.0, 2.2, 0.2))
    input_time_values = config.get('input_time_values', np.arange(0.1, 1.1, 0.1))

    # Generate shared input connectivity
    n_input = config.get('n_input', 3000)
    n_excite = config.get('n_excite', 4000)
    n_inhib = config.get('n_inhib', 1000)
    n_cells = n_excite + n_inhib
    pei_p = config.get('pei_p', 0.1)
    w_mu = config.get('w_mu', -0.64)
    w_sd = config.get('w_sd', 0.51)

    print("Generating shared input connectivity...")
    input_conn = connmat.gen_input_conn(n_input, n_cells, pei_p, w_mu, w_sd)

    # Build parameter point list
    param_points = []
    for input_time in input_time_values:
        for kappa in kappa_values:
            save_dir = f"{config['output_dir']}/input_time_{input_time:.1f}/kappa_{kappa:.1f}"

            param_args = {
                'kappa': kappa,
                'kappa_e': config.get('kappa_e'),  # kappa for E->E and E->I
                'kappa_i': config.get('kappa_i'),  # kappa for I->I and I->E
                'input_time': input_time,
                'input_conn': input_conn,
                'n_excite': n_excite,
                'n_inhib': n_inhib,
                'n_stimuli': config.get('n_stimuli', 3),
                'n_trials': config.get('n_trials', 100),
                'simulation_time': config.get('simulation_time', 1.05),
                'rate_mean': config.get('rate_mean', 2.8),
                'rate_std': config.get('rate_std', 0.3),
                'save_dir': save_dir,
                'ee_p': config.get('ee_p', 0.28),
                'ei_p': config.get('ei_p', 0.24),
                'ie_p': config.get('ie_p', 0.28),
                'ii_p': config.get('ii_p', 0.2),
                'w_mu': w_mu,
                'w_sd': w_sd,
                'i_multiplier': config.get('i_multiplier', 10),
                'Vsi': config.get('Vsi', -75),
                'I_mu': config.get('I_mu', 0.0),
                'I_sigma': config.get('I_sigma', 0.0),
            }
            param_points.append(param_args)

    n_points = len(param_points)
    n_trials_per_point = config.get('n_stimuli', 3) * config.get('n_trials', 100)

    print(f"\nSweep configuration:")
    print(f"  Parameter points: {n_points}")
    print(f"  Trials per point: {n_trials_per_point}")
    print(f"  Total trials: {n_points * n_trials_per_point}")
    print(f"  Workers: {n_workers}")
    print()

    start_time = time.time()

    # Use spawn context for clean processes
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(run_parameter_point, param_points)

    elapsed = time.time() - start_time
    successful = sum(1 for r in results if r['success'])
    total_trials = sum(r['n_trials_completed'] for r in results)

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Completed {successful}/{n_points} parameter points")
    print(f"Total trials: {total_trials}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Average: {elapsed/n_points:.1f}s per parameter point")
    print(f"         {elapsed/total_trials:.2f}s per trial")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run parallel SNN sweep')
    parser.add_argument('--output_dir', type=str, default='outputs/fast_sweep',
                        help='Output directory')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of trials per stimulus (default: 100)')
    parser.add_argument('--n_stimuli', type=int, default=3,
                        help='Number of stimuli (default: 3)')
    parser.add_argument('--test', action='store_true',
                        help='Run a small test sweep')

    args = parser.parse_args()

    if args.test:
        # Small test configuration
        config = {
            'output_dir': args.output_dir,
            'kappa_values': np.array([0.0, 1.0]),
            'input_time_values': np.array([0.5]),
            'kappa_e': None,  # kappa for E->E and E->I (None = use kappa)
            'kappa_i': None,  # kappa for I->I and I->E (None = use kappa)
            'n_stimuli': 2,
            'n_trials': 5,
            'n_excite': 4000,
            'n_inhib': 1000,
            'n_input': 3000,
            'simulation_time': 1.05,
            'rate_mean': 2.8,
            'rate_std': 0.3,
            'pei_p': 0.1,
            'ee_p': 0.28,
            'ei_p': 0.24,
            'ie_p': 0.28,
            'ii_p': 0.2,
            'w_mu': -0.64,
            'w_sd': 0.51,
            'i_multiplier': 10,
            'Vsi': -75,
            'I_mu': 0.0,   # mean injected current for all neurons (pA)
            'I_sigma': 0.0,  # std of injected current across trials (pA)
        }
        print("Running TEST configuration (2 kappa x 1 input_time x 2 stim x 5 trials)")
    else:
        # Full configuration
        config = {
            'output_dir': args.output_dir,
            'kappa_values': np.arange(0.0, 2.2, 0.2),
            'input_time_values': np.arange(0.1, 1.1, 0.1),
            'kappa_e': None,  # kappa for E->E and E->I (None = use kappa)
            'kappa_i': None,  # kappa for I->I and I->E (None = use kappa)
            'n_stimuli': args.n_stimuli,
            'n_trials': args.n_trials,
            'n_excite': 4000,
            'n_inhib': 1000,
            'n_input': 3000,
            'simulation_time': 1.05,
            'rate_mean': 2.8,
            'rate_std': 0.3,
            'pei_p': 0.1,
            'ee_p': 0.28,
            'ei_p': 0.24,
            'ie_p': 0.28,
            'ii_p': 0.2,
            'w_mu': -0.64,
            'w_sd': 0.51,
            'i_multiplier': 10,
            'Vsi': -75,
            'I_mu': 0.0,   # mean injected current for all neurons (pA)
            'I_sigma': 0.0,  # std of injected current across trials (pA)
        }

    n_workers = args.n_workers or min(mp.cpu_count(), 16)
    print(f"Starting sweep with {n_workers} workers")

    results = run_sweep_parallel(config, n_workers=n_workers)

    # Summary
    print("\nPer-point summary:")
    for r in results:
        status = "OK" if r['success'] else f"FAILED: {r.get('error', 'incomplete')}"
        print(f"  kappa={r['kappa']:.1f}, input_time={r['input_time']:.1f}: "
              f"{r['n_trials_completed']} trials, {r['total_time']:.1f}s - {status}")
