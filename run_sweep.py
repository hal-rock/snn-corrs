"""
Configurable parameter sweep runner for SNN simulations.

Usage:
    # Run with defaults
    python run_sweep.py

    # Run with config file
    python run_sweep.py --config my_config.json

    # Run with command line overrides
    python run_sweep.py --n_trials 100 --kappa 0.5,1.0,1.5

    # Combine config file with overrides
    python run_sweep.py --config base.json --n_trials 50
"""
import argparse
import json
import os
import sys
import itertools
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import numpy as np

# Local imports
import ConnMatGenerator as connmat
from input_generation import generate_input_spikes
from fastSNN_parallel import run_trials_parallel


# Default configuration - all parameters with their default values
DEFAULT_CONFIG = {
    # Network architecture
    "n_excite": 4000,
    "n_inhib": 1000,
    "n_input": 3000,

    # Connectivity probabilities
    "ee_p": 0.28,
    "ei_p": 0.24,
    "ie_p": 0.28,
    "ii_p": 0.2,
    "pei_p": 0.1,  # input connectivity probability

    # Weight distribution parameters
    "w_mu": -0.64,
    "w_sd": 0.51,

    # Network structure
    "kappa": 0.0,           # ring connectivity parameter
    "i_multiplier": 10,     # inhibitory weight multiplier

    # Input parameters
    "input_time": 300,      # input duration in ms
    "rate_mean": 2.8,       # log-normal rate mean
    "rate_std": 0.3,        # log-normal rate std

    # Simulation parameters
    "simulation_time": 1.05,  # total simulation time in seconds
    "n_trials": 200,          # number of trials per stimulus
    "n_stimuli": 3,           # number of different stimuli

    # Neuron parameters
    "Vsi": -75,  # inhibitory reversal potential (mV)

    # Parallelization
    "n_workers": None,  # None = use all CPUs
}

# Parameters that can be swept (with their types for parsing)
SWEEPABLE_PARAMS = {
    "kappa": float,
    "input_time": float,
    "n_trials": int,
    "n_stimuli": int,
    "ee_p": float,
    "ei_p": float,
    "ie_p": float,
    "ii_p": float,
    "pei_p": float,
    "w_mu": float,
    "w_sd": float,
    "i_multiplier": float,
    "rate_mean": float,
    "rate_std": float,
    "simulation_time": float,
    "Vsi": float,
    "n_excite": int,
    "n_inhib": int,
    "n_input": int,
}


def parse_value(value_str, param_type):
    """Parse a string value or range specification."""
    value_str = str(value_str).strip()

    # Check for range specification: start:stop:step or start:stop:n_steps:lin/log
    if ':' in value_str:
        parts = value_str.split(':')
        if len(parts) == 3:
            # start:stop:step
            start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
            values = list(np.arange(start, stop + step/2, step))
        elif len(parts) == 4:
            # start:stop:n_steps:lin or start:stop:n_steps:log
            start, stop, n_steps = float(parts[0]), float(parts[1]), int(parts[2])
            if parts[3] == 'log':
                values = list(np.logspace(np.log10(start), np.log10(stop), n_steps))
            else:
                values = list(np.linspace(start, stop, n_steps))
        else:
            raise ValueError(f"Invalid range specification: {value_str}")
        return [param_type(v) for v in values]

    # Check for comma-separated list
    if ',' in value_str:
        return [param_type(v.strip()) for v in value_str.split(',')]

    # Single value
    return [param_type(value_str)]


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def merge_configs(base, override):
    """Merge override config into base config."""
    result = base.copy()
    for key, value in override.items():
        if value is not None:
            result[key] = value
    return result


def generate_sweep_combinations(config):
    """
    Generate all combinations of sweep parameters.

    Returns:
        List of (param_dict, path_parts) tuples where:
        - param_dict: dict of parameter name -> value for this combination
        - path_parts: list of "param_value" strings for directory naming
    """
    sweep_params = {}
    fixed_params = {}

    for key, value in config.items():
        if key in SWEEPABLE_PARAMS:
            if isinstance(value, list):
                sweep_params[key] = value
            elif isinstance(value, str) and (':' in value or ',' in value):
                sweep_params[key] = parse_value(value, SWEEPABLE_PARAMS[key])
            else:
                fixed_params[key] = value
        else:
            fixed_params[key] = value

    if not sweep_params:
        # No sweeps, just return fixed params
        return [(fixed_params, [])]

    # Generate all combinations
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[name] for name in param_names]

    combinations = []
    for values in itertools.product(*param_values):
        param_dict = fixed_params.copy()
        path_parts = []
        for name, value in zip(param_names, values):
            param_dict[name] = value
            # Format path part
            if isinstance(value, float):
                if value == int(value):
                    path_parts.append(f"{name}_{int(value)}")
                else:
                    path_parts.append(f"{name}_{value:.4g}")
            else:
                path_parts.append(f"{name}_{value}")
        combinations.append((param_dict, path_parts))

    return combinations


def run_single_sweep_point(params, output_dir, run_name):
    """Run simulation for a single parameter combination."""
    n_excite = params['n_excite']
    n_inhib = params['n_inhib']
    n_cells = n_excite + n_inhib
    n_input = params['n_input']
    n_trials = params['n_trials']
    n_stimuli = params['n_stimuli']
    n_workers = params.get('n_workers') or mp.cpu_count()

    # Generate connectivity matrices
    print("  Generating connectivity matrices...")
    input_conn = connmat.gen_input_conn(
        n_input, n_cells, params['pei_p'], params['w_mu'], params['w_sd']
    )
    recur_conn = connmat.gen_ring_conn(
        n_excite, n_inhib,
        params['ii_p'], params['ei_p'], params['ie_p'], params['ee_p'],
        params['w_mu'], params['w_sd'],
        params['kappa'], None,
        i_multiplier=params['i_multiplier']
    )

    # Save connectivity
    os.makedirs(output_dir, exist_ok=True)
    np.savez(f'{output_dir}/conn_mats.npz', inpt=input_conn, recurrent=recur_conn)

    # Save parameters for this point
    with open(f'{output_dir}/params.json', 'w') as f:
        json.dump(params, f, indent=2)

    # Run each stimulus
    for s in range(n_stimuli):
        print(f"  Stimulus {s+1}/{n_stimuli}")

        stim_dir = f'{output_dir}/stim{s}'
        os.makedirs(stim_dir, exist_ok=True)

        # Generate input rates
        input_rates = np.random.lognormal(
            mean=params['rate_mean'],
            sigma=params['rate_std'],
            size=n_input
        )

        # Generate input spikes
        inpt_times, inpt_spikes = generate_input_spikes(
            input_rates, n_trials, params['input_time'], output_file=None
        )

        # Save input spikes
        np.savez(f'{stim_dir}/input_times.npz', *inpt_times)
        np.savez(f'{stim_dir}/input_spikes.npz', *inpt_spikes)

        # Run trials in parallel
        results = run_trials_parallel(
            input_conn=input_conn,
            recur_conn=recur_conn,
            input_times_list=inpt_times,
            input_spikes_list=inpt_spikes,
            n_excite=n_excite,
            n_inhib=n_inhib,
            simulation_time=params['simulation_time'],
            output_dir=stim_dir,
            n_workers=n_workers,
            Vsi=params['Vsi']
        )

        successful = sum(1 for _, success in results if success)
        print(f"    Completed {successful}/{n_trials} trials")


def main():
    parser = argparse.ArgumentParser(
        description='Run SNN parameter sweep simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python run_sweep.py

  # Run with config file
  python run_sweep.py --config sweep_config.json

  # Sweep over kappa values
  python run_sweep.py --kappa 0.0,0.5,1.0,1.5,2.0

  # Sweep over range (start:stop:step)
  python run_sweep.py --kappa 0.0:2.0:0.2

  # Sweep over range with N points (start:stop:N:lin or start:stop:N:log)
  python run_sweep.py --input_time 100:1000:10:lin

  # Multiple sweeps (creates grid)
  python run_sweep.py --kappa 0.0,1.0,2.0 --input_time 100,300,500

  # Override specific parameters
  python run_sweep.py --n_trials 50 --n_workers 8
        """
    )

    parser.add_argument('--config', type=str, help='Path to JSON config file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Base output directory (default: outputs)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name for this run (default: timestamp)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print config and sweep combinations without running')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel workers (default: all CPUs)')

    # Add arguments for all sweepable parameters
    for param, ptype in SWEEPABLE_PARAMS.items():
        parser.add_argument(f'--{param}', type=str, default=None,
                            help=f'{param} (default: {DEFAULT_CONFIG.get(param)})')

    args = parser.parse_args()

    # Build configuration
    config = DEFAULT_CONFIG.copy()

    # Load config file if specified
    if args.config:
        file_config = load_config(args.config)
        config = merge_configs(config, file_config)

    # Apply command line overrides
    for param in SWEEPABLE_PARAMS:
        value = getattr(args, param)
        if value is not None:
            # Check if it's a sweep specification (contains : or ,)
            if ':' in value or ',' in value:
                config[param] = value  # Keep as string for later parsing
            else:
                # Single value - convert to proper type
                config[param] = SWEEPABLE_PARAMS[param](value)

    # Handle n_workers specially (not sweepable but settable)
    if args.n_workers is not None:
        config['n_workers'] = args.n_workers

    # Generate sweep combinations
    combinations = generate_sweep_combinations(config)

    # Create run directory
    run_name = args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output_dir = Path(args.output_dir) / run_name

    print(f"Run name: {run_name}")
    print(f"Output directory: {base_output_dir}")
    print(f"Number of parameter combinations: {len(combinations)}")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        print("\nBase configuration:")
        print(json.dumps({k: v for k, v in config.items()
                         if not isinstance(v, str) or (':' not in v and ',' not in v)},
                        indent=2))
        print("\nSweep combinations:")
        for i, (params, path_parts) in enumerate(combinations):
            sweep_vals = {k: v for k, v in params.items()
                         if k in SWEEPABLE_PARAMS and
                         (isinstance(config.get(k), list) or
                          (isinstance(config.get(k), str) and
                           (':' in str(config.get(k)) or ',' in str(config.get(k)))))}
            print(f"  {i+1}. {'/'.join(path_parts) or '(defaults)'}: {sweep_vals}")
        return

    # Create output directory and save master config
    os.makedirs(base_output_dir, exist_ok=True)

    master_config = {
        'run_name': run_name,
        'created': datetime.now().isoformat(),
        'base_config': {k: v for k, v in config.items()},
        'n_combinations': len(combinations),
        'sweep_parameters': [name for name, val in config.items()
                            if isinstance(val, list) or
                            (isinstance(val, str) and (':' in val or ',' in val))],
    }
    with open(base_output_dir / 'config.json', 'w') as f:
        json.dump(master_config, f, indent=2)

    print(f"\nSaved master config to {base_output_dir / 'config.json'}")

    # Run each combination
    for i, (params, path_parts) in enumerate(combinations):
        print(f"\n{'='*60}")
        print(f"Combination {i+1}/{len(combinations)}")
        if path_parts:
            print(f"Parameters: {', '.join(path_parts)}")
        print('='*60)

        # Build output path
        if path_parts:
            combo_output_dir = base_output_dir / '/'.join(path_parts)
        else:
            combo_output_dir = base_output_dir

        run_single_sweep_point(params, str(combo_output_dir), run_name)

    print(f"\n{'='*60}")
    print(f"Sweep complete! Results saved to: {base_output_dir}")
    print('='*60)


if __name__ == '__main__':
    main()
