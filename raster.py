"""Quick raster plot for a single trial's spike data."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def find_spike_file(path_str: str) -> str:
    """Resolve a path to a spike_monitor0.npz file.

    Accepts:
    - Direct path to spike_monitor0.npz
    - A trial directory containing spike_monitor0.npz
    - A stim directory (uses trial0)
    - A parameter point directory (uses stim0/trial0)
    """
    p = Path(path_str)

    if p.is_file() and p.name == 'spike_monitor0.npz':
        return str(p)

    if p.is_dir():
        # Direct trial dir
        candidate = p / 'spike_monitor0.npz'
        if candidate.exists():
            return str(candidate)

        # Stim dir -> trial0
        candidate = p / 'trial0' / 'spike_monitor0.npz'
        if candidate.exists():
            return str(candidate)

        # Parameter point dir -> stim0/trial0
        candidate = p / 'stim0' / 'trial0' / 'spike_monitor0.npz'
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"Could not find spike_monitor0.npz from path: {path_str}\n"
        f"Pass a trial dir, stim dir, or parameter point dir."
    )


def main():
    parser = argparse.ArgumentParser(description='Quick raster plot for SNN spike data')
    parser.add_argument('path', type=str,
                        help='Path to spike file, trial dir, stim dir, or parameter point dir')
    parser.add_argument('--ms', type=float, nargs=2, default=None, metavar=('START', 'END'),
                        help='Time window in ms (e.g. --ms 50 200)')
    parser.add_argument('--neurons', type=int, nargs=2, default=None, metavar=('START', 'END'),
                        help='Neuron index range (e.g. --neurons 0 500)')
    parser.add_argument('-s', '--save', type=str, default=None,
                        help='Save to file instead of showing')
    args = parser.parse_args()

    spike_file = find_spike_file(args.path)
    data = np.load(spike_file)
    idxs = data['idxs']
    times = data['times']
    data.close()

    # Apply filters
    mask = np.ones(len(idxs), dtype=bool)
    if args.ms is not None:
        mask &= (times >= args.ms[0]) & (times <= args.ms[1])
    if args.neurons is not None:
        mask &= (idxs >= args.neurons[0]) & (idxs < args.neurons[1])
    idxs, times = idxs[mask], times[mask]

    print(f"File: {spike_file}")
    print(f"Spikes: {len(idxs)}, Neurons: {int(idxs.max())+1 if len(idxs) else 0}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(times, idxs, marker='|', s=1, color='black', linewidths=0.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')
    ax.set_title(Path(spike_file).parent.parent.name + '/' + Path(spike_file).parent.name)

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
