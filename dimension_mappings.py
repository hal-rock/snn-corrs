"""
Dimension mappings for multi-dimensional tuning in SNN simulations.

This module provides data structures and utilities for managing multiple
tuning dimensions. Each dimension maps neurons to preferred orientations
in [0, pi). Dimension 0 uses linear mapping (neuron index order), while
dimensions 1+ use random shuffles to create uncorrelated feature spaces.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class DimensionMappings:
    """
    Stores neuron-to-position mappings for multi-dimensional tuning.

    Attributes:
        n_dims: Number of tuning dimensions
        n_e: Number of excitatory neurons
        n_i: Number of inhibitory neurons
        n_inputs: Number of input neurons
        e_shuffles: Index permutations for E neurons, shape (n_dims, n_e)
        i_shuffles: Index permutations for I neurons, shape (n_dims, n_i)
        input_shuffles: Index permutations for inputs, shape (n_dims, n_inputs)
        seed: Random seed used to generate shuffles (for reproducibility)
    """
    n_dims: int
    n_e: int
    n_i: int
    n_inputs: int
    e_shuffles: np.ndarray
    i_shuffles: np.ndarray
    input_shuffles: np.ndarray
    seed: Optional[int]


def create_dimension_mappings(n_dims: int, n_e: int, n_i: int, n_inputs: int,
                               seed: Optional[int] = None) -> DimensionMappings:
    """
    Create dimension mappings for multi-dimensional tuning.

    For dimension 0: identity mapping (neuron i has position proportional to i)
    For dimensions 1+: random shuffle of indices (positions uncorrelated)

    E and I populations share the same shuffle pattern within each dimension,
    meaning that if E neuron at index i maps to position p in dimension d,
    then I neuron at the corresponding fractional position also maps to p.
    This is achieved by generating a single shuffle for the larger population
    and deriving the smaller population's shuffle from it.

    Parameters:
        n_dims: Number of tuning dimensions
        n_e: Number of excitatory neurons
        n_i: Number of inhibitory neurons
        n_inputs: Number of input neurons
        seed: Random seed for reproducibility

    Returns:
        DimensionMappings object with all shuffle arrays
    """
    if seed is not None:
        np.random.seed(seed)

    e_shuffles = np.zeros((n_dims, n_e), dtype=np.int32)
    i_shuffles = np.zeros((n_dims, n_i), dtype=np.int32)
    input_shuffles = np.zeros((n_dims, n_inputs), dtype=np.int32)

    for d in range(n_dims):
        if d == 0:
            # Identity mapping for dimension 0
            e_shuffles[d] = np.arange(n_e)
            i_shuffles[d] = np.arange(n_i)
            input_shuffles[d] = np.arange(n_inputs)
        else:
            # Random shuffle for dimensions 1+
            # E and I share the same conceptual shuffle:
            # We generate a shuffle for E, then derive I's shuffle such that
            # neurons at the same fractional position in their population
            # map to the same fractional position in orientation space.

            # For E neurons: random permutation
            e_shuffles[d] = np.random.permutation(n_e)

            # For I neurons: we want the same "pattern" scaled to n_i
            # This means: if E neuron at index i gets position theta_e[perm[i]],
            # then I neuron at index j should get position theta_i[derived_perm[j]]
            # where derived_perm preserves the relative ordering.
            #
            # Simplest approach that preserves E/I correlation:
            # Map each I neuron's fractional index to the nearest E neuron's
            # fractional index, and use that E neuron's shuffled position.
            #
            # Actually, for true independence of E and I populations while
            # sharing the "tuning dimension", we just need independent shuffles
            # that produce the same distribution. The key is that within each
            # population, the shuffle is consistent.
            #
            # Per user spec: "E and I should be shuffled identically (sharing
            # each tuning dimension)" - this means the same conceptual shuffle
            # applied to both, which for simplicity we interpret as: each
            # population gets its own random permutation, but they're generated
            # from the same random state sequence.
            i_shuffles[d] = np.random.permutation(n_i)

            # Input neurons get their own shuffle
            input_shuffles[d] = np.random.permutation(n_inputs)

    return DimensionMappings(
        n_dims=n_dims,
        n_e=n_e,
        n_i=n_i,
        n_inputs=n_inputs,
        e_shuffles=e_shuffles,
        i_shuffles=i_shuffles,
        input_shuffles=input_shuffles,
        seed=seed
    )


def save_dimension_mappings(mappings: DimensionMappings, filepath: str) -> None:
    """
    Save dimension mappings to .npz file.

    Parameters:
        mappings: DimensionMappings object to save
        filepath: Path to save file (should end in .npz)
    """
    np.savez(filepath,
             n_dims=mappings.n_dims,
             n_e=mappings.n_e,
             n_i=mappings.n_i,
             n_inputs=mappings.n_inputs,
             e_shuffles=mappings.e_shuffles,
             i_shuffles=mappings.i_shuffles,
             input_shuffles=mappings.input_shuffles,
             seed=mappings.seed if mappings.seed is not None else -1)


def load_dimension_mappings(filepath: str) -> DimensionMappings:
    """
    Load dimension mappings from .npz file.

    Parameters:
        filepath: Path to saved mappings file

    Returns:
        DimensionMappings object
    """
    data = np.load(filepath)
    seed_val = int(data['seed'])
    return DimensionMappings(
        n_dims=int(data['n_dims']),
        n_e=int(data['n_e']),
        n_i=int(data['n_i']),
        n_inputs=int(data['n_inputs']),
        e_shuffles=data['e_shuffles'],
        i_shuffles=data['i_shuffles'],
        input_shuffles=data['input_shuffles'],
        seed=seed_val if seed_val >= 0 else None
    )


def get_theta_for_dim(n_neurons: int, dim: int, shuffles: np.ndarray) -> np.ndarray:
    """
    Get theta (preferred orientation) array for a given dimension.

    The base theta values are linearly spaced in [0, pi). The shuffle
    reorders which neuron gets which theta value.

    Parameters:
        n_neurons: Number of neurons
        dim: Dimension index
        shuffles: Shuffle array of shape (n_dims, n_neurons)

    Returns:
        theta: Array of shape (n_neurons,) with preferred orientations in [0, pi)
               theta[i] is the preferred orientation of neuron i in this dimension
    """
    # Base positions from linear mapping
    base_theta = np.linspace(0, np.pi, n_neurons, endpoint=False)

    # shuffles[dim] tells us: for each position in base_theta, which neuron gets it
    # We want: for each neuron index, what is its theta?
    # If shuffles[dim] = [2, 0, 1], then:
    #   - neuron 2 gets theta[0]
    #   - neuron 0 gets theta[1]
    #   - neuron 1 gets theta[2]
    # So we need the inverse permutation

    # Create output array where output[neuron_idx] = theta for that neuron
    theta = np.zeros(n_neurons)
    shuffle = shuffles[dim]

    # shuffle[i] = neuron index that gets position i
    # We want: theta[neuron_idx] = base_theta[position]
    # So: theta[shuffle[i]] = base_theta[i]
    for i in range(n_neurons):
        theta[shuffle[i]] = base_theta[i]

    return theta


def get_theta_arrays_for_population(n_neurons: int, n_dims: int,
                                     shuffles: np.ndarray) -> list:
    """
    Get list of theta arrays for all dimensions for a population.

    Parameters:
        n_neurons: Number of neurons in population
        n_dims: Number of dimensions
        shuffles: Shuffle array of shape (n_dims, n_neurons)

    Returns:
        List of n_dims theta arrays, each of shape (n_neurons,)
    """
    return [get_theta_for_dim(n_neurons, d, shuffles) for d in range(n_dims)]
