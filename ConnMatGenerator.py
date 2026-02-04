import csv
import numpy
import numpy as np
from os import path, makedirs

from scipy.stats import vonmises

from dimension_mappings import get_theta_for_dim, get_theta_arrays_for_population


def get_prob_matrix_multidim(n_src, n_dest, theta_src_list, theta_dest_list, target_p, kappa_list):
    """
    Compute probability matrix as multiplicative product of per-dimension von Mises.

    Each dimension contributes a von Mises factor normalized to mean=1, then the
    product is normalized to achieve the target average connection probability.

    Parameters:
        n_src: Number of source neurons
        n_dest: Number of destination neurons
        theta_src_list: List of theta arrays for source, one per dimension
        theta_dest_list: List of theta arrays for destination, one per dimension
        target_p: Target average connection probability
        kappa_list: Kappa values, one per dimension

    Returns:
        P: Probability matrix of shape (n_dest, n_src)
    """
    n_dims = len(theta_src_list)

    # Start with uniform probability of 1.0 (will normalize at end)
    combined_probs = np.ones((n_dest, n_src))

    for d in range(n_dims):
        k = kappa_list[d]
        if k < 1e-5:
            continue  # Uniform factor - no change to combined_probs

        T_dest, T_src = np.meshgrid(theta_src_list[d], theta_dest_list[d])
        diff_scaled = 2 * (T_dest - T_src)
        dim_probs = vonmises.pdf(diff_scaled, k)

        # Normalize this dimension to mean=1 before multiplying
        dim_probs = dim_probs / np.mean(dim_probs)
        combined_probs = combined_probs * dim_probs

    # Final normalization to achieve target_p
    scale_factor = target_p / np.mean(combined_probs)
    combined_probs = combined_probs * scale_factor

    # Clip to valid probability range
    return np.clip(combined_probs, 0, 1)


# AI slop for recurrent connectivity w/ structure
def gen_ring_conn(n_e, n_i, p_ii, p_ei, p_ie, p_ee, w_mu, w_sd, kappa, output_dir,
                  i_multiplier=10, kappa_e=None, kappa_i=None,
                  n_dims=1, dim_mappings=None, kappa_per_dim=None):
    '''
    Generates a connectivity matrix with ring-like tuning structure.

    Parameters:
    -----------
    kappa : float
        Concentration parameter for Von Mises distribution (used if kappa_e/kappa_i not specified).
        kappa = 0 implies uniform random connectivity (disordered).
        Higher kappa implies sharper tuning (connections mostly between similar orientations).
    kappa_e : float, optional
        Kappa for connections FROM excitatory cells (EE and IE connections).
        If None, uses the value of kappa.
    kappa_i : float, optional
        Kappa for connections FROM inhibitory cells (II and EI connections).
        If None, uses the value of kappa.
    n_dims : int, optional
        Number of tuning dimensions. Default 1 for backward compatibility.
    dim_mappings : DimensionMappings, optional
        Required if n_dims > 1. Contains shuffle mappings for each dimension.
    kappa_per_dim : list of float, optional
        Kappa value for each dimension. If None, uses kappa (or kappa_e/kappa_i) for all dimensions.
        Length must equal n_dims if provided.
    '''
    # Handle backward compatibility: if kappa_e/kappa_i not specified, use kappa
    if kappa_e is None:
        kappa_e = kappa
    if kappa_i is None:
        kappa_i = kappa
    n_t = n_e + n_i
    total_weights = np.zeros((n_t, n_t))

    if n_dims == 1:
        # Single-dimension code path (original behavior)

        # 1. Assign preferred orientations (0 to pi)
        theta_e = np.linspace(0, np.pi, n_e, endpoint=False)
        theta_i = np.linspace(0, np.pi, n_i, endpoint=False)

        # Helper to create probability matrices
        def get_prob_matrix(n_src, n_dest, theta_src, theta_dest, target_p, k):
            # Calculate circular distance (difference in preferred orientation)
            # We use meshgrid to get the (dest, src) matrix of differences
            # Note: Orientation space is usually 0-180 (pi), so we treat the domain as [0, pi]
            # and wrap differences accordingly.

            # Grid of orientations
            T_dest, T_src = np.meshgrid(theta_src, theta_dest)

            # Circular difference on a domain of [0, pi]
            # (This is equivalent to 2*angle difference on a 2pi circle)
            diff = T_dest - T_src

            # Map difference to [-pi, pi] for Von Mises standard usage (which assumes 2pi domain)
            # Because our physical domain is 0-pi, we multiply by 2 to stretch it to 0-2pi
            # for the standard vonmises PDF function.
            diff_scaled = 2 * diff

            if k < 1e-5:
                # Case: Uniform (Kappa approx 0)
                return np.full((n_dest, n_src), target_p)
            else:
                # Calculate unscaled Von Mises PDF
                # vonmises.pdf(x, kappa) is defined on [-pi, pi]
                probs = vonmises.pdf(diff_scaled, k)

                # Normalize so the average probability equals the target_p
                # We want mean(probs) * scale_factor = target_p
                scale_factor = target_p / np.mean(probs)
                probs = probs * scale_factor

                # Clip probabilities to [0, 1] to prevent math errors if target_p is high
                probs = np.clip(probs, 0, 1)

                return probs

        # 2. Generate probability matrices for each quadrant
        # kappa_e for connections FROM E cells (EE, IE), kappa_i for connections FROM I cells (EI, II)
        P_ee = get_prob_matrix(n_e, n_e, theta_e, theta_e, p_ee, kappa_e)
        P_ie = get_prob_matrix(n_e, n_i, theta_e, theta_i, p_ie, kappa_e)  # Input from E (cols), Output to I (rows)
        P_ei = get_prob_matrix(n_i, n_e, theta_i, theta_e, p_ei, kappa_i)  # Input from I, Output to E
        P_ii = get_prob_matrix(n_i, n_i, theta_i, theta_i, p_ii, kappa_i)

    else:
        # Multi-dimension code path
        assert dim_mappings is not None, "dim_mappings required when n_dims > 1"

        # Build kappa lists for E and I sources
        if kappa_per_dim is not None:
            assert len(kappa_per_dim) == n_dims, f"kappa_per_dim length {len(kappa_per_dim)} != n_dims {n_dims}"
            kappa_e_list = kappa_per_dim
            kappa_i_list = kappa_per_dim
        else:
            kappa_e_list = [kappa_e] * n_dims
            kappa_i_list = [kappa_i] * n_dims

        # Get theta arrays for each dimension
        theta_e_list = get_theta_arrays_for_population(n_e, n_dims, dim_mappings.e_shuffles)
        theta_i_list = get_theta_arrays_for_population(n_i, n_dims, dim_mappings.i_shuffles)

        # Generate probability matrices using multiplicative multi-dim combination
        P_ee = get_prob_matrix_multidim(n_e, n_e, theta_e_list, theta_e_list, p_ee, kappa_e_list)
        P_ie = get_prob_matrix_multidim(n_e, n_i, theta_e_list, theta_i_list, p_ie, kappa_e_list)
        P_ei = get_prob_matrix_multidim(n_i, n_e, theta_i_list, theta_e_list, p_ei, kappa_i_list)
        P_ii = get_prob_matrix_multidim(n_i, n_i, theta_i_list, theta_i_list, p_ii, kappa_i_list)

    # 3. Sample actual connections based on P matrices
    ee_cons = np.random.rand(n_e, n_e) < P_ee
    ie_cons = np.random.rand(n_i, n_e) < P_ie
    ei_cons = np.random.rand(n_e, n_i) < P_ei
    ii_cons = np.random.rand(n_i, n_i) < P_ii

    # Prevent self-connections
    np.fill_diagonal(ee_cons, False)
    np.fill_diagonal(ii_cons, False)

    # 4. Generate lognormal weights (independent of tuning)
    ee_vals = np.random.lognormal(w_mu, w_sd, np.sum(ee_cons))
    ie_vals = np.random.lognormal(w_mu, w_sd, np.sum(ie_cons))
    ei_vals = np.random.lognormal(w_mu, w_sd, np.sum(ei_cons)) * i_multiplier
    ii_vals = np.random.lognormal(w_mu, w_sd, np.sum(ii_cons)) * i_multiplier

    # 5. Fill the matrix
    total_weights[:n_e, :n_e][ee_cons] = ee_vals
    total_weights[n_e:, :n_e][ie_cons] = ie_vals
    total_weights[:n_e, n_e:][ei_cons] = ei_vals
    total_weights[n_e:, n_e:][ii_cons] = ii_vals
    
    return total_weights

def gen_recurrent_conn(n_e, n_i, p_ii, p_ei, p_ie, p_ee, w_mu, w_sd, output_dir,
        i_multiplier=10):
    '''
    '''
    n_t = n_e + n_i
    total_weights = np.zeros((n_t, n_t))

    # do each quadrant manually I guess
    # this doesn't guarantee same n_cxns to each cell, like tarek's did.
    ee_cons = np.random.rand(n_e, n_e) < p_ee
    ie_cons = np.random.rand(n_i, n_e) < p_ie
    ei_cons = np.random.rand(n_e, n_i) < p_ei
    ii_cons = np.random.rand(n_i, n_i) < p_ii

    # Prevent self-connections
    np.fill_diagonal(ee_cons, False)
    np.fill_diagonal(ii_cons, False)

    # get values for those
    ee_vals = np.random.lognormal(w_mu, w_sd, np.sum(ee_cons))
    ie_vals = np.random.lognormal(w_mu, w_sd, np.sum(ie_cons))
    ei_vals = np.random.lognormal(w_mu, w_sd, np.sum(ei_cons)) * i_multiplier
    ii_vals = np.random.lognormal(w_mu, w_sd, np.sum(ii_cons)) * i_multiplier

    # and set them in the full matrix
    total_weights[:n_e, :n_e][ee_cons] = ee_vals
    total_weights[n_e:, :n_e][ie_cons] = ie_vals
    total_weights[:n_e, n_e:][ei_cons] = ei_vals
    total_weights[n_e:, n_e:][ii_cons] = ii_vals
    
    return total_weights # should probably save to output dir...
    
    
def gen_input_conn(n_inputs, n_targets, prob_conn, w_mu, w_sd): # I think shape here might be opposite of post-pre situation for recurrent
    '''
    I think this just needs to be an input/n_target matrix...
    '''
    weights = np.zeros((n_inputs, n_targets))
    cons = np.random.rand(n_inputs, n_targets) < prob_conn
    w_vals = np.random.lognormal(w_mu, w_sd, np.sum(cons))
    weights[cons] = w_vals

    return weights


def gen_ring_input_conn(n_inputs, n_targets, prob_conn, w_mu, w_sd, kappa,
                        n_dims=1, dim_mappings=None, kappa_per_dim=None):
    '''
    Generates input connectivity with ring-like tuning structure.

    Connection probability depends on similarity of preferred orientations
    between input and target neurons, controlled by kappa parameter.

    Parameters:
    -----------
    n_inputs : int
        Number of input (Poisson) neurons
    n_targets : int
        Number of target neurons in the network (n_e + n_i)
    prob_conn : float
        Target average connection probability
    w_mu, w_sd : float
        Log-normal weight distribution parameters
    kappa : float
        Concentration parameter for Von Mises distribution.
        kappa = 0 implies uniform random connectivity (equivalent to gen_input_conn).
        Higher kappa implies sharper tuning (connections mostly between similar orientations).
    n_dims : int, optional
        Number of tuning dimensions. Default 1 for backward compatibility.
    dim_mappings : DimensionMappings, optional
        Required if n_dims > 1. Contains shuffle mappings for each dimension.
    kappa_per_dim : list of float, optional
        Kappa value for each dimension. If None, uses kappa for all dimensions.

    Returns:
    --------
    weights : ndarray of shape (n_inputs, n_targets)
        Input weight matrix with lognormal weights where connected
    '''
    if n_dims == 1:
        # Single-dimension code path (original behavior)

        # Assign preferred orientations (0 to pi) to both populations
        theta_input = np.linspace(0, np.pi, n_inputs, endpoint=False)
        theta_target = np.linspace(0, np.pi, n_targets, endpoint=False)

        # Calculate connection probability matrix
        if kappa < 1e-5:
            # Uniform case (kappa ~ 0)
            P = np.full((n_inputs, n_targets), prob_conn)
        else:
            # Grid of orientations: T_target[i,j] = theta_target[j], T_input[i,j] = theta_input[i]
            T_target, T_input = np.meshgrid(theta_target, theta_input)

            # Circular difference on domain [0, pi]
            # Scale by 2 to map to standard Von Mises domain [-pi, pi]
            diff_scaled = 2 * (T_target - T_input)

            # Von Mises PDF for connection probability
            P = vonmises.pdf(diff_scaled, kappa)

            # Scale so average probability equals target prob_conn
            P = P * (prob_conn / np.mean(P))

            # Clip to valid probability range
            P = np.clip(P, 0, 1)

    else:
        # Multi-dimension code path
        assert dim_mappings is not None, "dim_mappings required when n_dims > 1"

        # Build kappa list
        if kappa_per_dim is not None:
            assert len(kappa_per_dim) == n_dims, f"kappa_per_dim length {len(kappa_per_dim)} != n_dims {n_dims}"
            kappa_list = kappa_per_dim
        else:
            kappa_list = [kappa] * n_dims

        # Get theta arrays for inputs
        theta_input_list = get_theta_arrays_for_population(n_inputs, n_dims, dim_mappings.input_shuffles)

        # Get theta arrays for targets (concatenate E and I for each dimension)
        n_e = dim_mappings.n_e
        n_i = dim_mappings.n_i
        theta_target_list = []
        for d in range(n_dims):
            theta_e = get_theta_for_dim(n_e, d, dim_mappings.e_shuffles)
            theta_i = get_theta_for_dim(n_i, d, dim_mappings.i_shuffles)
            theta_target_list.append(np.concatenate([theta_e, theta_i]))

        # Calculate probability matrix using multiplicative multi-dim combination
        # get_prob_matrix_multidim(n_src, n_dest, ...) returns shape (n_dest, n_src)
        # We want shape (n_inputs, n_targets), so pass n_src=n_targets, n_dest=n_inputs
        P = get_prob_matrix_multidim(n_targets, n_inputs, theta_target_list, theta_input_list,
                                      prob_conn, kappa_list)

    # Sample connections based on probability matrix
    cons = np.random.rand(n_inputs, n_targets) < P

    # Generate lognormal weights for existing connections
    weights = np.zeros((n_inputs, n_targets))
    w_vals = np.random.lognormal(w_mu, w_sd, np.sum(cons))
    weights[cons] = w_vals

    return weights
    

class ConnectivityMatrixGenerator(object):

    def __init__(self, n_excite, n_inhib, p_ii, p_ei, p_ie, p_ee, w_mu, w_sd, output_dir):

        # Determine numbers of neurons
        self.n_excite = n_excite
        self.n_inhib = n_inhib
        self.n_neurons = n_excite + n_inhib

        # Initialize connectivity matrix
        self.conn_mat = numpy.zeros((self.n_neurons, self.n_neurons))

        # Initialize weight matrix
        self.weight_mat = numpy.zeros((self.n_neurons, self.n_neurons))

        # Initialize clustering coefficients vector
        self.actual_coefficient = []

        # Calculate total number of connections per neuron (remove neuron from target if included (ee and ii))
        self.k_ii = int(round(p_ii * (self.n_inhib - 1)))
        self.k_ei = int(round(p_ei * self.n_inhib))
        self.k_ie = int(round(p_ie * self.n_excite))
        self.k_ee = int(round(p_ee * (self.n_excite - 1)))

        # Weights distribution
        self.w_mu = w_mu
        self.w_sd = w_sd

        # Define output directory
        self.output_dir = output_dir

    def run_generator(self):

        try:

            # Generate connectivity matrix and check that it's successful
            if not self.generate_conn_mat():
                raise Exception('Failed to generate connectivity matrix.')
            print("Connectivity Matrix Generated")

            # Generate weight matrix and check that it's successful
            if not self.make_weighted():
                raise Exception('Failed to weight connectivity matrix.')
            print("Weighted")

            # Write to file and check that it's successful
            if not self.write_to_file():
                raise Exception('Failed to write files.')

            return True

        except Exception as e:
            print(e)
            return False

    def generate_conn_mat(self):

        try:

            tot = 0

            # E to E connections
            for n in range(0, self.n_excite):
                for a in range(0, self.k_ee):
                    rand = numpy.random.randint(0, self.n_excite)
                    while rand == n or self.conn_mat[n][rand] == 1:
                        rand = numpy.random.randint(0, self.n_excite)
                    self.conn_mat[n][rand] = 1
                    tot += 1

            # E to I connections
            for n in range(0, self.n_excite):
                for a in range(0, self.k_ei):
                    rand = numpy.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                    while self.conn_mat[n][rand] == 1:
                        rand = numpy.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                    self.conn_mat[n][rand] = 1
                    tot += 1

            # I to E connections
            for n in range(0, self.n_inhib):
                for a in range(0, self.k_ie):
                    rand = numpy.random.randint(0, self.n_excite)
                    while self.conn_mat[n + self.n_excite][rand] == 1:
                        rand = numpy.random.randint(0, self.n_excite)
                    self.conn_mat[n + self.n_excite][rand] = 1
                    tot += 1

            # I to I connections
            for n in range(0, self.n_inhib):
                for a in range(0, self.k_ii):
                    rand = numpy.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                    while rand == (n + self.n_excite) or self.conn_mat[n + self.n_excite][rand] == 1:
                        rand = numpy.random.randint(self.n_excite, self.n_excite + self.n_inhib)
                    self.conn_mat[n + self.n_excite][rand] = 1
                    tot += 1

            print(tot)
            print((tot / self.n_neurons) / (self.n_neurons - 1))

            return True

        except Exception as e:
            print(e)
            return False

    def make_weighted(self):

        try:

            # Generate random weights and fill matrix
            for i in range(0, self.n_neurons):
                for j in range(0, self.n_neurons):
                    if self.conn_mat[i][j] == 1:
                        self.weight_mat[i][j] = (numpy.random.lognormal(self.w_mu, self.w_sd))
                        # Make all I 10 times stronger
                        if self.n_neurons > i > (self.n_neurons - self.n_inhib):
                            self.weight_mat[i][j] = self.weight_mat[i][j] * 10

            return True

        except Exception as e:
            print(e)
            return False

    def write_to_file(self):

        try:

            print("Writing conn_mat and weight_mat files...")
            if not path.exists(self.output_dir):
                makedirs(self.output_dir)

            # Save connectivity matrix in csv file
            with open(self.output_dir + '/conn_mat.csv', 'w', newline='') as f:
                writer_f = csv.writer(f, delimiter=',')
                for x in range(0, self.n_neurons):
                    writer_f.writerow(self.conn_mat[x])

            # Save weight matrix in csv file
            with open(self.output_dir + '/weight_mat.csv', 'w', newline='') as f:
                writer_f = csv.writer(f, delimiter=',')
                for x in range(0, self.n_neurons):
                    writer_f.writerow(self.weight_mat[x])

            return True

        except Exception as e:
            print(e)
            return False


if __name__ == '__main__':

    n_excite = 4000
    n_inhib = 1000

    ee_p = 0.30
    ei_p = 0.28
    ie_p = 0.26
    ii_p = 0.20

    w_mu = -0.64
    w_sd = 0.51

    connmat_generator = ConnectivityMatrixGenerator(n_excite, n_inhib, ii_p, ei_p, ie_p, ee_p, w_mu, w_sd, 'ConnMat')

    connmat_generator.run_generator()
