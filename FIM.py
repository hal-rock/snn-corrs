__author__ = 'Tarek'

from SNN import run_sims
from SNN import NetworkSimulator
import PoissonInputGenerator as poisson
from scipy.stats import linregress
import shutil
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def likelihood(parameter_vector):

    ee_p = parameter_vector[0]
    ei_p = parameter_vector[1]
    ie_p = parameter_vector[2]
    ii_p = parameter_vector[3]
    w_mu = best_mu  # parameter_vector[4]
    w_sd = best_sd  # parameter_vector[5]

    if len(testing) == 0:
        av = [0, 0, 0, 0]
    else:
        dir_path = 'FIM_results/' \
                   'ee_' + str(round(ee_p, 4)) + '_ei_' + str(round(ei_p, 4)) + '_ie_' + str(round(ie_p, 4)) + '_ii_' + str(round(ii_p, 4)) \
                + '_mu_' + str(round(w_mu, 4)) + '_sd_' + str(round(w_sd, 4))

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        e_fr = 0
        i_fr = 0
        num_sustained = 0
        synch = 0

        failed = 0

        for ppp in range(0, num_trials):

            try:
                directory_name = dir_path + '/trial_' + str(ppp)
                result = run_sims(ii_p, ie_p, ei_p, ee_p, n_excite, n_inhib, directory_name, num_sims_per, input_dictionary[ppp], n_input,
                                  input_time, best_mu, best_sd, simulation_time)

                print(result)

                if result:
                    e_fr += result[0]
                    i_fr += result[1]
                    if result[2] > 1045:
                        num_sustained += 1
                    synch += result[3].real
                else:
                    failed += 1
                    print('No results')

                shutil.rmtree(directory_name)

            except OSError:
                # solution if I can't find why OSError 24;
                # pass  # TODO- change in SNN_gpu and scp to 205.208.22.226
                print("00000")

        if failed != num_trials:
            av_1 = e_fr / (num_trials-failed)
            av_2 = i_fr / (num_trials-failed)
            av_3 = num_sustained / (num_trials-failed)
            av_4 = synch / (num_trials-failed)
        else:
            av_1 = 0
            av_2 = 0
            av_3 = 0
            av_4 = 0

        av = [av_1, av_2, av_3, av_4]

        all_e_fr.append(av_1)
        all_i_fr.append(av_2)
        all_sustained.append(av_3)
        all_synch.append(av_4)

    print(av)

    return av[0]


if __name__ == '__main__':

    num_trials = 5  # For each parameters vector, generate 5 connectivity matrix  Todo- change to 10
    num_sims_per = 1  # and run with each once

    n_excite = 4000
    n_inhib = 1000

    simulation_time = 1.05

    n_input = 3000
    n_target = n_excite + n_inhib
    input_time = 1.050
    input_rate_mean = 2.8
    input_rate_std = 0.3
    input_connectivity_p = 0.100  # TODO- CAREFUL
    input_weight_multiplier = 1

    # From grid search:
    best_ee = 0.120  # 0.059
    best_ei = 0.160  # 0.076
    best_ie = 0.240  # 0.117
    best_ii = 0.240  # 0.080
    best_mu = -0.64
    best_sd = 0.51

    # Directory name
    directory = 'FIM/Input_' + str(input_time) + '_in_' + str(input_connectivity_p)\
                + "_at_ee_" + str(best_ee) + "_ei_" + str(best_ei) + "_ie_" + str(best_ie) + "_ii_" + str(best_ii)
    if not os.path.exists(directory):
        os.mkdir(directory)

    # Generate new inputs
    '''
    input_dictionary = {}
    for n in range(0, num_trials):
        print("Generating Poisson Input")
        poisson_input_loc = 'output/poisson_inputs_' + str(n) + '.csv'
        poisson_generator = poisson.PoissonInputGenerator(n_input, n_target, n_excite + n_inhib, input_connectivity_p, input_rate_mean,
                                                          input_rate_std, input_time, poisson_input_loc, input_weight_multiplier, cyclical=False)
        input_dictionary[n] = poisson_generator.generate_inputs()
    np.save('Inputs/Inputs_' + str(input_time) + '_conn_' + str(input_connectivity_p), input_dictionary)
    '''
    # Use old inputs
    input_dictionary = np.load('Inputs/Inputs_' + str(input_time) + '_conn_' + str(input_connectivity_p) + '.npy', allow_pickle=True).item()

    engine = NetworkSimulator('Logs/current_log.log', 'DEBUG', simulation_time)

    # Calculate the 4 gradients for each parameter
    for www in range(0, 4):

        # To record likelihoods
        ll = []

        # To record score (to estimate covariance later)
        all_e_fr = []
        all_i_fr = []
        all_sustained = []
        all_synch = []

        # To summarize scores
        scores = []

        if www == 0:
            testing = [best_ee - 0.01, best_ee - 0.005, best_ee - 0.002, best_ee - 0.001, best_ee - 0.0005, best_ee - 0.0002, best_ee - 0.0001,
                       best_ee,
                       best_ee + 0.0001, best_ee + 0.0002, best_ee + 0.0005, best_ee + 0.001, best_ee + 0.002, best_ee + 0.005, best_ee + 0.01]
            np.save(directory + '/ee_range', testing)

            for ee_ee in testing:
                likelihood([ee_ee, best_ei, best_ie, best_ii])

            dy_dee = [0, 0, 0, 0]
            dy_dee[0], intercept, r_value, p_value, std_err = linregress(testing, all_e_fr)
            dy_dee[1], intercept, r_value, p_value, std_err = linregress(testing, all_i_fr)
            dy_dee[2], intercept, r_value, p_value, std_err = linregress(testing, all_sustained)
            dy_dee[3], intercept, r_value, p_value, std_err = linregress(testing, all_synch)
            np.save(directory + '/dy_dee', dy_dee)

            scores = np.array([all_e_fr, all_i_fr, all_sustained, all_synch])
            np.save(directory + '/Scores_for_ee_slopes', scores)

        if www == 1:
            testing = [best_ei - 0.01, best_ei - 0.005, best_ei - 0.002, best_ei - 0.001, best_ei - 0.0005, best_ei - 0.0002, best_ei - 0.0001,
                       best_ei,
                       best_ei + 0.0001, best_ei + 0.0002, best_ei + 0.0005, best_ei + 0.001, best_ei + 0.002, best_ei + 0.005, best_ei + 0.01]
            np.save(directory + '/ei_range', testing)

            for ei_ei in testing:
                likelihood([best_ee, ei_ei, best_ie, best_ii])

            dy_dei = [0, 0, 0, 0]
            dy_dei[0], intercept, r_value, p_value, std_err = linregress(testing, all_e_fr)
            dy_dei[1], intercept, r_value, p_value, std_err = linregress(testing, all_i_fr)
            dy_dei[2], intercept, r_value, p_value, std_err = linregress(testing, all_sustained)
            dy_dei[3], intercept, r_value, p_value, std_err = linregress(testing, all_synch)
            np.save(directory + '/dy_dei', dy_dei)

            scores = np.array([all_e_fr, all_i_fr, all_sustained, all_synch])
            np.save(directory + '/Scores_for_ei_slopes', scores)

        if www == 2:
            testing = [best_ie - 0.01, best_ie - 0.005, best_ie - 0.002, best_ie - 0.001, best_ie - 0.0005, best_ie - 0.0002, best_ie - 0.0001,
                       best_ie,
                       best_ie + 0.0001, best_ie + 0.0002, best_ie + 0.0005, best_ie + 0.001, best_ie + 0.002, best_ie + 0.005, best_ie + 0.01]
            np.save(directory + '/ie_range', testing)

            for ie_ie in testing:
                likelihood([best_ee, best_ei, ie_ie, best_ii])

            dy_die = [0, 0, 0, 0]
            dy_die[0], intercept, r_value, p_value, std_err = linregress(testing, all_e_fr)
            dy_die[1], intercept, r_value, p_value, std_err = linregress(testing, all_i_fr)
            dy_die[2], intercept, r_value, p_value, std_err = linregress(testing, all_sustained)
            dy_die[3], intercept, r_value, p_value, std_err = linregress(testing, all_synch)
            np.save(directory + '/dy_die', dy_die)

            scores = np.array([all_e_fr, all_i_fr, all_sustained, all_synch])
            np.save(directory + '/Scores_for_ie_slopes', scores)

        if www == 3:
            testing = [best_ii - 0.01, best_ii - 0.005, best_ii - 0.002, best_ii - 0.001, best_ii - 0.0005, best_ii - 0.0002, best_ii - 0.0001,
                       best_ii,
                       best_ii + 0.0001, best_ii + 0.0002, best_ii + 0.0005, best_ii + 0.001, best_ii + 0.002, best_ii + 0.005, best_ii + 0.01]
            np.save(directory + '/ii_range', testing)

            for ii_ii in testing:
                likelihood([best_ee, best_ei, best_ie, ii_ii])

            dy_dii = [0, 0, 0, 0]
            dy_dii[0], intercept, r_value, p_value, std_err = linregress(testing, all_e_fr)
            dy_dii[1], intercept, r_value, p_value, std_err = linregress(testing, all_i_fr)
            dy_dii[2], intercept, r_value, p_value, std_err = linregress(testing, all_sustained)
            dy_dii[3], intercept, r_value, p_value, std_err = linregress(testing, all_synch)
            np.save(directory + '/dy_dii', dy_dii)

            scores = np.array([all_e_fr, all_i_fr, all_sustained, all_synch])
            np.save(directory + '/Scores_for_ii_slopes', scores)

    sco_gra = np.array([[dy_dee[0], dy_dei[0], dy_die[0], dy_dii[0]], [dy_dee[1], dy_dei[1], dy_die[1], dy_dii[1]],
                        [dy_dee[2], dy_dei[2], dy_die[2], dy_dii[2]], [dy_dee[3], dy_dei[3], dy_die[3], dy_dii[3]]])

    sco_cov = np.load('Covariances/Input_' + str(input_time) + '_in_' + str(input_connectivity_p) + '/Covariance.npy')
    K = np.linalg.inv(sco_cov)

    num_parameters = 4
    num_scores = 4

    H = np.zeros((num_parameters, num_parameters))
    for m in range(0, num_parameters):
        for n in range(0, num_parameters):
            for j in range(0, num_scores):
                for i in range(0, num_scores):
                    H[m][n] += -sco_gra[i][m] * sco_gra[j][n] * K[i][j]
    np.save(directory + '/Hessian_estimate', H)

    F = - H
    np.save(directory + '/FIM_estimate', F)

    # Calculate the eigenvalues and eigenvectors of the Hessian
    vals, vec = np.linalg.eig(F)

    print(vals)
    if np.all(vals > -1e-8):
        print('All good')
    else:
        print('FIM is not positive semi-definite')

    np.save(directory + '/FIM_eigenvalues', vals)
    np.save(directory + '/FIM_eigenvectors', vec)

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [5, 1, 5]})
    fim = ax1.imshow(F, cmap='hot', interpolation='nearest')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(fim, cax=cax, orientation='vertical')
    ax1.set_title('FIM', fontsize=14)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.plot([' ', ' ', ' ', ' '], vals/max(vals), '_', markersize='20')
    ax2.set_yscale('log')
    ax2.set_xticks([])
    ax2.set_title('Eigenvalues', fontsize=14)
    eigenvec = ax3.imshow(vec, cmap='winter', interpolation='nearest', extent=[-1, 1, -1, 1])
    divider2 = make_axes_locatable(ax3)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(eigenvec, cax=cax2, orientation='vertical')
    ax3.set_title('Eigenvectors', fontsize=14)
    ax3.set_xticks([-0.5, 0, 0.5])
    ax3.set_xticklabels([])
    ax3.tick_params(axis='x', color=(0, 0, 0, 0))
    ax3.set_yticks([])
    ax3.grid(which='both', axis='x', color='w', linestyle='-', linewidth=3)
    ax3.set_frame_on(False)
    plt.subplots_adjust(hspace=0, top=3)
    plt.tight_layout()
    plt.savefig(directory + '/FIM, Eigenvalues, Eigenvectors')
