import numpy as np
from scipy.stats import linregress

directory = 'Input_1.05_in_0.1_at_ee_0.059_ei_0.076_ie_0.117_ii_0.08/'

num_parameters = 4
num_scores = 4

ranges_loc = ['ee_range.npy', 'ei_range.npy', 'ie_range.npy', 'ii_range.npy']
scores_loc = ['Scores_for_ee_slopes.npy', 'Scores_for_ei_slopes.npy',
              'Scores_for_ie_slopes.npy', 'Scores_for_ii_slopes.npy']
ranges = []
scores = []
for i in range(num_parameters):
    ranges.append(np.load(directory + ranges_loc[i]))
    scores.append(np.load(directory + scores_loc[i]))

dy = []
for i in range(num_parameters):
    di = []
    for j in range(num_scores):
        res = linregress(ranges[i], scores[i][j])
        di.append(res.slope)
    dy.append(di)
sco_gra = np.transpose(dy)

sco_cov = np.load('Covariances/Input_1.05_in_0.1/Covariance.npy')
K = np.linalg.inv(sco_cov)

H = np.zeros((num_parameters, num_parameters))
for m in range(0, num_parameters):
    for n in range(0, num_parameters):
        for j in range(0, num_scores):
            for i in range(0, num_scores):
                H[m][n] += -sco_gra[i][m] * sco_gra[j][n] * K[i][j]

F = - H
vals, vec = np.linalg.eig(F)
