import numpy as np
import matplotlib.pyplot as plt

input_dictionary = np.load('Inputs/sample_brief.npy', allow_pickle=True)[()]
input_loc = input_dictionary[0]
[poisson_inds_b, input_spikes_b, flat_adj_b] = input_loc

input_dictionary = np.load('Inputs/sample_continuous.npy', allow_pickle=True)[()]
input_loc = input_dictionary[0]
[poisson_inds_co, input_spikes_co, flat_adj_co] = input_loc

input_dictionary = np.load('Inputs/sample_cyclical.npy', allow_pickle=True)[()]
input_loc = input_dictionary[0]
[poisson_inds_cy, input_spikes_cy, flat_adj_cy] = input_loc

input_spikes_b = np.array(input_spikes_b)
input_spikes_cy = np.array(input_spikes_cy)
input_spikes_co = np.array(input_spikes_co)

poisson_inds_b = np.array(poisson_inds_b)
poisson_inds_cy = np.array(poisson_inds_cy)
poisson_inds_co = np.array(poisson_inds_co)

test_b = []
test_cy = []
test_co = []
for i in range(50):
    test_b += list(np.where(poisson_inds_b == i)[0])
    test_cy += list(np.where(poisson_inds_cy == i)[0])
    test_co += list(np.where(poisson_inds_co == i)[0])

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.scatter(input_spikes_b[test_b], poisson_inds_b[test_b], marker='|', color='k')
ax1.set_xlim(0, 1050)
ax1.set_ylabel('Neuron ID')
ax1.set_title('Brief')
ax2 = fig.add_subplot(312)
ax2.scatter(input_spikes_cy[test_cy], poisson_inds_cy[test_cy], marker='|', color='k')
ax2.set_xlim(0, 1050)
ax2.set_ylabel('Neuron ID')
ax2.set_title('Cyclical')
ax3 = fig.add_subplot(313)
ax3.scatter(input_spikes_co[test_co], poisson_inds_co[test_co], marker='|', color='k')
ax3.set_xlim(0, 1050)
ax3.set_ylabel('Neuron ID')
ax3.set_xlabel('Time (ms)')
ax3.set_title('Continuous')
plt.tight_layout()
plt.savefig('Inputs/Inputs.tif', facecolor='white', dpi=600)
