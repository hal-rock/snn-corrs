# SNN

Code used for SNN simulations to generate results reported in:

Jabri, T., & MacLean, J. N. (2021). Large-scale algorithmic search identifies stiff and sloppy dimensions in synaptic architectures consistent with murine neocortical wiring. _bioRxiv_.

To run a simulation, use SNN.py. 
You will need to start by calling ConnMatGenerator.py to generate the connectivity matrix and PoissonInputGenertor.py to generate the input. 
Then, it will use NetworkScore.py to score the output.

Use FIM.py to run all the simulations needed to calculate the FIM at any combination of parameters.

FIM_estimation.py is for estimating the FIM given the scores of all the simulations.

"brief_input_classifier" contains the scaler and classifier needed for scoring the runs using brief input. They are used by SNN.py.

"Covariances" includes the covariance matrix needed to estimate the FIM using continuous input. 
The rest of the files in the folder include the results of the simulations needed to estimate the covariance and the Hessian matrix estimated using an older (incorrect) method.

The code itself should be self-explanatory.