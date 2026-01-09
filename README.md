# SNN-corrs

tweaks of Tarek's original code at github.com/MacLean-Lab-UChicago/sloppySNN.git

looking at stim dependence, in a very sloppy state right now

need to update this when things are in less flux

---

## File Descriptions (AI-generated)

(these are mostly pretty spot-on -- Hal)

### Simulation Core
- **run_sweep.py** - Main entry point for parameter sweeps; handles config parsing, parameter grid generation, and orchestrates parallel simulations
- **fastSNN.py** - Single-trial Brian2 simulation with NetworkSimulator class; exponential integrate-and-fire neurons with adaptation
- **fastSNN_parallel.py** - Parallel simulation using Brian2 standalone mode (recompiles per trial)
- **fastSNN_fastparallel.py** - Optimized parallel simulation using runtime mode; faster than standalone for multi-trial runs
- **ConnMatGenerator.py** - Connectivity matrix generation; supports ring structure (Von Mises tuning via kappa) and random Erdős–Rényi
- **input_generation.py** - Fast vectorized Poisson spike generation with optional spatial structure (Von Mises bump)

### Analysis
- **analysis.py** - analysis: spike loading, binning, correlation computation, variance decomposition
- **corr_analyses.py** - Correlation statistics: ICC, explainable variance proportion, bootstrap methods
- **decode_stimulus.py** - Linear (SVM) and quadratic (QDA) stimulus decoding with cross-validation
- **analyze_shared_input.py** - Analysis of relationship between spike correlations and shared input magnitude

### Visualization
- **sweep_heatmaps.py** - Generate heatmaps of correlation metrics across 2D parameter sweeps
- **rate_heatmaps.py** - Heatmaps of excitatory/inhibitory firing rates across parameter sweeps
- **fano_heatmaps.py** - Heatmaps of Fano factor across parameter sweeps
- **corr_heatmaps.py** - Heatmaps of shared input vs noise correlation relationships
- **Input_plotting.py** - Visualization of input spike patterns (brief, continuous, cyclical)

### Legacy (from original sloppySNN)
- **SNN.py** - Original simulation code with CSV output; slower than fastSNN variants
- **PoissonInputGenerator.py** - Original OO-style input generation; slower than input_generation.py
- **NetworkScore.py** - Original network scoring using Elephant/NEO libraries
- **FIM.py** - Fisher Information Matrix estimation (exploratory)
- **FIM_estimation.py** - FIM computation from parameter sweep results
