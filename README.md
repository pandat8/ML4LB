# Code and data for paper "Revisiting local branching with a machine learning lens"

## Hardware requirement
Your computer should have at least 2 CPU cores with at least 64 GB RAM and 1 GPU (recommended GPU model: Tesla V100) with at least 16 GB memory.

## Prerequisites & Installation

1. Install SCIP 7.03 and Python 3.8.17 with the following libraries (Pytorch 1.7.1, Pytorch Geometric 2.0.2, PySCIPOpt 3.1.1, GeCO 1.0.7, numpy 1.21.2, pickleshare 0.7.5, pathlib 1.0.1, scipy 1.10.1, matplotlib 3.4.3, pandas 2.0.3) on your computer. 

2. Install internal library `ecole` according to the following instructions:
    - go into the folder 'ecole'
    - install the package according to 'installation.rst'

## Data Access
The size of the datasets for the evaluated benchmarks (>31GB) in the paper are too large to upload to Github, please contact (defengliu91@gmail.com) for getting the whole dataset. We can either upload those files to your virtual machine server if given the access, or share the compressed files to you through google drive.

## Code layout

- `src/` contains the runnable scripts: training (`train_*.py`), evaluation (`evaluation_*.py`), and result aggregation (`compute_evaluation_results*.py`, `print_appendix_table.py`).
- `src/ml4lb/` is the library package used by these scripts: the local branching algorithm (`localbranching.py`), the ML training/evaluation library (`localbranching_ml.py`), the SCIP-integration library for Section 6 (`execute_heuristics.py`), the LB primal heuristic plugin (`primal_heur_localbranch.py`), the model definitions (`models.py`, `models_rl.py`), shared helpers (`utilities.py`, `dataset.py`, `event.py`), and the customized ecole environments (`ecole_extend/`).
- `ecole/` contains the ecole package to be installed (see Installation above).
- `result/` contains the pre-trained models and is where evaluation results are written.
- `archived_test_scripts/` contains retired development scripts kept for reference; they are not needed to reproduce the results of the paper.

## Running the experiments

Note: run all commands below from the root folder of this repository (the datasets and results are resolved relative to it, e.g. `./data/...` and `./result/...`).

### Produce results of Section 5.3.1 
#### Evaluating the provided pre-trained ML models on 5 datasets (SC, MIS, CA, GISP, MIPLIB) on your own machine (>=64 GB RAM)
##### i.e. the ML models were pre-trained on our machine, and then you can call them on your own machine without re-training, this is a valid approach to evaluate the generalization performance of our pre-trained models for any new environment.
##### (dataset_id = {0: 'SC', 1:'MIS', 2:'CA', 3: 'GISP', 4:'MIPLIB'})
##### (if you want to evaluate all the datasets all at one run, the machine should have at least 512GB RAM. Otherwise, run each dataset separately)
```

# evaluate Algorithm lb-baseline, lb-sr, lb-srm. 
# for example, to evaluate SC and LSC Dataset, run:
python src/evaluation_regression_k_prime.py --t_total=60 --dataset_id=0
...
# to evaluate MIPLIB Dataset, run:
python src/evaluation_regression_k_prime.py --t_total=60 --dataset_id=4

# evaluate Algorithm lb-rl, lb-srmrl
# for example, for SC and LSC Dataset, run:
python src/evaluation_reinforce4lb.py --t_total=60 --dataset_id=0
...
# to evaluate MIPLIB Dataset, run:
python src/evaluation_reinforce4lb.py --t_total=60 --dataset_id=4

# after completing all the datasets (0-4), to print the results of Table 3-8, run:
python src/compute_evaluation_results.py --mean='geometric'

```

### Produce results of Section 5.3.2
#### Evaluating the provided pre-trained ML models on 2 datasets (GISP, MIPLIB) on your own machine(>=256 GB RAM)
##### (dataset_id = {3: 'GISP', 4:'MIPLIB'})
```
# evaluate Algorithm lb-baseline, lb-sr, lb-srm. 
# to evaluate GISP Dataset, run:
python src/evaluation_regression_k_prime.py --t_total=600 --dataset_id=3
# to evaluate MIPLIB Dataset, run:
python src/evaluation_regression_k_prime.py --t_total=600 --dataset_id=4

# evaluate Algorithm lb-rl, lb-srmrl
# for GISP Dataset, run:
python src/evaluation_reinforce4lb.py --t_total=600 --dataset_id=3
# for MIPLIB Dataset, run:
python src/evaluation_reinforce4lb.py --t_total=600 --dataset_id=4

# evaluate Algorithm lb-rl, lb-srmrl
# for GISP Dataset, run:
python src/evaluation_reinforce4lb_kt.py --t_total=600 --dataset_id=3
# for MIPLIB Dataset, run:
python src/evaluation_reinforce4lb_kt.py --t_total=600 --dataset_id=4


# After completing all the runs above, to print the results of Table 9-10 and plot Figure 4, run:
# ( Figure 4 will be saved in
# "result/plots/plot_primalintegral_miplib_39binary_-small_firstsol_hybrid_rlpolicy-tk_enable-tbaseline_t1seed100_geometric.png" (left)
# "result/plots/plot_primalintegral_miplib_39binary_-small_rootsol_hybrid_rlpolicy-tk_enable-tbaseline_t1seed100_geometric.png" (right))
python src/compute_evaluation_results.py --t_total=600 --mean='geometric'

```

### Produce results of Section 6
#### Evaluating the provided pre-trained ML models on MIPLIB dataset on your own machine(>=128 GB RAM)
### to get the results of all 5 seeds, run the three algorithms below once per seed (--seed=2021, ..., --seed=2025); the final script averages over these seeds.
```
# to evaluate Algorithm scip, run:
python src/evaluation_scip_baseline.py --t_total=3600 --dataset_id=5 --seed=2021 --enable_gpu
# to evaluate Algorithm scip-lb-regression-rl-single, run:
python src/evaluation_scip_lb_regression_rl.py --t_total=3600 --dataset_id=5 --freq=0 --seed=2021 --enable_gpu
# to evaluate Algorithm scip-lb-regression-rl-freq1, run:
python src/evaluation_scip_lb_regression_rl.py --t_total=3600 --dataset_id=5 --freq=1 --seed=2021 --enable_gpu
# to evaluate Algorithm scip-lb-regression-rl-freq100, run:
python src/evaluation_scip_lb_regression_rl.py --t_total=3600 --dataset_id=5 --freq=100 --seed=2021 --enable_gpu

# After completing all the runs above, run the following script to print the computed metrics for Table 11 and Table 12:
python src/compute_evaluation_results_scip_seeds_averaged.py --t_total=3600 --mean=geometric  --dataset_id=5 --enable_gpu 


```

### (Optional step, Not recommended) 
#### This step involves collecting training data collection through heavy optimization runs and training the models. Those heave data collection process can takes a few weeks, therefore it is not recommended to do it by yourself if the purpose is to only validate the models and claims in the paper.  

However, if you have enough time budget to collect your own training data, train your own regression ML model and RL models on your own machine, you can run the following scripts and then repeat above (for Section 5.3.1, Section 5.3.2, Section 6) to evaluate the results on your machine.
```
# train regression models
python src/train_regression.py

# train RL models (first the policy for k, then the policy for t)
python src/train_reinforce4lb_k_policy.py
python src/train_reinforce4lb_t_policy.py

# repeat the experiments for Section 5.3.1, Section 5.3.2, Section 6

# example for evaluating Algorithm lb-sr, lb-srm by your own regression model, evaluate lb-baseline 
# Parameters:
# --t_total= total time limit (s) of each run
# --t_node= node time limit (s) of each LB sub-MIP (optional, the defaults reproduce the paper)
# --dataset_id= ID of dataset
# --regression_model_path='path to your own model trained by mixed dataset' # after training, you can select the models from './result/saved_models/regression/' folder 
python src/evaluation_regression_k_prime.py --t_total=60 --dataset_id=0 --regression_model_path='path to your saved regression model'

# example for evaluating Algorithm lb-rl, lb-srmrl
python src/evaluation_reinforce4lb.py --t_total=60 --dataset_id=0 --regression_model_path='path to your saved regression model' --rl_model_path='path to your saved RL model for adapting k'

# example for evaluating Algorithm lb-srmrl-adapt-t
python src/evaluation_reinforce4lb_kt.py --t_total=60 --dataset_id=0 --regression_model_path='path to your saved regression model' --rl_k_model_path='path to your saved RL model for adapting k' --rl_t_model_path='path to your saved RL model for adapting t' 

```
