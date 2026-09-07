# Code and data for learning to search in local branching

## Hardware requirement
Your computer should have at least 2 CPU cores with at least 64 GB memory and 1 GPU (recommended GPU model: Tesla V100) with at least 16 GB memory.

## Prerequisites & Installation

1. Install SCIP 7.03 and Python 3.8.17 with the following libraries (Pytorch 1.7.1, Pytorch Geometric 2.0.2, PySCIPOpt3.1.1, GeCO 1.0.7, numpy 1.21.2, pickleshare 0.7.5, pathlib 1.0.1, scipy 1.10.1, matplotlib 3.4.3, memory-profiler) on your computer. 

2. Install internal library `ecole` according to the following instructions:
    - go into the folder 'ecole'
    - install the package according to 'installation.rst'

## Data Access
The size of the datasets for the evaluated benchmarks (>31GB) in the paper are too large to upload to Github, please contact (defengliu91@gmail.com) for getting the whole dataset. We can either upload those files to your virtual machine server if gvien the access, or share the suppressed files to you through google drive.

## Running the experiments

### Produce results of Section 5.3.1 
#### Evaluating the provided pre-trained ML models on 5 datasets (SC, MIS, CA, GISP, MIPLIB) on your own machine (>=64 GB RAM)
##### i.e. the ML models were pre-trained on our machine, and then you can call them on your own machine without re-training, this is a valid approach to evaluate the generalization performance of our pre-trained models for any new environment)
##### (dataset_id = {0: 'SC', 1:'MIS', 2:'CA', 3: 'GISP', 4:'MIPLIB'})
##### (if you want to evaluate all the datasets all at one run, the machine should have at least 512GB RAM. Otherwise, run each dataset seperately)
```

# evaluate Algorithm lb-baseline, lb-sr, lb-srm. 
# for example, to evaluate SC and LSC Dataset, run:
python evaluation_regression_k_prime.py --t_total=60 --dataset_id=0
...
# to evaluate MIPLIB Dataset, run:
python evaluation_regression_k_prime.py --t_total=60 --dataset_id=4

# evaluate Algorithm lb-rl, lb-srmrl
# for example, for SC and LSC Dataset, run:
python evaluation_reinforce4lb.py --t_total=60 --dataset_id=0
...
# to evaluate MIPLIB Dataset, run:
python evaluation_reinforce4lb.py --t_total=60 --dataset_id=4

# after completing all the datasets (0-4), to print the results of Table 3-8, run:
python compute_evaluation_results.py --mean='geometric'

```

### Produce results of Section 5.3.2
#### Evaluating the provided pre-trained ML models on 2 datasets (GISP, MIPLIB) on your own machine(>=256 GB RAM)
##### (dataset_id = {3: 'GISP', 4:'MIPLIB'})
```
# evaluate Algorithm lb-baseline, lb-sr, lb-srm. 
# to evaluate GISP Dataset, run:
python evaluation_regression_k_prime.py --t_total=600 --dataset_id=3
# to evaluate MIPLIB Dataset, run:
python evaluation_regression_k_prime.py --t_total=600 --dataset_id=4

# evaluate Algorithm lb-rl, lb-srmrl
# for GISP Dataset, run:
python evaluation_reinforce4lb.py --t_total=600 --dataset_id=3
# for MIPLIB Dataset, run:
python evaluation_reinforce4lb.py --t_total=600 --dataset_id=4

# evaluate Algorithm lb-rl, lb-srmrl
# for GISP Dataset, run:
python evaluation_reinforce4lb_kt.py --t_total=600 --dataset_id=3
# for MIPLIB Dataset, run:
python evaluation_reinforce4lb_kt.py --t_total=600 --dataset_id=4


# After completing all the runs above, to print the results of Table 9-10 and plot Figure 4, run:
# ( Figure 4 will be saved in
# "result/plots/plot_primalintegral_miplib_39binary_-small_firstsol_hybrid_rlpolicy-tk_enable-tbaseline_t1seed100_geometric.png" (left)
# "result/plots/plot_primalintegral_miplib_39binary_-small_rootsol_hybrid_rlpolicy-tk_enable-tbaseline_t1seed100_geometric.png" (rigth))
python compute_evaluation_results.py --t_total=600 --mean='geometric'

```

### Produce results of Section 6
#### Evaluating the provided pre-trained ML models on MIPLIB dataset on your own machine(>=128 GB RAM)
### to get the results of all 5 seeds (e.g. 2021,...,2025), change run the algorithms for each seed accordingly.
```
# to evaluate Algorithm scip, run:
python evaluation_scip_baseline.py --t_total=3600 --dataset_id=5 --seed=2021 --enable_gpu
# to evaluate Algorithm scip-lb-regression-rl-single, run:
python evaluation_scip_lb_regression_rl.py --t_total=3600 --dataset_id=5 --freq=0 --seed=2021 --enable_gpu
# to evaluate Algorithm scip-lb-regression-rl-freq1, run:
python evaluation_scip_lb_regression_rl.py --t_total=3600 --dataset_id=5 --freq=1 --seed=2021 --enable_gpu
# to evaluate Algorithm scip-lb-regression-rl-freq100, run:
python evaluation_scip_lb_regression_rl.py --t_total=3600 --dataset_id=5 --freq=100 --seed=2021 --enable_gpu

# After completing all the runs above, run the following script to print the computed metrics for Table 11 and Table 12:
python compute_evaluation_results_scip_seeds_averaged.py --t_total=3600 --mean=geometric  --dataset_id=5 --enable_gpu 


```

### (Optional step, Not recommended) 
#### Train your own regression ML model and RL models on your own machine, then repeat above (for Section 5.3.1, Section 5.3.2, Section 6) to evaluate the results on your machine.
```
# train regression models
python train_regression.py

# train RL models (first the policy for k, then the policy for t)
python train_reinforce4lb_k_policy.py
python train_reinforce4lb_t_policy.py

# repeat the experiments for Section 5.3.1, Section 5.3.2, Section 6

# example for evaluating Algorithm lb-sr, lb-srm by your own regression model, evaluate lb-baseline 
# Parameters:
# --t_total= run time of ach 
# --dataset_id= ID of dataset
# --regression_model_path='path to your own model trained by mixed dataset' # after training, you can select the models from '.results/saved_models/regression/' folder 
python evaluation_regression_k_prime.py --t_total=60 --dataset_id=0 --regression_model_path='path to your saved regression model'

# example for evaluating Algorithm lb-rl, lb-srmrl
evaluation_reinforce4lb.py --t_total=60 --dataset_id=0 --regression_model_path='path to your saved regression model' --rl_model_path='path to your saved RL model for adapting k'

# example for evaluating Algorithm lb-srmrl-adapt-t
evaluation_reinforce4lb_kt.py --t_total=60 --dataset_id=0 --regression_model_path='path to your saved regression model' --rl_k_model_path='path to your saved RL model for adapting k' --rl_t_model_path='path to your saved RL model for adapting t' 

```
