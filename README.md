<<<<<<< HEAD
# Code and data for learning to search in local branching


## Installation

1. Please install Python 3.8, PyscipOPt 3.1.1, and SCIP 7.01 on your computer

2. install package ecole according to the following instructions:
    - go into the folder 'ecole'
    - install the package according to 'installation.rst'

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
```
# to evaluate Algorithm scip, run:
python evaluation_scip_baseline.py
# to evaluate Algorithm scip-lb-regression-rl-single, run:
python evaluation_scip_lb_regression_rl.py --freq=0
# to evaluate Algorithm scip-lb-regression-rl-freq1, run:
python evaluation_scip_lb_regression_rl.py --freq=1
# to evaluate Algorithm scip-lb-regression-rl-freq100, run:
python evaluation_scip_lb_regression_rl.py --freq=100

# After completing all the runs above, to plot Figure 5, run:
# ( Figure 5 will be saved in "result/plots/seed100_primalintegral_miplib_39binary_-small_rootsol_scip_ttotal1200_tnode2_disable_presolve_beforenode_multi_freq-0-1-100_geometric_0.png" )
python compute_evaluation_results_scip_multi_2.py --mean='geometric'

```

### (Optional step, Not recommended) 
#### Train your own regression ML model and RL models on your own machine, then repeat above (for Section 5.3.1, Section 5.3.2, Section 6) to evaluate the results on your machine.
```
# train regression models
train_regression.py

# train RL models
train_reinforce4lb.py

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
=======
boost the search of local branching algorithm with ML.
