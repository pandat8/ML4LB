"""Evaluate the local branching heuristics lb-rl and lb-srmrl.

For the selected dataset (--dataset_id, see ml4lb.utilities.instancetypes), the
script evaluates on the small test set, for both incumbent modes
('firstsol', 'rootsol'), the RL-guided local branching heuristics:

- lb-rl:    LB with the k updates selected by the pre-trained RL policy;
- lb-srmrl: lb-rl combined with the initial k_0 predicted by the regression
            model trained on the merged SC+MIS+CA dataset.

Results (primal bound trajectories) are stored under ./result/ and are
aggregated afterwards by compute_evaluation_results.py. See the README for
the exact commands reproducing Sections 5.3.1 and 5.3.2.
"""

import ecole
import numpy as np
import pyscipopt
import argparse
from ml4lb.localbranching_ml import RlLocalbranch
from ml4lb.utilities import instancetypes, instancesizes, lbconstraint_mode_for
import torch
import random

# Command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--regression_model_path', type=str,
                    default='./result/saved_models/regression/trained_params_mean_setcover-independentset-combinatorialauction_asymmetric_firstsol_k_prime_epoch163.pth',
                    help='path of the pre-trained regression model for predicting k_0')
parser.add_argument('--rl_model_path', type=str,
                    default='./result/saved_models/rl/reinforce/setcovering/checkpoint_trained_reward3_simplepolicy_rl4lb_reinforce_trainset_setcovering-small_lr0.01_epochs7.pth',
                    help='path of the pre-trained RL policy for adapting k')
parser.add_argument('--t_total', type=int, default=60, help='total time limit (s) per instance')
parser.add_argument('--t_node', type=int, default=10, help='node time limit (s) per LB sub-MIP')
parser.add_argument('--dataset_id', type=int, default=0,
                    help='dataset to evaluate, index into ml4lb.utilities.instancetypes '
                         "(0: 'setcovering', 1: 'independentset', 2: 'combinatorialauction', "
                         "3: 'generalized_independentset', 4: 'miplib_39binary')")
parser.add_argument('--enable_adapt_t', dest='enable_adapt_t', action='store_true',
                    help='enable the hand-crafted t adaptation policy')
parser.add_argument('--disable_adapt_t', dest='enable_adapt_t', action='store_false')
parser.set_defaults(enable_adapt_t=False)
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--enable_gpu', action='store_true', help='Enable CUDA GPU acceleration')
args = parser.parse_args()

# Experiment configuration from the command line.
enable_gpu = args.enable_gpu

regression_model_path = args.regression_model_path
rl_model_path = args.rl_model_path
print(regression_model_path)
print(rl_model_path)

enable_adapt_t = args.enable_adapt_t
print(enable_adapt_t)

# Fix all random seeds for reproducibility.
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

total_time_limit = args.t_total
node_time_limit = args.t_node
dataset_id = args.dataset_id

# The RL policy was trained on the small instance size; the evaluation of
# this script also runs on the small test instances.
instance_size = instancesizes[0]
test_instance_size = instancesizes[0]

# From the 2nd LB iteration on, k is reset to the default value of the LB
# baseline before the RL policy takes over.
reset_k_at_2nditeration = True

# Learning rate of the (loaded) policy optimizer; the policy is not updated
# during evaluation.
lr = 0.01

# Select the dataset and the LB constraint mode used for it in the paper.
instance_type = instancetypes[dataset_id]
lbconstraint_mode = lbconstraint_mode_for(instance_type)

# Main loop: evaluate both incumbent modes ('firstsol', 'rootsol').
for incumbent_mode in ['firstsol', 'rootsol']:

    # Log the configuration of this run.
    print(instance_type + test_instance_size)
    print(incumbent_mode)
    print(lbconstraint_mode)

    # Construct the evaluation runner for this configuration.
    reinforce_localbranch = RlLocalbranch(instance_type, instance_size, lbconstraint_mode,
                                          incumbent_mode, seed=seed, enable_gpu=enable_gpu)

    # Run the RL-guided LB evaluation (lb-rl and lb-srmrl) on the test set;
    # the primal bound trajectories are written under ./result/.
    reinforce_localbranch.evaluate_localbranching_rlactive(
        evaluation_instance_size=test_instance_size,
        total_time_limit=total_time_limit,
        node_time_limit=node_time_limit,
        reset_k_at_2nditeration=reset_k_at_2nditeration,
        lr=lr,
        regression_model_path=regression_model_path,
        rl_model_path=rl_model_path,
        enable_adapt_t=enable_adapt_t
        )
