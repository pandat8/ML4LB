"""Evaluate the local branching heuristic lb-srmrl-adapt-t (RL policies for k and t).

For the selected dataset (--dataset_id, see utilities.instancetypes), the
script evaluates on the test set, for both incumbent modes ('firstsol',
'rootsol') and both instance sizes ('-small', '-large'), the LB heuristic
guided by two pre-trained RL policies: one adapting the neighborhood size k
and one adapting the node time limit t.

Results (primal bound trajectories) are stored under ./result/ and are
aggregated afterwards by compute_evaluation_results.py. See the README for
the exact commands reproducing Section 5.3.2.
"""

import ecole
import numpy as np
import pyscipopt
import argparse
from localbranching_ml import RlLocalbranch
from utilities import instancetypes, instancesizes, t_reward_types, TRANSFER_DATASETS, lbconstraint_mode_for
import torch
import random

parser = argparse.ArgumentParser()
parser.add_argument('--regression_model_path', type=str,
                    default='./result/saved_models/regression/trained_params_mean_setcover-independentset-combinatorialauction_asymmetric_firstsol_k_prime_epoch163.pth',
                    help='path of the pre-trained regression model for predicting k_0')
parser.add_argument('--rl_k_model_path', type=str,
                    default='./result/saved_models/rl/reinforce/setcovering/checkpoint_trained_reward3_simplepolicy_rl4lb_reinforce_trainset_setcovering-small_lr0.01_epochs7.pth',
                    help='path of the pre-trained RL policy for adapting k')
parser.add_argument('--rl_t_model_path', type=str,
                    default='./result/saved_models/rl/reinforce/t_policy/setcovering/t_node10s-t_total600s/checkpoint_rl4lb_trained_-t_policy-simplepolicy-reward_k+t_reinforce_0.1trainset_setcovering-large_firstsol_total_timelimit600s_lr0.1_saved.pth',
                    help='path of the pre-trained RL policy for adapting t')
parser.add_argument('--t_total', type=int, default=60, help='total time limit (s) per instance')
parser.add_argument('--t_node', type=int, default=10, help='node time limit (s) per LB sub-MIP')
parser.add_argument('--dataset_id', type=int, default=0,
                    help='dataset to evaluate, index into utilities.instancetypes '
                         "(0: 'setcovering', 1: 'independentset', 2: 'combinatorialauction', "
                         "3: 'generalized_independentset', 4: 'miplib_39binary')")
parser.add_argument('--t_reward_type', type=int, default=1,
                    help='reward signal used when the t policy was trained, index into '
                         'utilities.t_reward_types (0: reward_k, 1: reward_k + reward_node_time, '
                         '2: reward_node_time)')
parser.add_argument('--enable_adapt_t', dest='enable_adapt_t', action='store_true',
                    help='enable the hand-crafted t adaptation policy')
parser.add_argument('--disable_adapt_t', dest='enable_adapt_t', action='store_false')
parser.set_defaults(enable_adapt_t=True)
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args()

regression_model_path = args.regression_model_path
rl_k_model_path = args.rl_k_model_path
rl_t_model_path = args.rl_t_model_path
print(regression_model_path)
print(rl_k_model_path)
print(rl_t_model_path)

enable_adapt_t = args.enable_adapt_t
print(enable_adapt_t)

t_reward_type = t_reward_types[args.t_reward_type]
print('t_reward_type: ', t_reward_type)

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

total_time_limit = args.t_total
node_time_limit = args.t_node
dataset_id = args.dataset_id

# The RL policies were trained on the small instance size.
instance_size = instancesizes[0]

# From the 2nd LB iteration on, k is reset to the default value of the LB
# baseline before the RL policies take over.
reset_k_at_2nditeration = True

# Learning rates of the (loaded) policy optimizers; the policies are not
# updated during evaluation.
lr = 0.01
lr_t = 0.01

instance_type = instancetypes[dataset_id]
lbconstraint_mode = lbconstraint_mode_for(instance_type)

for test_instance_size in instancesizes:

    for incumbent_mode in ['firstsol', 'rootsol']:

        print(instance_type + test_instance_size)
        print(incumbent_mode)
        print(lbconstraint_mode)

        reinforce_localbranch = RlLocalbranch(instance_type, instance_size, lbconstraint_mode,
                                              incumbent_mode, seed=seed)

        # The large sizes of the transfer datasets (GISP, MIPLIB) are not
        # part of the evaluation (see Section 5.3).
        skip_evaluation = (instance_type in TRANSFER_DATASETS
                           and test_instance_size == instancesizes[1])

        if not skip_evaluation:
            reinforce_localbranch.evaluate_localbranching_rlactive_policy_kt(
                evaluation_instance_size=test_instance_size,
                total_time_limit=total_time_limit,
                node_time_limit=node_time_limit,
                reset_k_at_2nditeration=reset_k_at_2nditeration,
                lr=lr,
                lr_t=lr_t,
                regression_model_path=regression_model_path,
                rl_k_model_path=rl_k_model_path,
                rl_t_model_path=rl_t_model_path,
                t_reward_type=t_reward_type,
                enable_adapt_t=enable_adapt_t
                )
