"""Train the RL policy (REINFORCE) for adapting the node time limit t.

The t-policy is trained on top of a pre-trained k-policy (--rl_k_policy_path):
during each LB training episode the k actions are selected by the fixed
k-policy while the t-policy is updated with the REINFORCE objective, using
the reward signal selected by --t_reward_type. Checkpoints are stored under
./result/saved_models/rl/reinforce/.
"""

import ecole
import numpy as np
import pyscipopt
from ml4lb.localbranching_ml import RlLocalbranch
from ml4lb.utilities import instancetypes, instancesizes, incumbent_modes, t_reward_types, lbconstraint_mode_for
import torch
import random
import argparse

# Command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--rl_k_policy_path', type=str,
                    default='./result/saved_models/rl/reinforce/setcovering/checkpoint_trained_reward3_simplepolicy_rl4lb_reinforce_trainset_setcovering-small_lr0.01_epochs7.pth',
                    help='path of the pre-trained RL policy for adapting k')
parser.add_argument('--seed', type=int, default=100, help='Random seed')
parser.add_argument('--t_reward_type', type=int, default=0,
                    help='Reward signal for policy t, 0: reward_k, 1: reward_k + reward_node_time, 2: reward_node_time')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--instance_type', type=int, default=0,
                    help='training dataset, index into ml4lb.utilities.instancetypes '
                         "(0: 'setcovering', 1: 'independentset', 2: 'combinatorialauction', "
                         "3: 'generalized_independentset', 4: 'miplib_39binary')")
parser.add_argument('--instance_size', type=int, default=0,
                    help='training instance size, 0: -small, 1: -large')
parser.add_argument('--incumbent_mode', type=int, default=0,
                    help='incumbent mode, 0: firstsol, 1: rootsol')
parser.add_argument('--t_total', type=int, default=600,
                    help='total time limit (s) of each LB training episode')
parser.add_argument('--t_node', type=int, default=10,
                    help='initial node time limit (s) per LB sub-MIP')
parser.add_argument('--enable_adapt_t', dest='enable_adapt_t', action='store_true',
                    help='enable the hand-made t adaptation policy')
parser.add_argument('--disable_adapt_t', dest='enable_adapt_t', action='store_false')
parser.set_defaults(enable_adapt_t=False)
args = parser.parse_args()

rl_k_policy_path = args.rl_k_policy_path
print(rl_k_policy_path)

# Fix all random seeds for reproducibility.
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Training configuration from the command line: reward signal, training
# dataset/size/incumbent mode and the time limits of each episode.
t_reward_type = t_reward_types[args.t_reward_type]
print('t_reward_type: ', t_reward_type)

instance_type = instancetypes[args.instance_type]
instance_size = instancesizes[args.instance_size]
incumbent_mode = incumbent_modes[args.incumbent_mode]

total_time_limit = args.t_total
node_time_limit = args.t_node
print('total_time_limit = ', total_time_limit)
print('initial_node_time_limit = ', node_time_limit)

enable_adapt_t = args.enable_adapt_t

# During training, k is not reset at the 2nd iteration and no checkpoint is resumed.
reset_k_at_2nditeration = False
use_checkpoint = False

epsilon = 0.0
lr = args.learning_rate
print('learning rate = ', lr)
print('epsilon = ', epsilon)

# Select the LB constraint mode used for the training dataset in the paper.
lbconstraint_mode = lbconstraint_mode_for(instance_type)

# Log the training configuration.
print(instance_type + instance_size)
print(incumbent_mode)
print(lbconstraint_mode)

# Construct the trainer for this configuration.
reinforce_localbranch = RlLocalbranch(instance_type, instance_size, lbconstraint_mode,
                                      incumbent_mode, seed=seed)

# Train the t-policy with REINFORCE on top of the fixed k-policy;
# checkpoints are saved under ./result/saved_models/rl/reinforce/.
reinforce_localbranch.train_agent_policy_t(train_instance_size=instance_size,
                                           train_incumbent_mode=incumbent_mode,
                                           total_time_limit=total_time_limit,
                                           node_time_limit=node_time_limit,
                                           reset_k_at_2nditeration=reset_k_at_2nditeration,
                                           lr_t=lr,
                                           n_epochs=301,
                                           epsilon=epsilon,
                                           use_checkpoint=use_checkpoint,
                                           rl_k_policy_path=rl_k_policy_path,
                                           t_reward_type=t_reward_type,
                                           enable_adapt_t=enable_adapt_t
                                           )
