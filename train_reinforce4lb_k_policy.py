"""Train the RL policy (REINFORCE) for adapting the neighborhood size k.

The k-policy is trained on the set covering training instances (large size,
'firstsol' incumbent mode) by running LB episodes and updating the policy
with the REINFORCE objective. Checkpoints are stored under
./result/saved_models/rl/reinforce/.
"""

import ecole
import numpy as np
import pyscipopt
from localbranching_ml import RlLocalbranch
from utilities import instancetypes, instancesizes, incumbent_modes, lbconstraint_mode_for
import torch
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=100, help='Random seed')
parser.add_argument('--t_total', type=int, default=60,
                    help='total time limit (s) of each LB training episode')
parser.add_argument('--t_node', type=int, default=10,
                    help='node time limit (s) per LB sub-MIP')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.0,
                    help='epsilon of the epsilon-greedy exploration')
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

total_time_limit = args.t_total
node_time_limit = args.t_node
lr = args.learning_rate
epsilon = args.epsilon

# The k-policy of the paper is trained on the large set covering instances
# with the incumbent given by the first solution found by SCIP.
instance_type = instancetypes[0]
instance_size = instancesizes[1]
incumbent_mode = incumbent_modes[0]
lbconstraint_mode = lbconstraint_mode_for(instance_type)

# During training, k is not reset at the 2nd iteration and no checkpoint is resumed.
reset_k_at_2nditeration = False
use_checkpoint = False

print('learning rate = ', lr)
print('epsilon = ', epsilon)

print(instance_type + instance_size)
print(incumbent_mode)
print(lbconstraint_mode)

reinforce_localbranch = RlLocalbranch(instance_type, instance_size, lbconstraint_mode,
                                      incumbent_mode, seed=seed)

reinforce_localbranch.train_agent_policy_k(train_instance_size=instance_size,
                                           train_incumbent_mode=incumbent_mode,
                                           total_time_limit=total_time_limit,
                                           node_time_limit=node_time_limit,
                                           reset_k_at_2nditeration=reset_k_at_2nditeration,
                                           lr=lr,
                                           n_epochs=301,
                                           epsilon=epsilon,
                                           use_checkpoint=use_checkpoint
                                           )
