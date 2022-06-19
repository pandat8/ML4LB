import ecole
import numpy as np
import pyscipopt
from mllocalbranch_fromfiles import RlLocalbranch
from utilities import instancetypes, instancesizes, incumbent_modes, lbconstraint_modes, t_reward_types
import torch
import random
import argparse


# Argument setting
parser = argparse.ArgumentParser()
# parser.add_argument('--regression_model_path', type = str, default='./result/saved_models/regression/trained_params_mean_setcover-independentset-combinatorialauction_asymmetric_firstsol_k_prime_epoch163.pth')
parser.add_argument('--rl_k_policy_path', type = str, default='./result/saved_models/rl/reinforce/setcovering/checkpoint_trained_reward3_simplepolicy_rl4lb_reinforce_trainset_setcovering-small_lr0.01_epochs7.pth')
parser.add_argument('--seed', type=int, default=100, help='Random seed') #50
parser.add_argument('--t_reward_type', type=int, default=0, help='Reward signal for policy t, 0: reward_k, 1: reward_k + reward_node_time, 2: reward_node_time')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--instance_type', type=int, default=0, help='Instance Type 0: sc, 1: mis, 2: ca, 3: gis, 4: miplib ')
parser.add_argument('--instance_size', type=int, default=0, help='Instance Type 0: -small, 1: -large ')
parser.add_argument('--incumbent_mode', type=int, default=0, help='Instance Type 0: -small, 1: -large ')
parser.add_argument('--t_total', type=int, default=600, help='total time limit')
parser.add_argument('--enable_adapt_t', dest='enable_adapt_t', action='store_true', help='enable enable the hand-made t adaptation policy')
parser.add_argument('--disable_adapt_t', dest='enable_adapt_t', action='store_false')
parser.set_defaults(enable_adapt_t=False)
args = parser.parse_args()

# regression_model_path = args.regression_model_path
rl_k_policy_path = args.rl_k_policy_path
# print(regression_model_path)
print(rl_k_policy_path)

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

t_reward_type = t_reward_types[args.t_reward_type]
print('t_reward_type: ', t_reward_type)
# instance_type = instancetypes[1]

instance_type = instancetypes[args.instance_type]
instance_size = instancesizes[args.instance_size]
incumbent_mode = incumbent_modes[args.incumbent_mode]

# incumbent_mode = 'firstsol'
samples_time_limit = 3

total_time_limit = args.t_total
node_time_limit = 10
print('total_time_limit = ', total_time_limit)
print('initial_node_time_limit = ', node_time_limit)

enable_adapt_t = args.enable_adapt_t

reset_k_at_2nditeration = False
use_checkpoint = False

# eps_list = [0, 0.02]
epsilon = 0.0

# lr_list = [ 0.01] # 0.1, 0.05, 0.01, 0.001,0.0001,1e-5, 1e-6,1e-8
# for lr in lr_list:
lr = args.learning_rate
print('learning rate = ', lr)
print('epsilon = ', epsilon)
# for i in range(4, 5):
#     instance_type = instancetypes[i]
if instance_type == instancetypes[0]:
    lbconstraint_mode = 'asymmetric'
else:
    lbconstraint_mode = 'symmetric'
# for j in range(0, 1):
#     incumbent_mode = incumbent_modes[j]

print(instance_type + instance_size)
print(incumbent_mode)
print(lbconstraint_mode)

reinforce_localbranch = RlLocalbranch(instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed=seed)

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

        # reinforce_localbranch.evaluate_localbranching(evaluation_instance_size='-small', total_time_limit=total_time_limit, node_time_limit=node_time_limit, reset_k_at_2nditeration=reset_k_at_2nditeration)

        # reinforce_localbranch.primal_integral(test_instance_size=instance_size, total_time_limit=total_time_limit, node_time_limit=node_time_limit)
        # regression_init_k.solve2opt_evaluation(test_instance_size='-small')
