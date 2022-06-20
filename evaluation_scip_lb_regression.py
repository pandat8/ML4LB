import ecole
import numpy as np
import pyscipopt
import argparse
from execute_heuristics import Execute_LB_Regression
from utilities import instancetypes, instancesizes, incumbent_modes, lbconstraint_modes
import torch
import random
import pathlib
from models import GNNPolicy

# Argument setting
parser = argparse.ArgumentParser()
parser.add_argument('--regression_model_path', type = str, default='./result/saved_models/regression/trained_params_mean_setcover-independentset-combinatorialauction_asymmetric_firstsol_k_prime_epoch163.pth')
parser.add_argument('--rl_model_path', type = str, default='./result/saved_models/rl/reinforce/setcovering/checkpoint_trained_reward3_simplepolicy_rl4lb_reinforce_trainset_setcovering-small_lr0.01_epochs7.pth')

args = parser.parse_args()

regression_model_path = args.regression_model_path
# regression_model_path = './result/saved_models/trained_params_mean_generalized_independentset-small_symmetric_rootsol_k_prime.pth'

rl_model_path = args.rl_model_path
print(regression_model_path)
print(rl_model_path)

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

samples_time_limit = 3

total_time_limit = 1200 # 60 # 600# 60
node_time_limit = 2 #10 # 60 # 5
is_heuristic = True
no_improve_iteration_limit = 2 # 10 # 3
enable_solve_master_problem = True

regression_model_gnn = GNNPolicy()
regression_model_gnn.load_state_dict(torch.load(regression_model_path))


for i in range(4, 5):
    instance_type = instancetypes[i]
    if instance_type == instancetypes[0]:
        lbconstraint_mode = 'asymmetric'
    else:
        lbconstraint_mode = 'symmetric'

    for j in range(0, 2):
        incumbent_mode = incumbent_modes[j]

        for k in range(0, 2):
            instance_size = instancesizes[k]

            print(instance_type + instance_size)
            print(incumbent_mode)


            source_directory = './data/generated_instances/' + instance_type + '/' + instance_size + '/'
            instance_directory = source_directory + 'transformedmodel' + '/' + 'test/'
            solution_directory = source_directory + incumbent_mode + '/' + 'test/'

            evaluation_directory = './result/generated_instances/' + instance_type + '/' + instance_size + '/' + incumbent_mode + '/' + 'scip/'

            if is_heuristic:
                evaluation_directory = evaluation_directory + 'heuristic_mode/'

            result_directory = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
                total_time_limit) + 's' + '-t_node' + str(node_time_limit) + 's' + instance_size + '_lb_k0_regression_beforenode_freq100/seed' + str(seed) + '/'
            pathlib.Path(result_directory).mkdir(parents=True, exist_ok=True) # beforenode_homo

            scip_as_baseline = Execute_LB_Regression(instance_directory,
                                                   solution_directory,
                                                   result_directory,
                                                   lbconstraint_mode=lbconstraint_mode,
                                                   no_improve_iteration_limit=no_improve_iteration_limit,
                                                   seed=seed,
                                                   is_heuristic=is_heuristic,
                                                   instance_type=instance_type,
                                                   incumbent_mode=incumbent_mode,
                                                   regression_model_gnn=regression_model_gnn
                                                   )

            if not ((i == 3 and k == 1) or (i == 4 and k == 1)):
                scip_as_baseline.execute_heuristic_baseline(
                    total_time_limit=total_time_limit,
                    node_time_limit=node_time_limit,
                                                                   )

            # reinforce_localbranch.primal_integral(test_instance_size=instance_size, total_time_limit=total_time_limit, node_time_limit=node_time_limit)
            # reinforce_localbranch.primal_integral_03(test_instance_size=instance_size, total_time_limit=total_time_limit, node_time_limit=node_time_limit)

            # regression_init_k.solve2opt_evaluation(test_instance_size='-small')
