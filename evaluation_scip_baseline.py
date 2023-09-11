import ecole
import numpy as np
import pyscipopt
import argparse
from execute_heuristics import ExecuteHeuristic
from utilities import instancetypes, instancesizes, incumbent_modes, lbconstraint_modes
import torch
import random
import pathlib

"""
Run this script for evaluating SCIP baseline
"""

# Argument setting
parser = argparse.ArgumentParser()
parser.add_argument('--regression_model_path', type = str, default='./result/saved_models/regression/trained_params_mean_setcover-independentset-combinatorialauction_asymmetric_firstsol_k_prime_epoch163.pth')
parser.add_argument('--rl_model_path', type = str, default='./result/saved_models/rl/reinforce/setcovering/checkpoint_trained_reward3_simplepolicy_rl4lb_reinforce_trainset_setcovering-small_lr0.01_epochs7.pth')
parser.add_argument('--dataset_id', type=int, default=4)
parser.add_argument('--t_total', type=int, default=1200)
parser.add_argument('--seed', type=int, default=0, help='Radom seed') ## 100 50 101
args = parser.parse_args()

regression_model_path = args.regression_model_path
rl_model_path = args.rl_model_path
print(regression_model_path)
print(rl_model_path)

seed = args.seed # 0 # 120 # 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

dataset_id = args.dataset_id

samples_time_limit = 3

total_time_limit = args.t_total # 1200 # 60 # 600# 60
node_time_limit = 2 #10 # 60 # 5
is_heuristic = True
no_improve_iteration_limit = 10 # 3
enable_solve_master_problem = True


instance_type = instancetypes[dataset_id]
for j in range(1, 2):
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
            total_time_limit) + 's' + instance_size + '_scip_baseline/seed' + str(seed) + '/'
        pathlib.Path(result_directory).mkdir(parents=True, exist_ok=True)

        print(result_directory)
        scip_as_baseline = ExecuteHeuristic(instance_directory,
                                            solution_directory,
                                            result_directory,
                                            no_improve_iteration_limit=no_improve_iteration_limit,
                                            seed=seed)

        if not ((dataset_id == 3 and k == 1) or (dataset_id == 4 and k == 1)):
            scip_as_baseline.execute_heuristic_baseline(
                total_time_limit=total_time_limit,
                node_time_limit=node_time_limit,
                                                               )

        # reinforce_localbranch.primal_integral(test_instance_size=instance_size, total_time_limit=total_time_limit, node_time_limit=node_time_limit)
        # reinforce_localbranch.primal_integral_03(test_instance_size=instance_size, total_time_limit=total_time_limit, node_time_limit=node_time_limit)

        # regression_init_k.solve2opt_evaluation(test_instance_size='-small')
