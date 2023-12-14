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
This script is for printing and plotting the results in Section 6

"""


# Argument setting
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=100, help='Radom seed') #50 101
parser.add_argument('--mean', type = str, default='arithmetic')
parser.add_argument('--dataset_id', type=int, default=4)
parser.add_argument('--t_total', type = int, default=600)
parser.add_argument('--t_node', type = int, default=2)
args = parser.parse_args()

# regression_model_path = args.regression_model_path
# rl_model_path = args.rl_model_path
# print(regression_model_path)
# print(rl_model_path)

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

dataset_id = args.dataset_id

mean_option = args.mean
print(str(mean_option))

samples_time_limit = 3

total_time_limit = args.t_total
node_time_limit = args.t_node
print('total time limit:', total_time_limit)
print('node time limit:', node_time_limit)

is_heuristic = True


instance_type = instancetypes[dataset_id]
if instance_type == instancetypes[0]:
    lbconstraint_mode = 'asymmetric'
else:
    lbconstraint_mode = 'symmetric'

for j in range(1, 2):
    incumbent_mode = incumbent_modes[j]

    for k in range(0, 2):
        instance_size = instancesizes[k]

        print(instance_type + instance_size)
        print(incumbent_mode)
        print(lbconstraint_mode)

        plots_directory = './result/plots/'
        pathlib.Path(plots_directory).mkdir(parents=True, exist_ok=True)

        # result directory of scip baseline
        evaluation_directory = './result/generated_instances/' + instance_type + '/' + instance_size + '/' + incumbent_mode + '/' + 'scip/'
        if is_heuristic:
            evaluation_directory = evaluation_directory + 'heuristic_mode/'

        result_directory_1 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
            total_time_limit) + 's' + instance_size + '_scip_baseline/seed' + str(seed) + '/'



        result_directory_2 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
            total_time_limit) + 's' + '-t_node' + str(node_time_limit) + 's' + instance_size + '_lb_k0_regression_rl_beforenode_freq_0/seed' + str(seed) + '/'

        result_directory_3 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
            total_time_limit) + 's' + '-t_node' + str(
            node_time_limit) + 's' + instance_size + '_lb_k0_regression_rl_beforenode_freq_1/seed' + str(
            seed) + '/'

        result_directory_4 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
            total_time_limit) + 's' + '-t_node' + str(
            node_time_limit) + 's' + instance_size + '_lb_k0_regression_rl_beforenode_freq_100/seed' + str(
            seed) + '/'

        # result_directory_5 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
        #     total_time_limit) + 's' + '-t_node' + str(
        #     node_time_limit) + 's' + instance_size + '_lb_k0_regression_rl_beforenode_freq_1000/seed' + str(
        #     seed) + '/'


        source_directory = './data/generated_instances/' + instance_type + '/' + instance_size + '/'
        instance_directory = source_directory + 'transformedmodel' + '/' + 'test/'
        solution_directory = source_directory + incumbent_mode + '/' + 'test/'

        print(result_directory_1)
        print(result_directory_2)
        print(result_directory_3)
        print(result_directory_4)



        run_localbranch = ExecuteHeuristic(instance_type, instance_directory, solution_directory, result_directory_1, seed=seed)

        if not ((i == 3 and k == 1) or (i == 4 and k == 1)):
            run_localbranch.primal_integral_scip_comparison(
                seed_mcts=seed,
                instance_type=instance_type,
                instance_size=instance_size,
                incumbent_mode=incumbent_mode,
                total_time_limit=total_time_limit,
                node_time_limit=node_time_limit,
                mean_option=mean_option,
                result_directory_1=result_directory_1,
                result_directory_2=result_directory_2,
                result_directory_3=result_directory_3,
                result_directory_4=result_directory_4
                )

