import ecole
import numpy as np
import pyscipopt
import argparse
from execute_heuristics import ExecuteHeuristic
from utilities import instancetypes, instancesizes, incumbent_modes, lbconstraint_modes
import torch
import random
import pathlib

# Argument setting
parser = argparse.ArgumentParser()
# parser.add_argument('--regression_model_path', type = str, default='./result/saved_models/regression/trained_params_mean_setcover-independentset-combinatorialauction_asymmetric_firstsol_k_prime_epoch163.pth')
# parser.add_argument('--rl_model_path', type = str, default='./result/saved_models/rl/reinforce/setcovering/checkpoint_trained_reward3_simplepolicy_rl4lb_reinforce_trainset_setcovering-small_lr0.01_epochs7.pth')

args = parser.parse_args()

# regression_model_path = args.regression_model_path
# rl_model_path = args.rl_model_path
# print(regression_model_path)
# print(rl_model_path)

seed = 0
seed_mcts = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

samples_time_limit = 3

total_time_limit = 60
node_time_limit = 10
is_heuristic = True

for i in range(3, 5):
    instance_type = instancetypes[i]
    if instance_type == instancetypes[0]:
        lbconstraint_mode = 'asymmetric'
    else:
        lbconstraint_mode = 'symmetric'

    for j in range(0, 2):
        incumbent_mode = incumbent_modes[j]

        for k in range(0, 1):
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
            result_directory_4 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(
                total_time_limit) + 's' + instance_size + '_scip_baseline/seed' + str(seed) + '/'
            pathlib.Path(result_directory_4).mkdir(parents=True, exist_ok=True)

            # result directory of localbranch
            if is_heuristic:
                evaluation_directory = evaluation_directory + 'heuristic_mode/'

            result_directory_2 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
                total_time_limit) + 's' + instance_size + '_lb_baseline/seed' + str(seed) + '/'

            # result directory of lns-random, scip-lb-regressiono

            if is_heuristic:
                evaluation_directory = evaluation_directory + 'heuristic_mode/'

            result_directory_1 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
                total_time_limit) + 's' + instance_size + '_lb_k0_regression/seed' + str(seed) + '/'

            # result directory of lns-lb, scip-lb-rl
            if is_heuristic:
                evaluation_directory = evaluation_directory + 'heuristic_mode/'

            result_directory_3 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
                total_time_limit) + 's' + instance_size + '_lb_k0_rl/seed' + str(seed) + '/'

            # # result directory of lns-lblp
            # evaluation_directory = './result/generated_instances/' + instance_type + '/' + instance_size + '/' + incumbent_mode + '/' + 'lns' + '/'
            # if is_heuristic:
            #     evaluation_directory = evaluation_directory + 'heuristic_mode/'
            # result_directory_3 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_node' + str(
            #     node_time_limit) + 's' + '-t_total' + str(
            #     total_time_limit) + 's' + instance_size + '_lns_fixedbylblp_baseline/seed' + str(seed_mcts) + '/'
            # pathlib.Path(result_directory_3).mkdir(parents=True, exist_ok=True)

            # result directory of lns-lb-mcts, scip-lb-regression-rl
            if is_heuristic:
                evaluation_directory = evaluation_directory + 'heuristic_mode/'

            result_directory_5 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
                total_time_limit) + 's' + instance_size + '_lb_k0_regression_rl/seed' + str(seed) + '/'

            source_directory = './data/generated_instances/' + instance_type + '/' + instance_size + '/'
            instance_directory = source_directory + 'transformedmodel' + '/' + 'test/'
            solution_directory = source_directory + incumbent_mode + '/' + 'test/'

            run_localbranch = ExecuteHeuristic(instance_directory, solution_directory, result_directory_1, seed=seed)

            if not ((i == 3 and k == 1) or (i == 4 and k == 1)):
                run_localbranch.primal_integral_03(
                    seed_mcts=seed_mcts,
                    instance_type=instance_type,
                    instance_size=instance_size,
                    incumbent_mode=incumbent_mode,
                    total_time_limit=total_time_limit,
                    node_time_limit=node_time_limit,
                    result_directory_1=result_directory_1,
                    result_directory_2=result_directory_2,
                    result_directory_3=result_directory_3,
                    result_directory_4=result_directory_4,
                    result_directory_5=result_directory_5
                    )

