"""Print and plot the Section 6 results for a single seed.

Reads the result files produced by evaluation_scip_baseline.py and
evaluation_scip_lb_regression_rl.py (for --freq 0, 1 and 100) for one seed,
and computes the comparison metrics (solving times, primal integral, primal
gap) of the SCIP baseline against the three scip-lb-regression-rl variants.

For the seed-averaged tables of the paper, use
compute_evaluation_results_scip_seeds_averaged.py instead.
"""

import ecole
import numpy as np
import pyscipopt
import argparse
from execute_heuristics import ExecuteHeuristic
from utilities import instancetypes, instancesizes, lbconstraint_mode_for
import torch
import random
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=100, help='Random seed')
parser.add_argument('--mean', type=str, default='geometric',
                    help="averaging mode for the metrics: 'arithmetic' or 'geometric'")
parser.add_argument('--dataset_id', type=int, default=4,
                    help='dataset to aggregate, index into utilities.instancetypes '
                         "(4: 'miplib_39binary', 5: 'miplib2017_binary')")
parser.add_argument('--t_total', type=int, default=600,
                    help='total time limit (s) of the evaluation runs to aggregate')
parser.add_argument('--t_node', type=int, default=2,
                    help='node time limit (s) of the evaluation runs to aggregate')
parser.add_argument('--enable_gpu', action='store_true', help='Enable CUDA GPU acceleration')
args = parser.parse_args()

# Device tag used in the result directory names of the LB runs.
enable_gpu = args.enable_gpu
if enable_gpu and torch.cuda.is_available():
    device_str = 'cuda'
else:
    device_str = 'cpu'

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

dataset_id = args.dataset_id

mean_option = args.mean
print(str(mean_option))

total_time_limit = args.t_total
node_time_limit = args.t_node
print('total time limit:', total_time_limit)
print('node time limit:', node_time_limit)

# The evaluation runs were executed in heuristic mode.
is_heuristic = True

instance_type = instancetypes[dataset_id]
lbconstraint_mode = lbconstraint_mode_for(instance_type)

# Section 6 aggregates the small-size runs started from the root solution.
incumbent_mode = 'rootsol'
instance_size = instancesizes[0]

print(instance_type + instance_size)
print(incumbent_mode)
print(lbconstraint_mode)

plots_directory = './result/plots/'
pathlib.Path(plots_directory).mkdir(parents=True, exist_ok=True)

evaluation_directory = './result/generated_instances/' + instance_type + '/' + instance_size + '/' + incumbent_mode + '/' + 'scip/'
if is_heuristic:
    evaluation_directory = evaluation_directory + 'heuristic_mode/'

# result directory of the SCIP baseline (always run on cpu)
result_directory_1 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
    total_time_limit) + 's' + instance_size + '_scip_baseline' + '-' + 'cpu' + '/seed' + str(seed) + '/'

# result directories of scip-lb-regression-rl with freq 0, 1 and 100
result_directory_2 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
    total_time_limit) + 's' + '-t_node' + str(node_time_limit) + 's' + instance_size + '_lb_k0_regression_rl_beforenode_freq_0' + '-' + device_str + '/seed' + str(seed) + '/'

result_directory_3 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
    total_time_limit) + 's' + '-t_node' + str(
    node_time_limit) + 's' + instance_size + '_lb_k0_regression_rl_beforenode_freq_1' + '-' + device_str + '/seed' + str(
    seed) + '/'

result_directory_4 = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
    total_time_limit) + 's' + '-t_node' + str(
    node_time_limit) + 's' + instance_size + '_lb_k0_regression_rl_beforenode_freq_100' + '-' + device_str + '/seed' + str(
    seed) + '/'

source_directory = './data/generated_instances/' + instance_type + '/' + instance_size + '/'
instance_directory = source_directory + 'transformedmodel' + '/' + 'test/'
solution_directory = source_directory + incumbent_mode + '/' + 'test/'

print(result_directory_1)
print(result_directory_2)
print(result_directory_3)
print(result_directory_4)

run_localbranch = ExecuteHeuristic(instance_type, instance_directory, solution_directory,
                                   result_directory_1, seed=seed)

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
