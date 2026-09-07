"""Evaluate the SCIP baseline of Section 6.

For the selected dataset (--dataset_id, default 'miplib_39binary'), plain
SCIP solves every test instance, warm-started from the stored incumbent
solution ('firstsol' and 'rootsol'), with the given total time limit.

The primal bound trajectories are stored under ./result/ and are aggregated
afterwards by compute_evaluation_results_scip_seeds_averaged.py. See the
README for the exact commands reproducing Section 6.
"""

import ecole
import numpy as np
import pyscipopt
import argparse
from execute_heuristics import ExecuteHeuristic
from utilities import instancetypes, instancesizes, TRANSFER_DATASETS
import torch
import random
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_id', type=int, default=4,
                    help='dataset to evaluate, index into utilities.instancetypes '
                         "(4: 'miplib_39binary', 5: 'miplib2017_binary')")
parser.add_argument('--t_total', type=int, default=600, help='total time limit (s) per instance')
parser.add_argument('--t_node', type=int, default=2, help='node time limit (s) per LB sub-MIP')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--enable_gpu', action='store_true', help='Enable CUDA GPU acceleration')
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

dataset_id = args.dataset_id

total_time_limit = args.t_total
node_time_limit = args.t_node

# Run SCIP in heuristic mode (the LB heuristic result directories use the
# same layout, which makes the comparison scripts uniform).
is_heuristic = True

# The LB search stops after this many consecutive non-improving iterations.
no_improve_iteration_limit = 10

# Device tag used in the result directory name.
enable_gpu = args.enable_gpu
if enable_gpu and torch.cuda.is_available():
    device_str = 'cuda'
else:
    device_str = 'cpu'

instance_type = instancetypes[dataset_id]
for incumbent_mode in ['firstsol', 'rootsol']:

    for instance_size in instancesizes:

        print(instance_type + instance_size)
        print(incumbent_mode)

        source_directory = './data/generated_instances/' + instance_type + '/' + instance_size + '/'
        instance_directory = source_directory + 'transformedmodel' + '/' + 'test/'
        solution_directory = source_directory + incumbent_mode + '/' + 'test/'

        evaluation_directory = './result/generated_instances/' + instance_type + '/' + instance_size + '/' + incumbent_mode + '/' + 'scip/'

        if is_heuristic:
            evaluation_directory = evaluation_directory + 'heuristic_mode/'

        result_directory = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
            total_time_limit) + 's' + instance_size + '_scip_baseline' + '-' + device_str + '/seed' + str(seed) + '/'
        pathlib.Path(result_directory).mkdir(parents=True, exist_ok=True)

        print(result_directory)
        scip_as_baseline = ExecuteHeuristic(instance_type,
                                            instance_directory,
                                            solution_directory,
                                            result_directory,
                                            no_improve_iteration_limit=no_improve_iteration_limit,
                                            seed=seed)

        # The large sizes of the transfer datasets are not part of the evaluation.
        skip_evaluation = (instance_type in TRANSFER_DATASETS
                           and instance_size == instancesizes[1])

        if not skip_evaluation:
            scip_as_baseline.execute_heuristic_baseline(
                total_time_limit=total_time_limit,
                node_time_limit=node_time_limit,
                )
