"""Evaluate the LB baseline initialized with the average best k_0 (lb-baseline-k0-average).

Ablation variant of the LB baseline: instead of predicting k_0 with the
regression model, the first LB iteration uses the average of the best k_0
values observed on the training set (see utilities.k_0_bank). With
--enable_merged the average over the merged SC+MIS+CA training set is used
instead of the dataset-specific average.

Only the synthetic datasets are evaluated (the transfer datasets have no
training set to average over). Results are stored under ./result/.
"""

import ecole
import numpy as np
import pyscipopt
import argparse
import gc
from localbranching_ml import RegressionInitialK_KPrime
from utilities import instancetypes, instancesizes, TRANSFER_DATASETS, lbconstraint_mode_for

parser = argparse.ArgumentParser()
parser.add_argument('--t_total', type=int, default=60, help='total time limit (s) per instance')
parser.add_argument('--t_node', type=int, default=10, help='node time limit (s) per LB sub-MIP')
parser.add_argument('--dataset_id', type=int, default=0,
                    help='dataset to evaluate, index into utilities.instancetypes '
                         "(0: 'setcovering', 1: 'independentset', 2: 'combinatorialauction')")
parser.add_argument('--enable_merged', dest='merged', action='store_true',
                    help='use the average best k_0 of the merged training set')
parser.add_argument('--disable_merged', dest='merged', action='store_false')
parser.set_defaults(merged=False)
parser.add_argument('--seed', type=int, default=100, help='Random seed')
args = parser.parse_args()

total_time_limit = args.t_total
node_time_limit = args.t_node
dataset_id = args.dataset_id
merged = args.merged
seed = args.seed

# From the 2nd LB iteration on, k is reset to the default value of the LB baseline.
reset_k_at_2nditeration = True

# The class directory is set up with the large instance size; the actual
# train/test sizes are passed to evaluate_localbranching_baseline_k0_average below.
instance_size = instancesizes[1]

instance_type = instancetypes[dataset_id]
lbconstraint_mode = lbconstraint_mode_for(instance_type)

for test_instance_size in instancesizes:

    for incumbent_mode in ['firstsol', 'rootsol']:

        print(instance_type + test_instance_size)
        print(incumbent_mode)
        print(lbconstraint_mode)

        print('lb_baseline_k0_average started!')
        print('merged :,', merged)

        regression_init_k = RegressionInitialK_KPrime(instance_type, instance_size, lbconstraint_mode,
                                                      incumbent_mode, seed=seed)

        # This ablation is only defined for the synthetic datasets.
        skip_evaluation = instance_type in TRANSFER_DATASETS

        if not skip_evaluation:
            gc.collect()
            regression_init_k.evaluate_localbranching_baseline_k0_average(test_instance_size=test_instance_size,
                                                                          train_instance_size='-small',
                                                                          total_time_limit=total_time_limit,
                                                                          node_time_limit=node_time_limit,
                                                                          reset_k_at_2nditeration=reset_k_at_2nditeration,
                                                                          merged=merged)
        print('lb_baseline_k0_average finished!')
