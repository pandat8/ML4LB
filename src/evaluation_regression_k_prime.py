"""Evaluate the local branching heuristics lb-baseline, lb-sr and lb-srm.

For the selected dataset (--dataset_id, see ml4lb.utilities.instancetypes), the
script evaluates on the test set, for both incumbent modes ('firstsol',
'rootsol') and both instance sizes ('-small', '-large'):

- lb-baseline: LB with the default initial neighborhood size (regre_mode 'baseline');
- lb-sr:      LB with k_0 predicted by the regression model trained on the
              same dataset (regre_mode 'homo');
- lb-srm:     LB with k_0 predicted by the regression model trained on the
              merged SC+MIS+CA dataset (regre_mode 'merged').

Results (primal bound trajectories) are stored under ./result/ and are
aggregated afterwards by compute_evaluation_results.py. See the README for
the exact commands reproducing Sections 5.3.1 and 5.3.2.
"""

import ecole
import numpy as np
import pyscipopt
import argparse
import gc
from ml4lb.localbranching_ml import RegressionInitialK_KPrime
from ml4lb.utilities import instancetypes, instancesizes, regression_modes, SYNTHETIC_DATASETS, TRANSFER_DATASETS, lbconstraint_mode_for

# Command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--regression_model_path', type=str,
                    default='./result/saved_models/regression/trained_params_mean_setcover-independentset-combinatorialauction_asymmetric_firstsol_k_prime_epoch163.pth',
                    help='path of the pre-trained regression model for predicting k_0')
parser.add_argument('--t_total', type=int, default=60, help='total time limit (s) per instance')
parser.add_argument('--t_node', type=int, default=10, help='node time limit (s) per LB sub-MIP')
parser.add_argument('--dataset_id', type=int, default=0,
                    help='dataset to evaluate, index into ml4lb.utilities.instancetypes '
                         "(0: 'setcovering', 1: 'independentset', 2: 'combinatorialauction', "
                         "3: 'generalized_independentset', 4: 'miplib_39binary')")
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--enable_gpu', action='store_true', help='Enable CUDA GPU acceleration')
args = parser.parse_args()

# Experiment configuration from the command line.
enable_gpu = args.enable_gpu

regression_model_path = args.regression_model_path
print(regression_model_path)

total_time_limit = args.t_total
node_time_limit = args.t_node
dataset_id = args.dataset_id
seed = args.seed

# From the 2nd LB iteration on, k is reset to the default value of the LB
# baseline (the ML prediction is only used for the first iteration).
reset_k_at_2nditeration = True

# The class directory is set up with the large instance size; the actual
# train/test sizes are passed to evaluate_localbranching_k_prime below.
instance_size = instancesizes[1]

# Select the dataset and the LB constraint mode used for it in the paper.
instance_type = instancetypes[dataset_id]
lbconstraint_mode = lbconstraint_mode_for(instance_type)

# Main loop: evaluate every combination of test instance size ('-small',
# '-large'), incumbent mode ('firstsol', 'rootsol') and algorithm variant
# (regre_mode 'homo' = lb-sr, 'merged' = lb-srm, 'baseline' = lb-baseline).
for test_instance_size in instancesizes:

    for incumbent_mode in ['firstsol', 'rootsol']:

        # Log the configuration of this run.
        print(instance_type + test_instance_size)
        print(incumbent_mode)
        print(lbconstraint_mode)

        # A homogeneous ('homo') regression model only exists for the
        # synthetic training datasets; all other datasets start at 'merged'.
        if instance_type in SYNTHETIC_DATASETS:
            eval_regression_modes = regression_modes       # ['homo', 'merged', 'baseline']
        else:
            eval_regression_modes = regression_modes[1:]   # ['merged', 'baseline']

        for regre_mode in eval_regression_modes:
            # Translate the regression mode into the flags of
            # evaluate_localbranching_k_prime: 'homo' evaluates lb-sr,
            # 'merged' evaluates lb-srm, and 'baseline' evaluates lb-baseline.
            if regre_mode == 'homo':
                merged = False
                baseline = False
            elif regre_mode == 'merged':
                merged = True
                baseline = False
            elif regre_mode == 'baseline':
                baseline = True

            print('merged :,', merged)
            print('baseline :', baseline)

            # Construct the evaluation runner for this configuration.
            regression_init_k = RegressionInitialK_KPrime(instance_type, instance_size, lbconstraint_mode,
                                                          incumbent_mode, seed=seed, enable_gpu=enable_gpu)

            # The large sizes of the transfer datasets are not part of the
            # evaluation (see Section 5.3).
            skip_evaluation = instance_type in TRANSFER_DATASETS and (
                test_instance_size == instancesizes[1] or regre_mode == 'homo')

            # Run the LB evaluation on the test set; the primal bound
            # trajectories are written under ./result/.
            if not skip_evaluation:
                gc.collect()
                regression_init_k.evaluate_localbranching_k_prime(test_instance_size=test_instance_size,
                                                                  train_instance_size='-small',
                                                                  total_time_limit=total_time_limit,
                                                                  node_time_limit=node_time_limit,
                                                                  reset_k_at_2nditeration=reset_k_at_2nditeration,
                                                                  merged=merged,
                                                                  baseline=baseline,
                                                                  regression_model_path=regression_model_path)
