"""Print and plot the evaluation results of Section 5.

Reads the result files produced by evaluation_regression_k_prime.py,
evaluation_reinforce4lb.py and evaluation_reinforce4lb_kt.py, and computes
the primal integral / primal gap statistics reported in the paper:

- with --t_total=60  : the results of Section 5.3.1 (Tables 3-8);
- with --t_total=600 : the results of Section 5.3.2 (Tables 9-10, Figure 4;
  the figure is saved under ./result/plots/).

Run this script after all evaluation runs listed in the README have finished.
"""

import ecole
import numpy as np
import pyscipopt
from ml4lb.localbranching_ml import RlLocalbranch
from ml4lb.utilities import instancesizes, SYNTHETIC_DATASETS, TRANSFER_DATASETS, lbconstraint_mode_for
import torch
import random
import argparse

# Command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--mean', type=str, default='geometric',
                    help="averaging mode for the metrics: 'arithmetic' or 'geometric'")
parser.add_argument('--t_total', type=int, default=60,
                    help='total time limit (s) of the evaluation runs to aggregate')
parser.add_argument('--t_node', type=int, default=10,
                    help='node time limit (s) of the evaluation runs to aggregate')
args = parser.parse_args()

# Fix all random seeds for reproducibility.
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Averaging mode used for all reported metrics.
mean_option = args.mean
print(str(mean_option))

# The models were trained on the small instance size; only the small size is
# aggregated here (the transfer to large instances is reported separately).
instance_size = instancesizes[0]
test_instance_size = instancesizes[0]

total_time_limit = args.t_total
node_time_limit = args.t_node
print('total time limit:', total_time_limit)
print('node time limit:', node_time_limit)

# Main loop: aggregate the stored results of every dataset and incumbent mode.
for instance_type in SYNTHETIC_DATASETS + TRANSFER_DATASETS:
    lbconstraint_mode = lbconstraint_mode_for(instance_type)

    for incumbent_mode in ['firstsol', 'rootsol']:

        # Log the configuration being aggregated.
        print(instance_type + test_instance_size)
        print(incumbent_mode)
        print(lbconstraint_mode)

        # Construct the aggregation helper for this configuration.
        reinforce_localbranch = RlLocalbranch(instance_type, instance_size, lbconstraint_mode,
                                              incumbent_mode, seed=seed)

        if instance_type in SYNTHETIC_DATASETS:
            # Section 5.3.1, Tables 3-8 (synthetic datasets, 60s runs)
            if total_time_limit == 60:
                reinforce_localbranch.primal_integral(test_instance_size=test_instance_size,
                                                      total_time_limit=total_time_limit,
                                                      node_time_limit=node_time_limit,
                                                      mean_option=mean_option)
        else:
            if total_time_limit == 60:
                # Section 5.3.1, Tables 3-8 (GISP and MIPLIB datasets, 60s runs)
                reinforce_localbranch.primal_integral_03(test_instance_size=test_instance_size,
                                                         total_time_limit=total_time_limit,
                                                         node_time_limit=node_time_limit,
                                                         mean_option=mean_option)
            else:
                # Section 5.3.2, Tables 9-10 and Figure 4 (600s runs)
                reinforce_localbranch.primal_gap_integral_hybrid_03(test_instance_size=instance_size,
                                                                    total_time_limit=total_time_limit,
                                                                    node_time_limit=node_time_limit,
                                                                    mean_option=mean_option)
