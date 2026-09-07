"""Generate the training data for the regression model predicting k of the first LB iteration.

For each synthetic training dataset ('setcovering', 'independentset',
'combinatorialauction') and each incumbent mode ('firstsol', 'rootsol'),
this script collects the (instance, best k) samples on the small training
instances by running LB with different candidate values of k.

The subsequent training stages are provided by RegressionInitialK_KPrime
(see localbranching_ml.py):

1. generate_k_samples_k_prime()        - label collection (this script);
2. generate_regression_samples_k_prime() - build the regression samples
   (bipartite graph features plus the collected labels);
3. execute_regression_k_prime()        - train the dataset-specific model;
4. execute_regression_mergedatasets()  - train the model on the merged
   SC+MIS+CA dataset (used by lb-srm/lb-srmrl).
"""

import ecole
import numpy as np
import pyscipopt
from localbranching_ml import RegressionInitialK_KPrime
from utilities import instancesizes, SYNTHETIC_DATASETS, lbconstraint_mode_for

# Random seed of the data-collection runs.
seed = 200

# Time limit (s) of each LB probing run used to label an instance with its best k.
samples_time_limit = 3

# Samples are collected on the small training instances.
instance_size = instancesizes[0]

for instance_type in SYNTHETIC_DATASETS:
    lbconstraint_mode = lbconstraint_mode_for(instance_type)

    for incumbent_mode in ['firstsol', 'rootsol']:
        print(incumbent_mode)
        print(lbconstraint_mode)

        regression_init_k = RegressionInitialK_KPrime(instance_type, instance_size, lbconstraint_mode,
                                                      incumbent_mode, seed=seed)

        regression_init_k.generate_k_samples_k_prime(t_limit=samples_time_limit, instance_size=instance_size)
