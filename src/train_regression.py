"""Train the regression model predicting k of the first LB iteration.

The script runs the full data-generation and training pipeline of the
regression model (see RegressionInitialK_KPrime in ml4lb/localbranching_ml.py).
For each synthetic training dataset ('setcovering', 'independentset',
'combinatorialauction') and each incumbent mode ('firstsol', 'rootsol'):

1. generate_k_samples_k_prime()          - label collection: find the best k
   on the small training instances by running LB probing runs;
2. generate_regression_samples_k_prime() - build the regression samples
   (bipartite graph features plus the collected labels);
3. execute_regression_k_prime()          - train the dataset-specific
   regression model (used by lb-sr);

and finally, over the merged SC+MIS+CA dataset:

4. execute_regression_mergedatasets()    - train the merged regression model
   (used by lb-srm and lb-srmrl).

Trained models are saved under ./result/saved_models/regression/.
"""

import ecole
import numpy as np
import pyscipopt
from ml4lb.localbranching_ml import RegressionInitialK_KPrime
from ml4lb.utilities import instancesizes, SYNTHETIC_DATASETS, lbconstraint_mode_for

# Random seed of the data-collection runs.
seed = 200

# Time limit (s) of each LB probing run used to label an instance with its best k.
samples_time_limit = 3

# Learning rate for training the merged regression model (stage 4).
lr = 0.0001

# Samples are collected on the small training instances.
instance_size = instancesizes[0]

for instance_type in SYNTHETIC_DATASETS:
    lbconstraint_mode = lbconstraint_mode_for(instance_type)

    for incumbent_mode in ['firstsol', 'rootsol']:
        print(incumbent_mode)
        print(lbconstraint_mode)

        regression_init_k = RegressionInitialK_KPrime(instance_type, instance_size, lbconstraint_mode,
                                                      incumbent_mode, seed=seed)

        # stage 1: collect the best-k labels on the training instances
        regression_init_k.generate_k_samples_k_prime(t_limit=samples_time_limit, instance_size=instance_size)
        # stage 2: build the regression samples (graph features + labels)
        regression_init_k.generate_regression_samples_k_prime(t_limit=samples_time_limit, instance_size=instance_size)
        # stage 3: train the dataset-specific regression model
        regression_init_k.execute_regression_k_prime(lr=0.00001, n_epochs=21)

## Optional: reproduce the neighborhood-size illustration of the paper
## (normalized objective and solving time vs. the ratio r, on one set covering
## and one independent set example instance). Requires the k samples of stage 1.
regression_init_k.two_examples()

# stage 4: train the regression model on the merged SC+MIS+CA dataset
regression_init_k = RegressionInitialK_KPrime(instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed=100)
regression_init_k.execute_regression_mergedatasets(lr=lr, n_epochs=301)
