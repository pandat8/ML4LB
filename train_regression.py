import ecole
import numpy as np
import pyscipopt
from mllocalbranch_fromfiles import RegressionInitialK_KPrime
from utility import instancetypes, instancesizes, incumbent_modes, lbconstraint_modes, regression_mode

instance_size = instancesizes[0]
instance_type = instancetypes[0]
incumbent_mode = incumbent_modes[0]
lbconstraint_mode = 'asymmetric'
samples_time_limit = 3
node_time_limit = 10

total_time_limit = 60
reset_k_at_2nditeration = True

lr = 0.0001
print('learning rate:', lr)

for i in range(0, 3):
    instance_type = instancetypes[i]
    if instance_type == instancetypes[0]:
        lbconstraint_mode = 'asymmetric'
    else:
        lbconstraint_mode = 'symmetric'
    for j in range(0, 2):
        incumbent_mode = incumbent_modes[j]
        print(incumbent_mode)
        print(lbconstraint_mode)

        regression_init_k = RegressionInitialK_KPrime(instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed=100)

        # regression_init_k.generate_k_samples_k_prime(t_limit=samples_time_limit, instance_size=instance_size)
        # regression_init_k.two_examples()
        regression_init_k.generate_regression_samples_k_prime(t_limit=samples_time_limit, instance_size=instance_size)
        regression_init_k.execute_regression_k_prime(lr=0.00001, n_epochs=21) # setcovering small: lr=0.00002; capa-small: samne; independentset-small: first: lr=0.00002, root: lr=0.00003

regression_init_k = RegressionInitialK_KPrime(instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed=100)
regression_init_k.execute_regression_mergedatasets(lr=lr, n_epochs=301)  # setcovering small: lr=0.00002; capa-small: samne; independentset-small: first: lr=0.00002, root: lr=0.00003
