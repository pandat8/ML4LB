import ecole
import numpy as np
import pyscipopt
from mllocalbranch_fromfiles import RegressionInitialK
from utility import instancetypes, instancesizes, incumbent_modes, lbconstraint_modes

# instance_type = instancetypes[1]
instance_size = instancesizes[0]
# incumbent_mode = 'firstsol'
lbconstraint_mode = 'symmetric'
samples_time_limit = 3
node_time_limit = 10

total_time_limit = 60
reset_k_at_2nditeration = True


for i in range(0, 1):
    instance_type = instancetypes[i]
    if instance_type == instancetypes[0]:
        lbconstraint_mode = 'asymmetric'
    else:
        lbconstraint_mode = 'symmetric'
    for j in range(1, 2):
        incumbent_mode = incumbent_modes[j]
        for k in range(0, 1):
            test_instance_size = instancesizes[k]

            print(instance_type + test_instance_size)
            print(incumbent_mode)
            print(lbconstraint_mode)

            regression_init_k = RegressionInitialK(instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed=100)

            regression_init_k.generate_k_samples(t_limit=samples_time_limit, instance_size=instance_size)

            # regression_init_k.generate_regression_samples(t_limit=samples_time_limit, instance_size=instance_size)

            # regression_init_k.execute_regression(lr=0.00002, n_epochs=11) # setcovering small: lr=0.00002; capa-small: samne; independentset-small: first: lr=0.00002, root: lr=0.00003

            # regression_init_k.evaluate_localbranching(test_instance_size=test_instance_size,train_instance_size='-small', total_time_limit=total_time_limit, node_time_limit=node_time_limit, reset_k_at_2nditeration=reset_k_at_2nditeration)

            # regression_init_k.solve2opt_evaluation(test_instance_size='-small')

            # regression_init_k.primal_integral(test_instance_size=test_instance_size, total_time_limit=total_time_limit, node_time_limit=node_time_limit)
