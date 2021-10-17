import ecole
import numpy as np
import pyscipopt
from instance_generation import ImitationLocalbranch
from utility import instancetypes, instancesizes, incumbent_modes, lbconstraint_modes

# instance_type = instancetypes[1]
instance_size = instancesizes[0]
# incumbent_mode = 'firstsol'
lbconstraint_mode = 'symmetric'
samples_time_limit = 3

total_time_limit = 30
node_time_limit = 10

reset_k_at_2nditeration = True


for i in range(0, 1):
    instance_type = instancetypes[i]
    if instance_type == instancetypes[0]:
        lbconstraint_mode = 'asymmetric'
    else:
        lbconstraint_mode = 'symmetric'
    for j in range(0, 1):
        incumbent_mode = incumbent_modes[j]

        print(instance_type + instance_size)
        print(incumbent_mode)
        print(lbconstraint_mode)

        rl_localbranch = ImitationLocalbranch(instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed=100)

        # regression_init_k.generate_k_samples(t_limit=samples_time_limit)

        # regression_init_k.generate_regression_samples(t_limit=samples_time_limit)

        # regression_init_k.execute_regression(lr=0.0000001, n_epochs=6) # setcovering small: lr=0.0000001; capa-small: samne; independentset-small: first: lr=0.0000001, root: lr=0.0000005

        # rl_localbranch.execute_rl4localbranch(test_instance_size='-small', total_time_limit=total_time_limit,
        #                                       node_time_limit=node_time_limit, reset_k_at_2nditeration=reset_k_at_2nditeration,
        #                                       lr=0.0001, n_epochs=1)

        # rl_localbranch.execute_imitation(lr=0.001, n_epochs=50)

        rl_localbranch.evaluate_localbranching(test_instance_size='-small', total_time_limit=total_time_limit, node_time_limit=node_time_limit, reset_k_at_2nditeration=reset_k_at_2nditeration)

        # rl_localbranch.primal_integral(test_instance_size=instance_size, total_time_limit=total_time_limit, node_time_limit=node_time_limit)
        # regression_init_k.solve2opt_evaluation(test_instance_size='-small')
