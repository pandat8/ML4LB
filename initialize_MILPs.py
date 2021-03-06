import ecole
import numpy as np
import pyscipopt
from instance_generation import InstanceGeneration
from utility import instancetypes, instancesizes, incumbent_modes, lbconstraint_modes


for i in range(0, 5):
    for j in range(0, 2):
        instance_type = instancetypes[i]
        instance_size = instancesizes[j]

        if instance_type == instancetypes[0]:
            lbconstraint_mode = 'asymmetric'
        else:
            lbconstraint_mode = 'symmetric'

        print(instance_type + instance_size)
        print(lbconstraint_mode)
        mllb = InstanceGeneration(instance_type, instance_size, lbconstraint_mode, seed=100)

        if i<3:
            mllb.generate_instances(instance_type, instance_size)
        elif i == 3 and j == 0:
            mllb.generate_instances_generalized_independentset()
        elif i == 4 and j == 0:
            mllb.generate_instances_miplib_39binary()
