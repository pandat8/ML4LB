import pyscipopt
from pyscipopt import Model
import ecole
import numpy
import pathlib
import matplotlib.pyplot as plt
from geco.mips.miplib.base import Loader
from improvelp import improvelp
from utilities import lbconstraint_modes, instancetypes, instancesizes, generator_switcher
# instancetype = instancetypes[2]

lbconstraint_mode = lbconstraint_modes[0]
incumbent_mode = 'firstsol'
t_limit = 3
for t in range(1, 2):
    instancesize = instancesizes[0]
    instancetype = instancetypes[11]
    directory = './result/generated_instances/'+ instancetype + '/' + instancesize + '/'+lbconstraint_mode+'/'
    directory_labelsamples = directory + 'labelsample-' + incumbent_mode  + '/'
    pathlib.Path(directory_labelsamples).mkdir(parents=True, exist_ok=True)
    generator = generator_switcher(instancetype)
    generator.seed(100)
    index_instance = 0

    # while index_instance < 98:
    #     instance = next(generator)
    #     MIP_model = instance.as_pyscipopt()
    #     MIP_model.setProbName(instancetype + '-' + str(index_instance))
    #     print(MIP_model.getProbName())
    #     index_instance += 1

    while index_instance < 100:
        instance = next(generator)
        MIP_model = instance.as_pyscipopt()
        MIP_model.setProbName(instancetype + '-' + str(index_instance))
        print(MIP_model.getProbName())
        index_instance = improvelp(MIP_model, directory_labelsamples, lb_mode, index_instance, t_limit)


