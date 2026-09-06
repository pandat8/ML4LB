import pyscipopt
from pyscipopt import Model
import ecole
import numpy
import pathlib
import matplotlib.pyplot as plt
from geco.mips.miplib.base import Loader
from improvelp import improvelp
import gzip
import pickle
from utilities import instancetypes, generator_switcher, instancesizes

# instancetype = instancetypes[2]

instance_type = instancetypes[1]
instance_size = instancesizes[0]
dataset = instance_type + instance_size

directory_opt = './result/generated_instances/' + instance_type + '/' + instance_size + '/'
pathlib.Path(directory_opt).mkdir(parents=True, exist_ok=True)

generator = generator_switcher(dataset)
generator.seed(100)
opt_mode = 'rootsol-valid100-199'

objs = []
times = []
i = 0
while i > 99 and i < 200:

    instance = next(generator)
    MIP_model = instance.as_pyscipopt()
    MIP_model.setProbName(instance_type + '-' + str(i))
    instance_name = MIP_model.getProbName()
    print(instance_name)

    MIP_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
    MIP_model.setParam('presolving/maxrounds', 0)
    MIP_model.setParam('presolving/maxrestarts', 0)
    MIP_model.setParam("display/verblevel", 0)
    MIP_model.setParam("limits/nodes", 1)
    MIP_model.optimize()

    status = MIP_model.getStatus()

    stage = MIP_model.getStage()
    print("* Solve stage: %s" % stage)
    n_sols = MIP_model.getNSols()
    print('* number of solutions : ', n_sols)
    obj_root = MIP_model.getObjVal()
    print('* root obj : ', obj_root)

    incumbent_solution = MIP_model.getBestSol()

    if (not status == 'optimal') and MIP_model.checkSol(solution=incumbent_solution):

        if i > 99:
            MIP_model.resetParams()
            MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(problemName='Copy_' + MIP_model.getProbName(),
                                                                          origcopy=False)
            MIP_model_copy.optimize()
            status = MIP_model_copy.getStatus()
            if status == 'optimal':
                objs.append(MIP_model_copy.getObjVal())
                times.append(MIP_model_copy.getSolvingTime())
            print("instance:", MIP_model_copy.getProbName(),
                  "status:", MIP_model_copy.getStatus(),
                  "best obj: ", MIP_model_copy.getObjVal(),
                  "solving time: ", MIP_model_copy.getSolvingTime())
        i += 1

data = [objs, times]
filename = f'{directory_opt}root-obj-time-' + opt_mode + '.pkl'
with gzip.open(filename, 'wb') as f:
    pickle.dump(data, f)



