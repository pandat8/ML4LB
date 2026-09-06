import pyscipopt
from pyscipopt import Model
import ecole
import numpy
import pathlib
import matplotlib.pyplot as plt
# from improvelp import improvelp
import gzip
import pickle
from utilities import instancetypes, generator_switcher, instancesizes, binary_support

# instancetype = instancetypes[2]

instance_type = instancetypes[3]
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
while i > -1 and i < 200:
    instance = next(generator)
    MIP_model = instance.as_pyscipopt()
    MIP_model.setProbName(instance_type + '-' + str(i))
    instance_name = MIP_model.getProbName()
    print(instance_name)
    n_vars = MIP_model.getNVars()
    n_binvars = MIP_model.getNBinVars()
    print("N of variables: {}".format(n_vars))
    print("N of binary vars: {}".format(n_binvars))

    # MIP_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    # MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
    print("Solving first solution ...")
    MIP_model.setParam('presolving/maxrounds', 0)
    MIP_model.setParam('presolving/maxrestarts', 0)
    MIP_model.setParam("display/verblevel", 0)
    MIP_model.setParam("limits/solutions", 1)
    MIP_model.optimize()

    status = MIP_model.getStatus()
    stage = MIP_model.getStage()
    print("* Solve status: %s" % status)
    print("* Solve stage: %s" % stage)
    n_sols = MIP_model.getNSols()
    print('* number of solutions : ', n_sols)
    obj = MIP_model.getObjVal()
    print('* first sol obj : ', obj)
    print("first solution solving time: ", MIP_model.getSolvingTime())

    MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(problemName='Copy',
                                                                  origcopy=True)
    print("Solving root node ...")
    MIP_model_copy.resetParams()
    MIP_model_copy.setParam('presolving/maxrounds', 0)
    MIP_model_copy.setParam('presolving/maxrestarts', 0)
    MIP_model_copy.setParam("display/verblevel", 0)

    # MIP_model_copy.setSeparating(pyscipopt.SCIP_PARAMSETTING.FAST)
    # MIP_model_copy.setPresolve(pyscipopt.SCIP_PARAMSETTING.FAST)

    # MIP_model_copy.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    # MIP_model_copy.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)

    MIP_model_copy.setParam("limits/nodes", 1)
    MIP_model_copy.optimize()

    status = MIP_model_copy.getStatus()
    stage = MIP_model_copy.getStage()
    print("* Solve status: %s" % status)
    print("* Solve stage: %s" % stage)
    n_sols = MIP_model_copy.getNSols()
    print('* number of solutions : ', n_sols)
    obj_root = MIP_model_copy.getObjVal()
    print('* root node obj : ', obj_root)
    print("root node solving time: ", MIP_model_copy.getSolvingTime())
    sol_MIP_copy = MIP_model_copy.getBestSol()
    n_supportbinvars = binary_support(MIP_model_copy, sol_MIP_copy)
    print('Binary support: ', n_supportbinvars)

    lp_status = MIP_model_copy.getLPSolstat()
    print("* LP status: %s" % lp_status) # 1:optimal
    if lp_status:
        print('LP of root node is solved!')
        lp_obj = MIP_model_copy.getLPObjVal()
        print("LP objective: ", lp_obj)

    incumbent_solution_first = MIP_model.getBestSol()
    incumbent_solution_root = MIP_model_copy.getBestSol()
    first_sol_check = MIP_model.checkSol(solution=incumbent_solution_first)

    if first_sol_check:
        print('first solution is valid')
    else:
        print('Warning: first solution is not valid!')
    root_sol_check = MIP_model.checkSol(solution=incumbent_solution_root)
    if root_sol_check:
        print('root node solution is valid')
    else:
        print('Warning: root node solution is not valid!')

    if (not status == 'optimal') and first_sol_check and root_sol_check:

        if i > -1:

            MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(problemName='Copy2',
                                                                          origcopy=True)
            print("Solving to optimal ...")
            MIP_model_copy.resetParams()
            MIP_model_copy.setParam('presolving/maxrounds', 0)
            MIP_model_copy.setParam('presolving/maxrestarts', 0)
            MIP_model_copy.setParam("display/verblevel", 0)
            MIP_model_copy.setParam('limits/time', 600)
            MIP_model_copy.optimize()
            status = MIP_model_copy.getStatus()
            if status == 'optimal':
                print('instance is solved to optimal!')
                objs.append(MIP_model_copy.getObjVal())
                times.append(MIP_model_copy.getSolvingTime())
            print("instance:", MIP_model_copy.getProbName(),
                  "status:", MIP_model_copy.getStatus(),
                  "best obj: ", MIP_model_copy.getObjVal(),
                  "solving time: ", MIP_model_copy.getSolvingTime())
        i += 1
    else:
        "no solution"

    print("\n")

# data = [objs, times]
# filename = f'{directory_opt}root-obj-time-' + opt_mode + '.pkl'
# with gzip.open(filename, 'wb') as f:
#     pickle.dump(data, f)



