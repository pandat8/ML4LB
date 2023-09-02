from pyscipopt import Model
import pyscipopt
import pathlib
from utilities import instancetypes
from geco.mips.loading.miplib import Loader
# file_directory = './result/miplib2017/miplib2017_purebinary_solved.txt'
# with open(file_directory) as fp:
#     Lines = fp.readlines()
#     i = 0
#     for line in Lines:
#         i += 1
#         instance_str = line.strip()
#         # MIP_model = Loader().load_instance(instance_str)
#         # print(MIP_model.getProbName())
#     print(i)
instance_type = instancetypes[6] # 6
instance_directory = './data/generated_instances/'+ instance_type +'/-small/transformedmodel/'
instance_filename = f'{instance_type}-*_transformed.cip'
sample_files = [str(path) for path in sorted(pathlib.Path(instance_directory).glob(instance_filename), key=lambda path: int(path.stem.replace('-', '_').rsplit("_", 2)[1]))]

print(sample_files)
i = 0

for instance in sample_files:
    print(instance)
    MIP_model = Model()
    MIP_model.readProblem(instance)
    print(MIP_model.getProbName())
    print('Number of variables', MIP_model.getNVars())
    print('Number of binary variables', MIP_model.getNBinVars())

    # directory_instance_name = instance.split('.')[0]
    # filename = f'{directory_instance_name}.mps'
    # print('Write to: '+ filename)
    # MIP_model.writeProblem(filename=filename, trans=False)

#
#     print("Solving first solution ...")
#     MIP_model.setParam('presolving/maxrounds', 0)
#     MIP_model.setParam('presolving/maxrestarts', 0)
#     MIP_model.setParam("display/verblevel", 0)
#     MIP_model.setParam("limits/solutions", 1)
#     MIP_model.optimize()
#
#     status = MIP_model.getStatus()
#     stage = MIP_model.getStage()
#     print("* Solve status: %s" % status)
#     print("* Solve stage: %s" % stage)
#     n_sols = MIP_model.getNSols()
#     print('* number of solutions : ', n_sols)
#     obj = MIP_model.getObjVal()
#     print('* first sol obj : ', obj)
#     print("first solution solving time: ", MIP_model.getSolvingTime())
#
#     MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(problemName='Copy',
#                                                                   origcopy=True)
#     print("Solving root node ...")
#     MIP_model_copy.resetParams()
#     MIP_model_copy.setParam('presolving/maxrounds', 0)
#     MIP_model_copy.setParam('presolving/maxrestarts', 0)
#     MIP_model_copy.setParam("display/verblevel", 0)
#     MIP_model_copy.setParam("limits/nodes", 1)
#     MIP_model_copy.optimize()
#
#     status = MIP_model_copy.getStatus()
#     stage = MIP_model_copy.getStage()
#     print("* Solve status: %s" % status)
#     print("* Solve stage: %s" % stage)
#     n_sols = MIP_model_copy.getNSols()
#     print('* number of solutions : ', n_sols)
#     obj_root = MIP_model_copy.getObjVal()
#     print('* root node obj : ', obj_root)
#     print("root node solving time: ", MIP_model_copy.getSolvingTime())
#     t_firstlp = MIP_model_copy.getFirstLpTime()
#     print("first LP time : ", t_firstlp)
#
#     lp_status = MIP_model_copy.getLPSolstat()
#     print("* LP status: %s" % lp_status)  # 1:optimal
#     if lp_status:
#         print('LP of root node is solved!')
#         lp_obj = MIP_model_copy.getLPObjVal()
#         print("LP objective: ", lp_obj)
#
#     incumbent_solution_first = MIP_model.getBestSol()
#     incumbent_solution_root = MIP_model_copy.getBestSol()
#     first_sol_check = MIP_model.checkSol(solution=incumbent_solution_first)
#
#     if first_sol_check:
#         print('first solution is valid')
#     else:
#         print('Warning: first solution is not valid!')
#     root_sol_check = MIP_model.checkSol(solution=incumbent_solution_root)
#     if root_sol_check:
#         print('root node solution is valid')
#     else:
#         print('Warning: root node solution is not valid!')
#
#     if (not status == 'optimal') and first_sol_check and root_sol_check:
#
#         if i > -1:
#
#             MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(problemName='Copy2',
#                                                                           origcopy=True)
#             print("Solving to optimal ...")
#             MIP_model_copy.resetParams()
#             MIP_model_copy.setParam('presolving/maxrounds', 0)
#             MIP_model_copy.setParam('presolving/maxrestarts', 0)
#             MIP_model_copy.setParam("display/verblevel", 0)
#             MIP_model_copy.setParam('limits/time', 600)
#             MIP_model_copy.optimize()
#             status = MIP_model_copy.getStatus()
#             if status == 'optimal':
#                 print('instance is solved to optimal!')
#                 # objs.append(MIP_model_copy.getObjVal())
#                 # times.append(MIP_model_copy.getSolvingTime())
#             print("instance:", MIP_model_copy.getProbName(),
#                   "status:", MIP_model_copy.getStatus(),
#                   "best obj: ", MIP_model_copy.getObjVal(),
#                   "solving time: ", MIP_model_copy.getSolvingTime())
#         i += 1
#     else:
#         "no solution"
#
#     print("\n")
