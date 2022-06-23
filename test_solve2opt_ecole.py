import pyscipopt
from pyscipopt import Model
import ecole
import numpy
import pathlib
import matplotlib.pyplot as plt
import gzip
import pickle
from utilities import instancetypes, generator_switcher, instancesizes, incumbent_modes
from event import PrimalBoundChangeEventHandler
from geco.mips.loading.miplib import Loader

# def generator_switcher(instancetype):
#     switcher = {
#         instancetypes[0]: lambda : ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05),
#         instancetypes[1]: lambda : ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=100),
#         instancetypes[2]: lambda : ecole.instance.IndependentSetGenerator(n_nodes=1000),
#         instancetypes[3]: lambda : ecole.instance.CombinatorialAuctionGenerator(n_items=300, n_bids=300),
#         instancetypes[4]: lambda: ecole.instance.SetCoverGenerator(n_rows=1000, n_cols=2000, density=0.05),
#         instancetypes[5]: lambda : ecole.instance.SetCoverGenerator(n_rows=2000, n_cols=4000, density=0.05),
#         instancetypes[6]: lambda: ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=200, n_facilities=200),
#         instancetypes[7]: lambda: ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=400, n_facilities=400),
#         instancetypes[8]: lambda: ecole.instance.IndependentSetGenerator(n_nodes=2000),
#         instancetypes[9]: lambda: ecole.instance.IndependentSetGenerator(n_nodes=4000),
#     }
#     return switcher.get(instancetype, lambda : "invalide argument")()

# instancetypes = ['setcovering', 'capacitedfacility', 'independentset', 'combinatorialauction','setcovering-row1000col2000', 'setcovering-row2000col4000', 'capacitedfacility-c200-f200', 'capacitedfacility-c400-f400', 'independentset-n2000', 'independentset-n4000']
# modes = ['improve-supportbinvars', 'improve-binvars']
# instancetype = instancetypes[2]


instance_size = instancesizes[0]
test_instance_size = instancesizes[0]
incumbent_mode = incumbent_modes[0]


for t in range(6, 7):
    instance_type = instancetypes[t]

    direc = './data/generated_instances/' + instance_type + '/' + test_instance_size + '/'
    directory_transformedmodel = direc + 'transformedmodel' + '/'
    directory_sol = direc + incumbent_mode + '/'

    # generator = generator_switcher(dataset)
    # generator.seed(100)

    for i in range(33, 34):
        filename = f'{directory_transformedmodel}{instance_type}-{str(i)}_transformed.cip'
        firstsol_filename = f'{directory_sol}{incumbent_mode}-{instance_type}-{str(i)}_transformed.sol'

        MIP_model = Model()
        print(filename)
        MIP_model.readProblem(filename)
        instance_name = MIP_model.getProbName()
        print(instance_name)
        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

        incumbent = MIP_model.readSolFile(firstsol_filename)

        feas = MIP_model.checkSol(incumbent)
        try:
            MIP_model.addSol(incumbent, False)
        except:
            print('Error: the root solution of ' + instance_name + ' is not feasible!')
        # if 13 < i:

        primalbound_handler = PrimalBoundChangeEventHandler()
        MIP_model.includeEventhdlr(primalbound_handler, 'primal_bound_update_handler', 'store every new primal bound and its time stamp')

        MIP_model.setParam('presolving/maxrounds', 0)
        MIP_model.setParam('presolving/maxrestarts', 0)
        MIP_model.setParam("display/verblevel", 0)
        MIP_model.setParam('limits/time', 10)
        MIP_model.optimize()
        status = MIP_model.getStatus()
        best_obj = MIP_model.getObjVal()
        if status == 'optimal':
            obj = MIP_model.getObjVal()
            time = MIP_model.getSolvingTime()
            data = [obj, time]

            # filename = f'{directory_opt}{instance_name}-optimal-obj-time.pkl'
            # with gzip.open(filename, 'wb') as f:
            #     pickle.dump(data, f)

        print("instance:", MIP_model.getProbName(),
              "status:", MIP_model.getStatus(),
              "best obj: ", MIP_model.getObjVal(),
              "solving time: ", MIP_model.getSolvingTime())
        print('primal bounds: ')
        print(primalbound_handler.primal_bounds)
        print('times: ')
        print(primalbound_handler.primal_times)

        MIP_model.freeTransform()
        print('primal bounds: ')
        print(primalbound_handler.primal_bounds)
        print('times: ')
        print(primalbound_handler.primal_times)

        primalbound_handler.primal_bounds = []
        primalbound_handler.primal_times = []

        print('primal bounds: ')
        print(primalbound_handler.primal_bounds)
        print('times: ')
        print(primalbound_handler.primal_times)

        if best_obj >= 0:
            MIP_model.setObjlimit(0.999 * best_obj)
        else:
            MIP_model.setObjlimit(1.001 * best_obj)

        MIP_model.setParam('presolving/maxrounds', 0)
        MIP_model.setParam('presolving/maxrestarts', 0)
        MIP_model.setParam("display/verblevel", 0)
        MIP_model.setParam('limits/time', 20)
        MIP_model.optimize()
        status = MIP_model.getStatus()
        if status == 'optimal':
            obj = MIP_model.getObjVal()
            time = MIP_model.getSolvingTime()
            data = [obj, time]

            # filename = f'{directory_opt}{instance_name}-optimal-obj-time.pkl'
            # with gzip.open(filename, 'wb') as f:
            #     pickle.dump(data, f)

        print("instance:", MIP_model.getProbName(),
              "status:", MIP_model.getStatus(),
              "best obj: ", MIP_model.getObjVal(),
              "solving time: ", MIP_model.getSolvingTime())
        print('primal bounds: ')
        print(primalbound_handler.primal_bounds)
        print('times: ')
        print(primalbound_handler.primal_times)


        del MIP_model







