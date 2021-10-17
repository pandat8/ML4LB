import pyscipopt
from pyscipopt import Model
import ecole
import numpy
import matplotlib.pyplot as plt
import pathlib
from localbranching import addLBConstraint
from geco.mips.loading.miplib import Loader
from event import PrimalBoundChangeEventHandler


modes = ['improve-supportbinvars', 'improve-binvars']
mode = modes[1]

directory = './result/miplib2017/' + mode + '/'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

# directory = './result/miplib2017/improve/'
data = numpy.load('./result/miplib2017/miplib2017_binary39.npz')
miplib2017_binary39 = data['miplib2017_binary39']

for p in range(0, len(miplib2017_binary39)):
    instance = Loader().load_instance(miplib2017_binary39[p] + '.mps.gz')
    MIP_model = instance
    print(MIP_model.getProbName())

    primalbound_handler = PrimalBoundChangeEventHandler()
    MIP_model.includeEventhdlr(primalbound_handler, 'primal_bound_update_handler',
                               'store every new primal bound and its time stamp')

    MIP_model.setParam('presolving/maxrounds', 0)
    MIP_model.setParam('presolving/maxrestarts', 0)
    MIP_model.setParam("display/verblevel", 0)
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

    MIP_model.freeProb()
    del MIP_model
