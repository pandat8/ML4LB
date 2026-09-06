from pyscipopt import Model
from localbranching import addLBConstraint
from repairlp import repairlp
import ecole
import numpy
import pathlib
import matplotlib.pyplot as plt
from geco.mips.miplib.base import Loader

# data = numpy.load('./result/miplib2017/miplib2017_binary39.npz')
# miplib2017_binary39 = data['miplib2017_binary39']
#
# for p in range(len(miplib2017_binary39)):
#     instance = Loader().load_instance(miplib2017_binary39[p] + '.mps.gz')
#     MIP_model = instance
#     print(MIP_model.getProbName())
#     repairlp(MIP_model)

# instance = Loader().load_instance('dano3mip.mps.gz')
# MIP_model = instance

instancetypes = ['setcovering', 'capacitedfacility', 'independentset', 'combinatorialauction']
modes = ['repair-slackvars', 'repair-supportbinvars', 'repair-binvars']

instancetype = instancetypes[2]
mode = modes[0]

directory = './result/generated_instances/'+ instancetype +'/'+mode+'/'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


# modes = ['repair-nviolations','repair-nbinvars']
# mode = modes[1]
#
# if mode == 'repair-nbinvars':
#     nsample=41
#     directory = './result/generated_instances/setcovering_asym/repair/timesofbinvars/'
# elif mode == 'repair-nviolations':
#     nsample=21
#     directory = './result/generated_instances/setcovering_asym/repair/timesofviolations/'


"""Test output of set covering instance."""


def generator_switcher(instancetype):
    switcher = {
        instancetypes[0]: lambda : ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05),
        instancetypes[1]: lambda : ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=100),
        instancetypes[2]: lambda : ecole.instance.IndependentSetGenerator(n_nodes=1000),
        instancetypes[3]: lambda : ecole.instance.CombinatorialAuctionGenerator(),
    }
    return switcher.get(instancetype, lambda : "invalide argument")()


generator = generator_switcher(instancetype)
generator.seed(100)

for i in range(0, 100):
    instance = next(generator)
    MIP_model = instance.as_pyscipopt()
    MIP_model.setProbName(instancetype+'-'+str(i))
    print(MIP_model.getProbName())
    repairlp(MIP_model, directory, mode)



