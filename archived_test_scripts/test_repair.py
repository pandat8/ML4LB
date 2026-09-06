from pyscipopt import Model
from localbranching import addLBConstraint
from repairlp import repairlp
import ecole
import numpy
import pathlib
import matplotlib.pyplot as plt
from geco.mips.loading.miplib import Loader

modes = ['repair-slackvars', 'repair-supportbinvars', 'repair-binvars']
mode = modes[1]

directory = './result/miplib2017/'+mode+'/'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

data = numpy.load('./result/miplib2017/miplib2017_binary39.npz')
miplib2017_binary39 = data['miplib2017_binary39']

for p in range(len(miplib2017_binary39)): #
    instance = Loader().load_instance(miplib2017_binary39[p] + '.mps.gz')
    MIP_model = instance
    print(MIP_model.getProbName())
    repairlp(MIP_model, directory, mode)

# instance = Loader().load_instance('dano3mip.mps.gz')
# MIP_model = instance

# """Test output of set covering instance."""
# # genetator_setcover = ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05)
# # genetator_setcover = ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=100)
# # genetator_setcover = ecole.instance.IndependentSetGenerator(n_nodes=1000)
# genetator_setcover = ecole.instance.CombinatorialAuctionGenerator()
# genetator_setcover.seed(100)
# instance = next(genetator_setcover)
#
# print('Set covering: :')
# # print(instance)
# MIP_model = instance.as_pyscipopt()



