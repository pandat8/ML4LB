import pyscipopt
from pyscipopt import Model
import ecole
import numpy
import pathlib
import matplotlib.pyplot as plt
from geco.mips.miplib.base import Loader
from improvelp import improvelp
from localbranching import LocalBranching
from models import *
from utility import copy_sol, binary_support, modes, instancetypes, generator_switcher

# modes = ['tree-improve-supportbinvars', 'tree-improve-binvars']
# instancetype = instancetypes[2]

env = ecole.environment.Configuring(

    # set up a few SCIP parameters
    scip_params={
        "presolving/maxrounds": 0,  # deactivate presolving
        "presolving/maxrestarts": 0,
    },

    observation_function=ecole.observation.MilpBipartite(),

    reward_function=None,

    # collect additional metrics for information purposes
    information_function={
        'time': ecole.reward.SolvingTime().cumsum(),
    }
)

seed = 100
env.seed(seed)  # environment (SCIP)


mode = modes[4]
instancetype = instancetypes[2]
symmetric = True
time_limit = 60

directory = './result/generated_instances/'+ instancetype +'/'+mode+'/'
directory_lb_test = directory + 'lb-from-firstsol-60s/'
pathlib.Path(directory_lb_test).mkdir(parents=True, exist_ok=True)

generator = generator_switcher(instancetype)
generator.seed(100)

for i in range(100):
    instance = next(generator)
    MIP_model = instance.as_pyscipopt()
    MIP_model.setProbName(instancetype + '-' + str(i))
    instance_name = MIP_model.getProbName()
    print(instance_name)
    if i > 34:

        observation, _, _, done, _ = env.reset(instance)
        print(observation)
        action = {'limits/solutions': 1}
        # action = {'limits/nodes': 1}  #
        sample_observation, _, _, done, _ = env.step(action)
        print(sample_observation)

        graph = BipartiteNodeData(sample_observation.constraint_features, sample_observation.edge_features.indices,
                                  sample_observation.edge_features.values, sample_observation.variable_features)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = sample_observation.constraint_features.shape[0] + sample_observation.variable_features.shape[
            0]

        # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
        # MIP_model = instance


        # MIP_model.optimize()
        # print("Status:", MIP_model.getStatus())
        # print("best obj: ", MIP_model.getObjVal())
        # print("Solving time: ", MIP_model.getSolvingTime())

        MIP_model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
        MIP_model.setParam('limits/solutions', 1)
        MIP_model.setParam("display/verblevel", 0)
        MIP_model.optimize()

        incumbent_solution = MIP_model.getBestSol()
        initial_obj = MIP_model.getSolObjVal(incumbent_solution)

        n_binvars = MIP_model.getNBinVars()
        binary_supports= binary_support(MIP_model, incumbent_solution)

        print("Initial obj before LB: {}".format(initial_obj))
        assert MIP_model.checkSol(solution=incumbent_solution), "The incumbent solution is not feasible!"

        # create a copy of MIP
        MIP_model.resetParams()
        MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(problemName='Copy_' + MIP_model.getProbName(), origcopy=False)
        MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(problemName='Copy2_' + MIP_model.getProbName(),
                                                                      origcopy=False)


        MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent_solution, MIP_copy_vars)
        MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent_solution, MIP_copy_vars2)

        initial_obj = MIP_model_copy.getSolObjVal(sol_MIP_copy)
        print("Initial obj before LB: {}".format(initial_obj))

        lb_model = LocalBranching(MIP_model=MIP_model_copy, MIP_sol_bar=sol_MIP_copy,k=20, total_time_limit=time_limit)
        status, obj_best, elapsed_time = lb_model.search_localbranch(symmeric=symmetric)
        print("Instance:", MIP_model_copy.getProbName())
        print("Status of LB: ", status)
        print("Best obj of LB: ", obj_best)
        print("Solving time: ", elapsed_time)

        model_gnn = GNNPolicy()
        model_gnn.load_state_dict(torch.load('trained_params_' + instancetype + '.pth'))
        k_model = model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr, graph.variable_features)

        k_pred = k_model.item() * n_binvars
        if symmetric == False:
            k_pred = k_model.item() * binary_supports

        lb_model2 = LocalBranching(MIP_model=MIP_model_copy2, MIP_sol_bar=sol_MIP_copy2, k=k_pred, total_time_limit=time_limit)
        status, obj_best, elapsed_time = lb_model2.search_localbranch(symmeric=symmetric)

        print("Instance:", MIP_model_copy2.getProbName())
        print("Status of LB: ", status)
        print("Best obj of LB: ",obj_best)
        print("Solving time: ", elapsed_time)

        lb_bits = lb_model.lb_bits
        times = lb_model.times
        objs = lb_model.objs

        lb_bits_pred = lb_model2.lb_bits
        times_pred = lb_model2.times
        objs_pred = lb_model2.objs

        data = [objs, times, objs_pred, times_pred]

        filename = f'{directory_lb_test}lb-test-{instance_name}.pkl'
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)



        # plt.close('all')
        # plt.clf()
        # fig, ax = plt.subplots(figsize=(8, 6.4))
        # fig.suptitle("Test Result: prediction of initial k")
        # fig.subplots_adjust(top=0.5)
        # ax.set_title(instance_name, loc='right')
        # ax.plot(times, objs, label='lb baseline')
        # ax.plot(times_pred, objs_pred, label='lb with k predicted')
        # ax.set_xlabel('time /s')
        # ax.set_ylabel("objective")
        # ax.legend()
        # plt.show()




