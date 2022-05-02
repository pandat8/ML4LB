import ecole
import numpy
import pyscipopt
import pathlib
import gzip
import pickle
import matplotlib.pyplot as plt
from utilities import generator_switcher, instancetypes, lbconstraint_modes
from ecole_extend.environment_extend import SimpleConfiguring, SimpleConfiguringEnablecuts


instancetype = instancetypes[11]
mode = lbconstraint_modes[4]
incumbent_mode = 'rootsol'
t_limit = 3

directory = './result/generated_instances/'+ instancetype + '/' + mode + '/'
directory_labelsamples = directory + 'labelsample-' +  incumbent_mode + '/'
directory_samples = directory + 'samples-' + incumbent_mode + '_kinit/'
pathlib.Path(directory_samples).mkdir(parents=True, exist_ok=True)


env = SimpleConfiguring(

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


generator = generator_switcher(instancetype)
seed = 100
generator.seed(seed)
env.seed(seed)  # environment (SCIP)
index_instance = 0

while index_instance < 100:

    instance = next(generator)
    MIP_model = instance.as_pyscipopt()
    instance_name = instancetype + '-' + str(index_instance)
    MIP_model.setProbName(instance_name)
    print(MIP_model.getProbName())

    MIP_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    # MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
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

    incumbent_solution = MIP_model.getBestSol()

    if (not status == 'optimal') and MIP_model.checkSol(solution=incumbent_solution):
        # if not i == 38:
        data = numpy.load(directory_labelsamples + instance_name + '.npz')
        k = data['neigh_sizes']
        t = data['t']
        objs_abs = data['objs']

        # normalize the objective and solving time
        t = t / t_limit
        objs = (objs_abs - numpy.min(objs_abs))
        objs = objs / numpy.max(objs)

        #compute the performance score
        alpha = 1 / 2
        perf_score = alpha * t + (1 - alpha) * objs
        k_bests = k[numpy.where(perf_score == perf_score.min())]
        k_init = k_bests[0]

        # plt.clf()
        # fig, ax = plt.subplots(3, 1, figsize=(6.4, 6.4))
        # fig.suptitle("LB to improve (over all bins)")
        # fig.subplots_adjust(top=0.5)
        # ax[0].plot(k, objs_abs)
        # ax[0].set_title(instance_name, loc='right')
        # ax[0].set_xlabel(r'$\alpha$   ' + '(Neighborhood size: ' + r'$K = \alpha \times N_{binvars}$)')
        # ax[0].set_ylabel("Objective")
        # ax[1].plot(k, t)
        # # ax[1].set_ylim([0,31])
        # ax[1].set_ylabel("Solving time")
        # ax[2].plot(k, perf_score)
        # ax[2].set_ylabel("Performance score")
        # plt.show()

        # MIP_model.freeTransform()
        #
        # stage = MIP_model.getStage()
        # print("* Solve stage: %s" % stage)
        # n_sols = MIP_model.getNSols()
        # print('* number of solutions : ', n_sols)
        #
        # MIP_model._freescip = True
        # instance = ecole.scip.Model.from_pyscipopt(MIP_model)


        # MIP_copy, subMIP_copy_vars, success = MIP_model.createCopy(problemName='subMIPmodelCopy', origcopy=False)
        # sol_subMIP_copy = MIP_copy.createSol()
        #
        # # create a primal solution for the copy MIP by copying the solution of original MIP
        # n_vars = MIP_model.getNVars()
        # subMIP_vars = MIP_model.getVars()
        #
        # for j in range(n_vars):
        #     val = MIP_model.getSolVal(incumbent_solution, subMIP_vars[j])
        #     MIP_copy.setSolVal(sol_subMIP_copy, subMIP_copy_vars[j], val)
        # feasible = MIP_copy.checkSol(solution=sol_subMIP_copy)
        # # print("Vars: ",subMIP_copy.getVars())
        # if feasible:
        #     # print("the trivial solution of subMIP is feasible ")
        #     MIP_copy.addSol(sol_subMIP_copy, False)
        #     print("the feasible solution of subMIP_copy is added to subMIP_copy")
        # else:
        #     print("Error: the trivial solution of subMIP_copy is not feasible!")


        # instance = ecole.scip.Model.from_pyscipopt(MIP_model)
        observation, _, _, done, _= env.reset(instance)
        # print(observation)
        if incumbent_mode == 'rootsol':
            action = {'limits/nodes': 1}
        # action = {'limits/solutions': 1}  #
        observation, _, _, done, _ = env.step(action)
        # print(observation)

        data_sample = [observation, k_init]

        filename = f'{directory_samples}sample-kinit-{instance_name}.pkl'
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data_sample, f)

        index_instance += 1


# MIP_model = instance.as_pyscipopt()
# MIP_model.setProbName(instancetype + '-' + str(0))
