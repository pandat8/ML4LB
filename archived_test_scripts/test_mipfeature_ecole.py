import ecole
import numpy
import pyscipopt
import pathlib
import gzip
import pickle
from utilities import generator_switcher, instancetypes, lbconstraint_modes
from ecole_extend.environment_extend import SimpleConfiguring


instancetype = instancetypes[2]
mode = modes[4]

directory = './result/generated_instances/'+ instancetype + '/' + mode + '/'
directory_samples = directory + 'samples_kinit/'
pathlib.Path(directory_samples).mkdir(parents=True, exist_ok=True)


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


generator = generator_switcher(instancetype)
seed = 100
generator.seed(seed)
env.seed(seed)  # environment (SCIP)

for i in range(0, 20):
    # if not i == 38:
    instance_name = instancetype + '-' + str(i)
    print(instance_name)
    data = numpy.load(directory + instance_name + '.npz')
    k = data['neigh_sizes']
    t = data['t']
    objs = data['objs']

    # normalize the objective and solving time
    t = t / 3
    objs = (objs - numpy.min(objs))
    objs = objs / numpy.max(objs)

    #compute the performance score
    alpha = 1 / 3
    perf_score = alpha * t + (1 - alpha) * objs
    k_bests = k[numpy.where(perf_score == perf_score.min())]
    k_init = k_bests[0]

    instance = next(generator)

    MIP_model = instance.as_pyscipopt()

    MIP_model.setParam('presolving/maxrounds', 0)
    MIP_model.setParam('presolving/maxrestarts', 0)
    # MIP_model.setParam("limits/nodes", 1)
    MIP_model.setParam('limits/solutions', 1)
    MIP_model.optimize()

    stage = MIP_model.getStage()
    print("* Solve stage: %s" % stage)
    n_sols = MIP_model.getNSols()
    print('* number of solutions : ', n_sols)

    incumbent_solution = MIP_model.getBestSol()


    # MIP_model.freeTransform()
    #
    # stage = MIP_model.getStage()
    # print("* Solve stage: %s" % stage)
    # n_sols = MIP_model.getNSols()
    # print('* number of solutions : ', n_sols)

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
    print(observation)
    action = {'limits/solutions': 1}
    # action = {'limits/nodes': 1}  #
    observation, _, _, done, _ = env.step(action)
    print(observation)

    data_sample = [observation, k_init]
    filename = f'{directory_samples}sample-kinit-{instance_name}.pkl'
    with gzip.open(filename, 'wb') as f:
        pickle.dump(data_sample, f)





# MIP_model = instance.as_pyscipopt()
# MIP_model.setProbName(instancetype + '-' + str(0))
