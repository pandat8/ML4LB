import pyscipopt
from pyscipopt import Model
import ecole
import numpy
import matplotlib.pyplot as plt
from localbranching import addLBConstraint, addLBConstraintAsymmetric
from utility import binary_support


def improvelp(MIP_model, directory, mode, index_instance, t_limit):
    # model_copy = Model('model orig copy', sourceModel=MIP_model, origcopy=True)
    # print("original model:", MIP_model)
    n_vars = MIP_model.getNVars()
    n_binvars = MIP_model.getNBinVars()
    print("N of variables: {}".format(n_vars))
    print("N of binary vars: {}".format(n_binvars))
    print("N of constraints: {}".format(MIP_model.getNConss()))
    # print("copyed model:", model_copy)

    # MIP_model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
    # MIP_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    # MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
    MIP_model.setParam('presolving/maxrounds', 0)
    MIP_model.setParam('presolving/maxrestarts', 0)
    # MIP_model.setParam("limits/nodes", 1)
    MIP_model.setParam('limits/solutions', 1)
    MIP_model.optimize()

    t = MIP_model.getSolvingTime()
    status = MIP_model.getStatus()
    lp_status = MIP_model.getLPSolstat()
    stage = MIP_model.getStage()
    n_sols = MIP_model.getNSols()

    print("* Model status: %s" % status)
    print("* LP status: %s" % lp_status)
    print("* Solve stage: %s" % stage)
    print("* Solving time: %s" % t)
    print('* number of sol : ', n_sols)

    incumbent_solution = MIP_model.getBestSol()
    feasible = MIP_model.checkSol(solution=incumbent_solution)

    if (not status == 'optimal') and feasible:

        # MIP_model.freeTransform()

        initial_obj = MIP_model.getObjVal()
        print("Initial obj before LB: {}".format(initial_obj))
        print('Relative gap: ', MIP_model.getGap())

        n_supportbinvars = binary_support(MIP_model, incumbent_solution)
        print('binary support: ', n_supportbinvars)

        neigh_sizes = []
        objs = []
        t = []
        n_supportbins = []
        MIP_model.resetParams()
        nsample = 101
        for i in range(nsample):

            # create a copy of the MIP to be 'locally branched'
            subMIP_copy, subMIP_copy_vars, success = MIP_model.createCopy(problemName='subMIPmodelCopy', origcopy= False)
            sol_subMIP_copy = subMIP_copy.createSol()

            # create a primal solution for the copy MIP by copying the solution of original MIP
            n_vars = MIP_model.getNVars()
            subMIP_vars = MIP_model.getVars()

            for j in range(n_vars):
                val = MIP_model.getSolVal(incumbent_solution, subMIP_vars[j])
                subMIP_copy.setSolVal(sol_subMIP_copy, subMIP_copy_vars[j], val)
            feasible = subMIP_copy.checkSol(solution=sol_subMIP_copy)

            if feasible:
                # print("the trivial solution of subMIP is feasible ")
                subMIP_copy.addSol(sol_subMIP_copy,False)
                # print("the feasible solution of subMIP_copy is added to subMIP_copy")
            else:
                print("Warn: the trivial solution of subMIP_copy is not feasible!")

            # add LB constraint to subMIP model
            alpha = 0.01 * i
            # if nsample == 41:
            #     if i<11:
            #         alpha = 0.01*i
            #     elif i<31:
            #         alpha = 0.02*(i-5)
            #     else:
            #         alpha = 0.05*(i-20)

            if mode == 'improve-supportbinvars':
                neigh_size = alpha * n_supportbinvars
                subMIP_copy = addLBConstraintAsymmetric(subMIP_copy, sol_subMIP_copy, neigh_size)
            elif mode == 'improve-binvars':
                neigh_size = alpha * n_binvars
                subMIP_copy = addLBConstraint(subMIP_copy, sol_subMIP_copy, neigh_size)

            subMIP_copy.setParam('limits/time', t_limit)
            subMIP_copy.optimize()

            status = subMIP_copy.getStatus()
            best_obj = subMIP_copy.getSolObjVal(subMIP_copy.getBestSol())
            solving_time = subMIP_copy.getSolvingTime() # total time used for solving (including presolving) the current problem

            best_sol = subMIP_copy.getBestSol()

            vars_subMIP = subMIP_copy.getVars()
            n_binvars_subMIP = subMIP_copy.getNBinVars()
            n_supportbins_subMIP = 0
            for i in range(n_binvars_subMIP):
                val = subMIP_copy.getSolVal(best_sol, vars_subMIP[i])
                assert subMIP_copy.isFeasIntegral(val), "Error: Value of a binary varialbe is not integral!"
                if subMIP_copy.isFeasEQ(val, 1.0):
                    n_supportbins_subMIP += 1

            neigh_sizes.append(alpha)
            objs.append(best_obj)
            t.append(solving_time)
            n_supportbins.append(n_supportbins_subMIP)

        for i in range(len(t)):
            print('Neighsize: {:.4f}'.format(neigh_sizes[i]),
                  'Best obj: {:.4f}'.format(objs[i]),
                  'Binary supports:{}'.format(n_supportbins[i]),
                  'Solving time: {:.4f}'.format(t[i]),
                  'Status: {}'.format(status)
                  )

        neigh_sizes = numpy.array(neigh_sizes).reshape(-1).astype('float64')
        t = numpy.array(t).reshape(-1)
        objs = numpy.array(objs).reshape(-1)
        f = directory + MIP_model.getProbName()
        numpy.savez(f, neigh_sizes = neigh_sizes, objs=objs, t=t)
        index_instance += 1

    return index_instance
        # plt.clf()
        # fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4))
        # fig.suptitle("LB to improve")
        # fig.subplots_adjust(top=0.5)
        # ax[0].plot(neigh_sizes, objs)
        # ax[0].set_title(MIP_model.getProbName(), loc='right')
        # ax[0].set_xlabel("Neighborhood Size     in coef. of violation")
        # ax[0].set_ylabel("Objective")
        # ax[1].plot(neigh_sizes, t)
        # ax[1].set_ylim([0,31])
        # ax[1].set_ylabel("Solving time")
        # plt.show()

