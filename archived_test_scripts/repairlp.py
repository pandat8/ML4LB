from pyscipopt import Model
import pyscipopt
from localbranching import addLBConstraint, addLBConstraintAsymmetric, addLBConstraintAsymJustslackvars
import numpy
import matplotlib.pyplot as plt


# def addLBConstraint(mip_model, mip_sol, neighborhoodsize):
#
#     vars = mip_model.getVars()
#     n_binvars = mip_model.getNBinVars()
#
#     lhs = 0
#     rhs = neighborhoodsize
#     cons_vars = numpy.empty(n_binvars, dtype=numpy.object)
#     cons_vals = numpy.empty(n_binvars)
#
#     # compute coefficients for LB constraint
#     for i in range(0, n_binvars):
#         val = mip_model.getSolVal(mip_sol, vars[i])
#         assert mip_model.isFeasIntegral(val), "Error: Solution passed to LB is not integral!"
#
#         if mip_model.isFeasEQ(val, 1.0):
#             cons_vals[i] = -1.0
#             lhs -= 1.0
#             rhs -= 1.0
#         else:
#             cons_vals[i] = 1.0
#         cons_vars[i] = vars[i]
#         assert cons_vars[i].vtype() == "BINARY"
#
#     # create and add LB constraint to mip_model
#     constraint_LB = mip_model.createConsBasicLinear(mip_model.getProbName()+"_localbranching", n_binvars, cons_vars, cons_vals, lhs, rhs)
#     mip_model.addPyCons(constraint_LB)
#     for j in range(0, n_binvars):  # release cons_vars variables after creating a constraint
#         mip_model.releaseVar(cons_vars[j])
#
#     return mip_model


def repairlp(model, directory, mode):
    """
    Repair a the LP solution of a MIP model by rounding, constraint relaxation, local branching
    :param model:
    :return:
    """
    MIP_model = model
    # model_copy = Model('model orig copy', sourceModel=MIP_model, origcopy=True)
    print("original model:", MIP_model)
    print("N of variables: {}".format(MIP_model.getNVars()))
    print("N of constraints: {}".format(MIP_model.getNConss()))
    # print("copyed model:", model_copy)
    MIP_model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
    MIP_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
    MIP_model.setIntParam("lp/solvefreq", 0)
    MIP_model.setParam("limits/nodes", 1)
    MIP_model.setParam("limits/solutions", 1)

    # MIP_model.setParam("limits/solutions", 1)
    MIP_model.optimize()
    #
    status = MIP_model.getStatus()
    lp_status = MIP_model.getLPSolstat()
    stage = MIP_model.getStage()
    n_sols = MIP_model.getNSols()
    t = MIP_model.getSolvingTime()
    print("* Model status: %s" % status)
    print("* Solve stage: %s" % stage)
    print("* LP status: %s" % lp_status)
    print('* number of sol : ', n_sols)

    if status == 'optimal':
        print("Optimal value:", MIP_model.getObjVal())
        # for v in pyscipopt_model.getVars():
        #     if v.name != "n":
        #         print("%s: %d" % (v, pyscipopt_model.getVal(v)))
    else:
        print("Not optimal")
        print("Obj    value:", MIP_model.getObjVal())
        print("LP Obj value:", MIP_model.getLPObjVal())
        lpcands, lpcandssol, lpcadsfrac, nlpcands, npriolpcands, nfracimplvars = MIP_model.getLPBranchCands()
        print("No. of branching cands", nlpcands)
        # print("Branching cands  : ", lpcands)
        # print("Branching values :", lpcandssol)
        # print("fractional part of cands:", lpcadsfrac )

        # for v in pyscipopt_model.getVars():
        #     if v.name != "n":
        #         print("%s: %d" % (v, pyscipopt_model.getVal(v)))

        sol_lp = MIP_model.createLPSol() # get current LP solution
        # print("LP solution:", sol_lp)

        # round the LP solution to nearest integer
        for c in range(0, nlpcands):
            lpcand = lpcands[c]
            lpcand_val = lpcandssol[c]
            rounded_cand_val = MIP_model.floor(lpcand_val + 0.5)
            # rounded_cand_val = MIP_model.floor(lpcand_val + 0.5)
            assert MIP_model.isFeasIntegral(rounded_cand_val), "rounded value is not integral"
            MIP_model.setSolVal(sol_lp, lpcand, rounded_cand_val)

        # created a sub-MIP and copy rounded solution of MIP to subMIP

        subMIP_model = Model('repair subMIP model')
        # subMIP_model.copyParamSettings(MIP_model)
        sol_subMIP = subMIP_model.createSol()

        MIP_vars = MIP_model.getVars()
        n_vars = MIP_model.getNVars()

        subMIP_vars = subMIP_model.getVars()
        n_sub_vars = subMIP_model.getNVars()
        # print("no. of sub_MIP variables: ", n_sub_vars)

        # add variables of original MIP to subMIP and set obj coefficient to 0(by default)
        for i in range(0, n_vars):
            lb = MIP_vars[i].getLbGlobal()
            ub = MIP_vars[i].getUbGlobal()
            var_type = MIP_vars[i].vtype()
            value = MIP_model.getSolVal(sol_lp, MIP_vars[i])
            subMIP_var = subMIP_model.addVar("sub_"+ MIP_vars[i].name, var_type, lb, ub, 0.0)
            subMIP_model.setSolVal(sol_subMIP, subMIP_var, value)



        subMIP_vars = subMIP_model.getVars()
        n_sub_vars = subMIP_model.getNVars()
        print("no. of MIP variables: ", n_vars)
        print("no. of sub_MIP variables: ", n_sub_vars)
        # print("list of sub_MIP variables: ", subMIP_vars)
        # print("list of sub_MIP rounded solution: ", sol_subMIP)

        # for i in range(n_sub_vars):
        #     val = MIP_model.getSolVal(sol_lp, MIP_vars[i])
        #     subMIP_model.setSolVal(sol_subMIP, subMIP_vars[i], val)

        # modify the sub-MIP by relaxing the violated constrants of rounded solution

        # conss = subMIP_model.getConss()
        # n_conss = subMIP_model.getNConss()
        # print("Constraints: ", conss)
        # print("number of constraints:", n_conss)

        rows = MIP_model.getLPRowsData()
        n_rows = MIP_model.getNLPRows()
        print("number of rows:", n_rows)
        slacks = []
        n_violatedcons = 0
        vars_slack = []

        for i in range(0,n_rows):

            constant = rows[i].getConstant()
            lhs = rows[i].getLhs()
            rhs = rows[i].getRhs()
            vals = rows[i].getVals()
            n_nonzeros = rows[i].getNNonz()
            cols = rows[i].getCols()
            rowsol_activity = MIP_model.getRowSolActivity(rows[i], sol_lp)

            cons_vars = numpy.empty(n_nonzeros, dtype = numpy.object)

            # compute the coefficient of slack variables
            if (MIP_model.isFeasLT(rowsol_activity,lhs)):
                slack_coeff = lhs - rowsol_activity
                n_violatedcons += 1
            elif MIP_model.isFeasGT(rowsol_activity, rhs):
                slack_coeff = rhs - rowsol_activity
                n_violatedcons +=1
            else:
                slack_coeff = 0.0
            slacks.append(slack_coeff)

            for j in range(0, n_nonzeros):
                var = cols[j].getVar()
                pos = var.getProbindex()
                cons_vars[j] = subMIP_vars[pos]

            # create the constraint of original MIP for subMIP
            constraint = subMIP_model.createConsBasicLinear(rows[i].getName(), n_nonzeros, cons_vars, vals, lhs, rhs)
            for j in range(0, n_nonzeros): #release cons_vars variables after creating a constraint
                subMIP_model.releaseVar(cons_vars[j])

            # add a slack variable for violated constraints (slack coefficient != 0)
            if not subMIP_model.isFeasEQ(slack_coeff, 0.0):
                var_slack = subMIP_model.addVar(name="artificialslack_"+str(i),vtype='BINARY', lb=0.0, ub=1.0, obj=1.0)
                subMIP_model.setSolVal(sol_subMIP,var_slack, 1.0)
                vars_slack.append(var_slack)

                subMIP_model.addConsCoeff(constraint, var_slack, slack_coeff)
                subMIP_model.releaseVar(var_slack) #release slack variable after calling addConsCoeff

            # add the relaxed constriant to subMIP
            subMIP_model.addPyCons(constraint)


        print("sub-MIP relaxed!")

        n_conss = subMIP_model.getNConss()
        n_vars = subMIP_model.getNVars()
        print("number of  variables after relaxation: ", n_vars)
        print("number of violated constraints: ", n_violatedcons )
        print("repaired-sub-MIP number of constraints:", n_conss)
        # print("Variable set of sub-MIP: ", subMIP_model.getVars())
        # conss = subMIP_model.getConss()
        # print("repaired-sub-MIP Constraints: ", conss)

        feasible = subMIP_model.checkSol(solution=sol_subMIP)
        if feasible:
            # print("the trivial solution of subMIP is feasible ")
            subMIP_model.addSol(sol_subMIP, False)
            print("the feasible solution of subMIP is added to subMIP")
        else:
            print("Error: the trivial solution of subMIP is not feasible!")
        # bestSol = subMIP_model.getBestSol()
        # print("BestSol: ", bestSol)
        # print("BestObj: ", subMIP_model.getSolObjVal(sol=bestSol))

        # collect the index of slack variables in subMIP
        indexlist_varsslack = []
        for i in range(len(vars_slack)):
            pos = vars_slack[i].getProbindex()
            indexlist_varsslack.append(pos)

        n_binvars = subMIP_model.getNBinVars()
        vars = subMIP_model.getVars()

        n_supportbinvars = 0
        for i in range(n_binvars):
            val = subMIP_model.getSolVal(sol_subMIP, vars[i])
            assert subMIP_model.isFeasIntegral(val), "Error: Value of a binary varialbe is not integral!"
            if subMIP_model.isFeasEQ(val, 1.0):
                n_supportbinvars += 1

        neigh_sizes = []
        objs = []
        t = []
        subMIP_model.resetParams()

        nsample = 41
        for i in range(nsample):

            # create a copy of the MIP to be 'locally branched'

            # subMIP_copy = Model('model orig copy', sourceModel=subMIP_model, origcopy=True)
            subMIP_copy, subMIP_copy_vars, success = subMIP_model.createCopy(problemName='subMIPmodelCopy', origcopy=True)
            sol_subMIP_copy = subMIP_copy.createSol()

            # create a primal solution for the copy MIP by copying the solution of original MIP
            n_vars = subMIP_model.getNVars()
            subMIP_vars = subMIP_model.getVars()
            for j in range(n_vars):
                val = subMIP_model.getSolVal(sol_subMIP, subMIP_vars[j])
                subMIP_copy.setSolVal(sol_subMIP_copy, subMIP_copy_vars[j], val)
            feasible = subMIP_copy.checkSol(solution=sol_subMIP_copy)
            # print("Vars: ",subMIP_copy.getVars())
            if feasible:
                # print("the trivial solution of subMIP is feasible ")
                subMIP_copy.addSol(sol_subMIP_copy,False)
                print("the feasible solution of subMIP_copy is added to subMIP_copy")
            else:
                print("Error: the trivial solution of subMIP_copy is not feasible!")

            initial_obj = subMIP_copy.getSolObjVal(sol_subMIP_copy)
            print("Initial obj before LB: {}".format(initial_obj))

            # add LB constraint to subMIP model

            # if nsample==21:
            #     if i<10:
            #         alpha = 0.1 * (i+1)
            #     elif i<20:
            #         alpha = (i + 1 - 10)
            #     else:
            #         alpha = i
            #     neigh_size = alpha * n_violatedcons
            #
            # elif nsample==41:
            #     if i < 11:
            #         alpha = 0.01 * i
            #     elif i < 31:
            #         alpha = 0.02 * (i - 5)
            #     else:
            #         alpha = 0.05 * (i - 20)
            #
            #     neigh_size = alpha * n_binvars

            # sample strategy: 0.01 over [0, 0.1], 0.02 over [0.1, 0.5], 0.05 over [0.5, 1.0]
            if nsample==41:
                if i < 11:
                    alpha = 0.01 * i
                elif i < 31:
                    alpha = 0.02 * (i - 5)
                else:
                    alpha = 0.05 * (i - 20)

                if mode == 'repair-slackvars':
                    neigh_size = alpha * n_violatedcons
                    subMIP_copy = addLBConstraintAsymJustslackvars(subMIP_copy, sol_subMIP_copy, neigh_size, indexlist_varsslack)
                elif mode == 'repair-supportbinvars':
                    neigh_size = alpha * n_supportbinvars #
                    subMIP_copy = addLBConstraintAsymmetric(subMIP_copy, sol_subMIP_copy, neigh_size)
                elif mode == 'repair-binvars':
                    neigh_size = alpha * n_binvars
                    subMIP_copy = addLBConstraint(subMIP_copy, sol_subMIP_copy, neigh_size)

            subMIP_copy.setParam('limits/time', 30)
            subMIP_copy.optimize()

            status = subMIP_copy.getStatus()
            best_obj = subMIP_copy.getSolObjVal(subMIP_copy.getBestSol())
            solving_time = subMIP_copy.getSolvingTime() # total time used for solving (including presolving) the current problem

            print('Status: {}'.format(status),
                  "Best obj: {}".format(best_obj),
                  'Solving time: {}'.format(solving_time)
                      )

            neigh_sizes.append(alpha)
            objs.append(best_obj)
            t.append(solving_time)

        for i in range(len(t)):
            print('Neighsize: {:.4f}'.format(neigh_sizes[i]),
                  "Best obj: {:.4f}".format(objs[i]),
                  'Solving time: {:.4f}'.format(t[i])
                  )

        neigh_sizes = numpy.array(neigh_sizes).reshape(-1).astype('float64')
        # neigh_sizes = numpy.log10(neigh_sizes)
        t = numpy.array(t).reshape(-1)
        objs = numpy.array(objs).reshape(-1)

        numpy.savez(directory + MIP_model.getProbName(), neigh_sizes=neigh_sizes, objs=objs, t=t)

        # plt.clf()
        # fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4))
        # fig.suptitle("LB to repair")
        # fig.subplots_adjust(top=0.5)
        # ax[0].plot(neigh_sizes, objs)
        # ax[0].set_title(MIP_model.getProbName(), loc='right')
        # ax[0].set_xlabel("Neighborhood Size     log(coef. of violation)")
        # ax[0].set_ylabel("Number of violations")
        # ax[1].plot(neigh_sizes, t)
        # ax[1].set_ylim([0, 31])
        # ax[1].set_ylabel("Solving time")
        # plt.show()












