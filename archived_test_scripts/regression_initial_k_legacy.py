"""Archived: the absolute-k variant of the regression pipeline (legacy).

This module contains the original RegressionInitialK class, which collects
data for, trains and evaluates a regression model predicting the initial
neighborhood size k of local branching directly (in absolute terms). It was
superseded by RegressionInitialK_KPrime in src/ml4lb/localbranching_ml.py,
which predicts k as a fraction of k' and produced the results of the paper.

Kept for reference only; it is used solely by the retired scripts in this
folder (test_regression_kinit.py, test_lp.py). Running it requires the src/
directory on sys.path (it imports the ml4lb package).
"""

import pyscipopt
from pyscipopt import Model
import ecole
import numpy as np
import random
import pathlib
import gzip
import pickle
import json
import matplotlib.pyplot as plt
from geco.mips.loading.miplib import Loader
from ml4lb.utilities import lbconstraint_modes, instancetypes, incumbent_modes, instancesizes, t_reward_types, generator_switcher, binary_support, copy_sol, mean_filter,mean_forward_filter, imitation_accuracy, haming_distance_solutions, haming_distance_solutions_asym, copy_sol, mean_shift, k_0_bank
from ml4lb.localbranching import addLBConstraint, addLBConstraintAsymmetric
from ml4lb.ecole_extend.environment_extend import SimpleConfiguring, SimpleConfiguringEnablecuts, SimpleConfiguringEnableheuristics
from ml4lb.models import GraphDataset, GNNPolicy, BipartiteNodeData
import torch.nn.functional as F
import torch_geometric
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from scipy.interpolate import interp1d
from ml4lb.localbranching import LocalBranching
from ml4lb.event import StopWhenFirstLPSolvedEventHandler

import gc
import sys

from ml4lb.models_rl import SimplePolicy, ImitationLbDataset, AgentReinforce
from ml4lb.dataset import InstanceDataset, custom_collate, InstanceDataset_2


class RegressionInitialK:
    """Data collection, training and evaluation for the regression model that
    predicts the initial neighborhood size k directly (absolute-k variant;
    kept for reference, the paper results use RegressionInitialK_KPrime).
    """

    def __init__(self, instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed=100):
        self.instance_type = instance_type
        self.instance_size = instance_size
        self.incumbent_mode = incumbent_mode
        self.lbconstraint_mode = lbconstraint_mode
        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
        self.seed = seed
        self.directory = './result/generated_instances/' + self.instance_type + '/' + self.instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'

        self.initialize_ecole_env()

        self.env.seed(self.seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)


    def initialize_ecole_env(self):

        if self.incumbent_mode == 'firstsol':

            self.env = ecole.environment.Configuring(

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

        elif self.incumbent_mode == 'rootsol':

            if self.instance_type == 'independentset':
                self.env = SimpleConfiguring(

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
            else:
                self.env = SimpleConfiguringEnablecuts(

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
            # elif self.instance_type == 'capacitedfacility':

    def set_and_optimize_MIP(self, MIP_model, incumbent_mode):

        preprocess_off = True
        if incumbent_mode == 'firstsol':
            heuristics_off = False
            cuts_off = False
        elif incumbent_mode == 'rootsol':
            if self.instance_type == 'independentset':
                heuristics_off = True
                cuts_off = True
            else:
                heuristics_off = True
                cuts_off = False
            # elif self.instance_type == 'capacitedfacility':

        if preprocess_off:
            MIP_model.setParam('presolving/maxrounds', 0)
            MIP_model.setParam('presolving/maxrestarts', 0)

        if heuristics_off:
            MIP_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)

        if cuts_off:
            MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)

        if incumbent_mode == 'firstsol':
            MIP_model.setParam('limits/solutions', 1)
        elif incumbent_mode == 'rootsol':
            MIP_model.setParam("limits/nodes", 1)

        MIP_model.optimize()

        t = MIP_model.getSolvingTime()
        status = MIP_model.getStatus()
        lp_status = MIP_model.getLPSolstat()
        stage = MIP_model.getStage()
        n_sols = MIP_model.getNSols()


        incumbent_solution = MIP_model.getBestSol()
        feasible = MIP_model.checkSol(solution=incumbent_solution)

        return status, feasible, MIP_model, incumbent_solution

    def initialize_MIP(self, MIP_model):

        MIP_model_2, MIP_2_vars, success = MIP_model.createCopy(
            problemName='Baseline', origcopy=False)

        incumbent_mode = self.incumbent_mode
        if self.incumbent_mode == 'firstsol':
            incumbent_mode_2 = 'rootsol'
        elif self.incumbent_mode == 'rootsol':
            incumbent_mode_2 = 'firstsol'

        status, feasible, MIP_model, incumbent_solution = self.set_and_optimize_MIP(MIP_model, incumbent_mode)
        status_2, feasible_2, MIP_model_2, incumbent_solution_2 = self.set_and_optimize_MIP(MIP_model_2, incumbent_mode_2)

        feasible = feasible and feasible_2

        if (not status == 'optimal') and (not status_2 == 'optimal'):
            not_optimal = True
        else:
            not_optimal = False

        if not_optimal and feasible:
            valid = True
        else:
            valid = False

        return valid, MIP_model, incumbent_solution

    def solve_lp(self, MIP_model, lp_algo='s'):

        # solve the LP relaxation of root node
        status = MIP_model.getStatus()
        print("* Model status: %s" % status)
        MIP_model.resetParams()
        MIP_model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
        MIP_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
        MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
        MIP_model.setIntParam("lp/solvefreq", 0)
        MIP_model.setParam("limits/nodes", 1)
        MIP_model.setParam("display/verblevel", 0)
        MIP_model.setParam("lp/disablecutoff", 1)

        MIP_model.setParam("lp/initalgorithm", lp_algo)
        MIP_model.setParam("lp/resolvealgorithm", lp_algo)

        MIP_model.optimize()
        #
        status = MIP_model.getStatus()
        lp_status = MIP_model.getLPSolstat()
        stage = MIP_model.getStage()
        n_sols = MIP_model.getNSols()
        print("* Model status: %s" % status)
        print("* Solve stage: %s" % stage)
        print("* LP status: %s" % lp_status)
        print('* number of sol : ', n_sols)

        return MIP_model, lp_status

    def compute_k_prime(self, MIP_model, incumbent):

        # solve the root node and get the LP solution

        # solve the LP relaxation of root node
        MIP_model, lp_status = self.solve_lp(MIP_model)
        if lp_status == 1:
            sol_lp = MIP_model.createLPSol()

            k_prime = haming_distance_solutions(MIP_model, incumbent, sol_lp)
            if not self.is_symmetric:
                k_prime = haming_distance_solutions_asym(MIP_model, incumbent, sol_lp)
            k_prime = np.ceil(k_prime)
            valid = True
        else:
            print("Warning: k_prime is not valid! Since LP is not solved to optimal! LP solution status is {}".format(str(lp_status)))
            k_prime = 0
            valid = False

        return k_prime, MIP_model, valid

    def sample_k_per_instance(self, t_limit, index_instance):

        filename = f'{self.directory_transformedmodel}{self.instance_type}-{str(index_instance)}_transformed.cip'
        firstsol_filename = f'{self.directory_sol}{self.incumbent_mode}-{self.instance_type}-{str(index_instance)}_transformed.sol'

        MIP_model = Model()
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

        initial_obj = MIP_model.getSolObjVal(incumbent)
        print("Initial obj before LB: {}".format(initial_obj))

        n_supportbinvars = binary_support(MIP_model, incumbent)
        print('binary support: ', n_supportbinvars)

        MIP_model.resetParams()

        neigh_sizes = []
        objs = []
        t = []
        n_supportbins = []
        statuss = []
        relax_grips = []
        n_nodes = []
        firstlp_times = []
        n_lps = []
        presolve_times = []

        firstlp_times_test = []
        solving_times_test = []


        nsample = 11
        # create a copy of the MIP to be 'locally branched'
        MIP_copy, subMIP_model_vars, success = MIP_model.createCopy(problemName='MIPCopy',
                                                                      origcopy=False)
        MIP_copy.resetParams()
        sol_MIP_copy = MIP_copy.createSol()

        # create a primal solution for the copy MIP by copying the solution of original MIP
        n_vars = MIP_model.getNVars()
        subMIP_vars = MIP_model.getVars()

        for j in range(n_vars):
            val = MIP_model.getSolVal(incumbent, subMIP_vars[j])
            MIP_copy.setSolVal(sol_MIP_copy, subMIP_model_vars[j], val)
        feasible = MIP_copy.checkSol(solution=sol_MIP_copy)

        if feasible:
            MIP_copy.addSol(sol_MIP_copy, False)
        else:
            print("Warn: the trivial solution of subMIP_model is not feasible!")

        n_supportbinvars = binary_support(MIP_copy, sol_MIP_copy)
        print('binary support: ', n_supportbinvars)

        k_base = n_binvars
        if self.is_symmetric == False:
            k_base = n_supportbinvars
        # solve the root node and get the LP solution
        k_prime, _, valid = self.compute_k_prime(MIP_model, incumbent)
        if valid is True:
            phi_prime = k_prime / k_base
            n_bins = MIP_model.getNBinVars()
            lpcands, lpcandssol, lpcadsfrac, nlpcands, npriolpcands, nfracimplvars = MIP_model.getLPBranchCands()
            relax_grip = 1 - nlpcands / n_bins
            print('relaxation grip of original problem :', relax_grip)
        else:
            phi_prime = 1

        print('phi_prime :', phi_prime)


        for i in range(nsample):

            # create a copy of the MIP to be 'locally branched', initialize it by 1. solving the LP 2. adding the incumbent
            subMIP_model, subMIP_model_vars, success = MIP_copy.createCopy(problemName='MIPCopy',
                                                                         origcopy=False)
            # solve LP relaxation of root node of original problem
            subMIP_model, lp_status = self.solve_lp(subMIP_model)

            # create a primal solution for the copy MIP by copying the solution of original MIP
            sol_subMIP_model = subMIP_model.createSol()
            n_vars = MIP_copy.getNVars()
            MIP_copy_vars = MIP_copy.getVars()

            for j in range(n_vars):
                val = MIP_copy.getSolVal(sol_MIP_copy, MIP_copy_vars[j])
                subMIP_model.setSolVal(sol_subMIP_model, subMIP_model_vars[j], val)
            feasible = subMIP_model.checkSol(solution=sol_subMIP_model)

            if feasible:
                subMIP_model.addSol(sol_subMIP_model, False)
            else:
                print("Warning: the trivial solution of subMIP_model is not feasible!")


            # add LB constraint to subMIP model
            alpha = 0.1 * (i)

            neigh_size = np.ceil(alpha * k_prime)
            if self.lbconstraint_mode == 'asymmetric':
                subMIP_model, constraint_lb = addLBConstraintAsymmetric(subMIP_model, sol_subMIP_model, neigh_size)
            else:
                subMIP_model, constraint_lb = addLBConstraint(subMIP_model, sol_subMIP_model, neigh_size)

            print('Neigh size:', alpha)
            stage = subMIP_model.getStage()
            print("* Solve stage: %s" % stage)

            subMIP_model2 = subMIP_model

            subMIP_model2, lp_status = self.solve_lp(subMIP_model2, lp_algo='d')
            relax_grip = 2

            n_bins = subMIP_model2.getNBinVars()
            lpcands, lpcandssol, lpcadsfrac, nlpcands, npriolpcands, nfracimplvars = subMIP_model2.getLPBranchCands()
            relax_grip = 1 - nlpcands / n_bins
            print('relaxation grip of subMIP :', relax_grip)

            firstlp_time_test = subMIP_model2.getFirstLpTime()  # time for solving first LP rexlaxation at the root node
            solving_time_test = subMIP_model2.getSolvingTime()  # total time used for solving (including presolving) the current problem
            print('firstLP time for subMIP test :', firstlp_time_test)
            print('root node time for LP subMIP test :', solving_time_test)

            subMIP_model.resetParams()
            subMIP_model.setParam('limits/time', t_limit)
            subMIP_model.setParam("display/verblevel", 0)
            subMIP_model.optimize()

            status_subMIP = subMIP_model.getStatus()
            best_obj = subMIP_model.getSolObjVal(subMIP_model.getBestSol())
            solving_time = subMIP_model.getSolvingTime()  # total time used for solving (including presolving) the current problem
            n_node = subMIP_model.getNTotalNodes()
            firstlp_time = subMIP_model.getFirstLpTime() # time for solving first LP rexlaxation at the root node
            presolve_time = subMIP_model.getPresolvingTime()
            n_lp = subMIP_model.getNLPs()

            best_sol = subMIP_model.getBestSol()

            vars_subMIP = subMIP_model.getVars()
            n_binvars_subMIP = subMIP_model.getNBinVars()
            n_supportbins_subMIP = 0
            for i in range(n_binvars_subMIP):
                val = subMIP_model.getSolVal(best_sol, vars_subMIP[i])
                assert subMIP_model.isFeasIntegral(val), "Error: Value of a binary varialbe is not integral!"
                if subMIP_model.isFeasEQ(val, 1.0):
                    n_supportbins_subMIP += 1


            neigh_sizes.append(alpha)
            objs.append(best_obj)
            t.append(solving_time)
            n_supportbins.append(n_supportbins_subMIP)
            statuss.append(status_subMIP)

            relax_grips.append(relax_grip)
            n_nodes.append(n_node)
            firstlp_times.append(firstlp_time)
            presolve_times.append(presolve_time)
            n_lps.append(n_lp)
            firstlp_times_test.append(firstlp_time_test)
            solving_times_test.append(solving_time_test)

            subMIP_model.freeTransform()
            subMIP_model.resetParams()
            subMIP_model.delCons(constraint_lb)
            subMIP_model.releasePyCons(constraint_lb)
            subMIP_model.freeProb()
            del subMIP_model
            del constraint_lb

        for i in range(len(t)):
            print('Neighsize: {:.4f}'.format(neigh_sizes[i]),
                  'Best obj: {:.4f}'.format(objs[i]),
                  'Binary supports:{}'.format(n_supportbins[i]),
                  'Solving time: {:.4f}'.format(t[i]),
                  'Presolve_time: {:.4f}'.format(presolve_times[i]),
                  'FirstLP time: {:.4f}'.format(firstlp_times[i]),
                  'solved LPs: {:.4f}'.format(n_lps[i]),
                  'B&B nodes: {:.4f}'.format(n_nodes[i]),
                  'Relaxation grip: {:.4f}'.format(relax_grips[i]),
                  'Solving time of LP root : {:.4f}'.format(t[i]),
                  'FirstLP time of LP root: {:.4f}'.format(firstlp_times[i]),
                  'Status: {}'.format(statuss[i])
                  )

        neigh_sizes = np.array(neigh_sizes).reshape(-1)
        t = np.array(t).reshape(-1)
        objs = np.array(objs).reshape(-1)
        relax_grips = np.array(relax_grips).reshape(-1)

        # normalize the objective and solving time
        t = t / t_limit
        objs_abs = objs
        objs = (objs_abs - np.min(objs_abs))
        objs = objs / np.max(objs)

        t = mean_filter(t, 5)
        objs = mean_filter(objs, 5)


        # compute the performance score
        alpha = 1 / 2
        perf_score = alpha * t + (1 - alpha) * objs
        k_bests = neigh_sizes[np.where(perf_score == perf_score.min())]
        k_init = k_bests[0]
        print('k_0_star:', k_init)


        plt.clf()
        fig, ax = plt.subplots(4, 1, figsize=(6.4, 6.4))
        fig.suptitle("Evaluation of size of lb neighborhood")
        fig.subplots_adjust(top=0.5)
        ax[0].plot(neigh_sizes, objs)
        ax[0].set_title(instance_name, loc='right')
        ax[0].set_xlabel(r'$\ r $   ' + '(Neighborhood size: ' + r'$K = r \times N$)')
        ax[0].set_ylabel("Objective")
        ax[1].plot(neigh_sizes, t)
        ax[1].set_ylabel("Solving time")
        ax[2].plot(neigh_sizes, perf_score)
        ax[2].set_ylabel("Performance score")
        ax[3].plot(neigh_sizes, relax_grips)
        ax[3].set_ylabel("Relaxation grip")
        plt.show()


        index_instance += 1

        return index_instance

    def generate_k_samples(self, t_limit, instance_size='-small'):
        """
        For each MIP instance, sample k from [0,1] * n_binary(symmetric) or [0,1] * n_binary_support(asymmetric),
        and evaluate the performance of 1st round of local-branching
        :param t_limit:
        :param k_samples_directory:
        :return:
        """

        self.k_samples_directory = self.directory + 'k_samples' + '/'
        pathlib.Path(self.k_samples_directory).mkdir(parents=True, exist_ok=True)

        direc = './data/generated_instances/' + self.instance_type + '/' + instance_size + '/'
        self.directory_transformedmodel = direc + 'transformedmodel' + '/'
        self.directory_sol = direc + self.incumbent_mode + '/'

        index_instance = 0


        while index_instance < 100:
            index_instance = self.sample_k_per_instance(t_limit, index_instance)

    def generate_regression_samples(self, t_limit, instance_size='-small'):

        self.k_samples_directory = self.directory + 'k_samples' + '/'
        self.regression_samples_directory = self.directory + 'regression_samples' + '/'
        pathlib.Path(self.regression_samples_directory).mkdir(parents=True, exist_ok=True)

        direc = './data/generated_instances/' + self.instance_type + '/' + instance_size + '/'
        self.directory_transformedmodel = direc + 'transformedmodel' + '/'
        self.directory_sol = direc + self.incumbent_mode + '/'

        index_instance = 0
        while index_instance < 100:

            filename = f'{self.directory_transformedmodel}{self.instance_type}-{str(index_instance)}_transformed.cip'
            firstsol_filename = f'{self.directory_sol}{self.incumbent_mode}-{self.instance_type}-{str(index_instance)}_transformed.sol'

            MIP_model = Model()
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

            instance = ecole.scip.Model.from_pyscipopt(MIP_model)

            instance_name = self.instance_type + '-' + str(index_instance)
            data = np.load(self.k_samples_directory + instance_name + '.npz')
            k = data['neigh_sizes']
            t = data['t']
            objs_abs = data['objs']

            # normalize the objective and solving time
            t = t / t_limit
            objs = (objs_abs - np.min(objs_abs))
            objs = objs / np.max(objs)

            t = mean_filter(t, 5)
            objs = mean_filter(objs, 5)


            # compute the performance score
            alpha = 1 / 2
            perf_score = alpha * t + (1 - alpha) * objs
            k_bests = k[np.where(perf_score == perf_score.min())]
            k_init = k_bests[0]


            observation, _, _, done, _ = self.env.reset(instance)

            data_sample = [observation, k_init]
            filename = f'{self.regression_samples_directory}regression-{instance_name}.pkl'
            with gzip.open(filename, 'wb') as f:
                pickle.dump(data_sample, f)

            index_instance += 1

    def test_lp(self, t_limit, instance_size='-small'):

        self.k_samples_directory = self.directory + 'k_samples' + '/'
        self.regression_samples_directory = self.directory + 'regression_samples' + '/'
        pathlib.Path(self.regression_samples_directory).mkdir(parents=True, exist_ok=True)

        direc = './data/generated_instances/' + self.instance_type + '/' + instance_size + '/'
        self.directory_transformedmodel = direc + 'transformedmodel' + '/'
        self.directory_sol = direc + self.incumbent_mode + '/'

        index_instance = 0
        list_phi_prime = []
        list_phi_lp2relax = []
        list_phi_star = []
        count_phi_star_smaller = 0
        count_phi_lp_relax_diff = 0
        while index_instance < 100:

            filename = f'{self.directory_transformedmodel}{self.instance_type}-{str(index_instance)}_transformed.cip'
            firstsol_filename = f'{self.directory_sol}{self.incumbent_mode}-{self.instance_type}-{str(index_instance)}_transformed.sol'

            MIP_model = Model()
            MIP_model.readProblem(filename)
            instance_name = MIP_model.getProbName()
            n_vars = MIP_model.getNVars()
            n_binvars = MIP_model.getNBinVars()
            print("N of binary vars: {}".format(n_binvars))

            incumbent = MIP_model.readSolFile(firstsol_filename)

            feas = MIP_model.checkSol(incumbent)
            try:
                MIP_model.addSol(incumbent, False)
            except:
                print('Error: the root solution of ' + instance_name + ' is not feasible!')

            instance = ecole.scip.Model.from_pyscipopt(MIP_model)

            instance_name = self.instance_type + '-' + str(index_instance)
            data = np.load(self.k_samples_directory + instance_name + '.npz')
            k = data['neigh_sizes']
            t = data['t']
            objs_abs = data['objs']

            # normalize the objective and solving time
            t = t / t_limit
            objs = (objs_abs - np.min(objs_abs))
            objs = objs / np.max(objs)

            t = mean_filter(t, 5)
            objs = mean_filter(objs, 5)


            # compute the performance score
            alpha = 1 / 2
            perf_score = alpha * t + (1 - alpha) * objs
            k_bests = k[np.where(perf_score == perf_score.min())]
            k_init = k_bests[0]

            # solve the root node and get the LP solution
            MIP_model.freeTransform()
            status = MIP_model.getStatus()
            print("* Model status: %s" % status)
            MIP_model.resetParams()
            MIP_model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
            MIP_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
            MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
            MIP_model.setIntParam("lp/solvefreq", 0)
            MIP_model.setParam("limits/nodes", 1)
            MIP_model.setParam("display/verblevel", 0)
            MIP_model.setParam("lp/disablecutoff", 1)

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

            sol_lp = MIP_model.createLPSol()

            k_prime = haming_distance_solutions(MIP_model, incumbent, sol_lp)

            n_bins = MIP_model.getNBinVars()
            k_base = n_bins

            # compute relaxation grip
            lpcands, lpcandssol, lpcadsfrac, nlpcands, npriolpcands, nfracimplvars = MIP_model.getLPBranchCands()
            print('binvars :', n_bins)
            print('nbranchingcands :', nlpcands)
            print('nfracimplintvars :', nfracimplvars)
            print('relaxation grip :', 1 - nlpcands / n_bins )

            if self.is_symmetric == False:
                k_prime = haming_distance_solutions_asym(MIP_model, incumbent, sol_lp)
                binary_supports = binary_support(MIP_model, incumbent)
                k_base = binary_supports

            phi_prime = k_prime / k_base

            phi_star = k_init
            list_phi_prime.append(phi_prime)
            list_phi_star.append(phi_star)

            if phi_star <= phi_prime:
                count_phi_star_smaller += 1
            else:
                list_phi_prime_invalid = phi_prime
                list_phi_star_invalid = list_phi_star

            print('instance : ', MIP_model.getProbName())
            print('phi_prime = ', phi_prime)
            print('phi_star = ', phi_star)
            print('valid count: ', count_phi_star_smaller)


            index_instance += 1

        arr_phi_prime = np.array(list_phi_prime).reshape(-1)
        arr_phi_star = np.array(list_phi_star).reshape(-1)
        ave_phi_prime = arr_phi_prime.sum() / len(arr_phi_prime)
        ave_phi_star = arr_phi_star.sum() / len(arr_phi_star)

        print(self.instance_type + self.instance_size)
        print(self.incumbent_mode + 'Solution')
        print('number of valid phi data points: ', count_phi_star_smaller)
        print('average phi_star :', ave_phi_star )
        print('average phi_prime: ', ave_phi_prime)

    def load_dataset(self, dataset_directory=None):

        self.regression_samples_directory = dataset_directory
        filename = 'regression-' + self.instance_type + '-*.pkl'
        sample_files = [str(path) for path in pathlib.Path(self.regression_samples_directory).glob(filename)]
        train_files = sample_files[:int(0.7 * len(sample_files))]
        valid_files = sample_files[int(0.7 * len(sample_files)):int(0.8 * len(sample_files))]
        test_files =  sample_files[int(0.8 * len(sample_files)):]

        train_data = GraphDataset(train_files)
        train_loader = torch_geometric.data.DataLoader(train_data, batch_size=1, shuffle=True)
        valid_data = GraphDataset(valid_files)
        valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=1, shuffle=False)
        test_data = GraphDataset(test_files)
        test_loader = torch_geometric.data.DataLoader(test_data, batch_size=1, shuffle=False)

        return train_loader, valid_loader, test_loader

    def train(self, gnn_model, data_loader, optimizer=None):
        """
        training function
        :param gnn_model:
        :param data_loader:
        :param optimizer:
        :return:
        """
        mean_loss = 0
        n_samples_precessed = 0
        with torch.set_grad_enabled(optimizer is not None):
            for batch in data_loader:
                k_model = gnn_model(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
                k_init = batch.k_init
                loss = F.l1_loss(k_model.float(), k_init.float())
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                mean_loss += loss.item() * batch.num_graphs
                n_samples_precessed += batch.num_graphs
        mean_loss /= n_samples_precessed

        return mean_loss

    def test(self, gnn_model, data_loader):
        n_samples_precessed = 0
        loss_list = []
        k_model_list = []
        k_init_list = []
        graph_index = []
        for batch in data_loader:
            k_model = gnn_model(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
            k_init = batch.k_init
            loss = F.l1_loss(k_model, k_init)

            if batch.num_graphs == 1:
                loss_list.append(loss.item())
                k_model_list.append(k_model.item())
                k_init_list.append(k_init)
                graph_index.append(n_samples_precessed)
                n_samples_precessed += 1

            else:

                for g in range(batch.num_graphs):
                    loss_list.append(loss.item()[g])
                    k_model_list.append(k_model[g])
                    k_init_list.append(k_init(g))
                    graph_index.append(n_samples_precessed)
                    n_samples_precessed += 1

        loss_list = np.array(loss_list).reshape(-1)
        k_model_list = np.array(k_model_list).reshape(-1)
        k_init_list = np.array(k_init_list).reshape(-1)
        graph_index = np.array(graph_index).reshape(-1)

        loss_ave = loss_list.mean()
        k_model_ave = k_model_list.mean()
        k_init_ave = k_init_list.mean()

        return loss_ave, k_model_ave, k_init_ave

    def execute_regression(self, lr=0.0000001, n_epochs=20):

        saved_gnn_directory = './result/saved_models/'
        pathlib.Path(saved_gnn_directory).mkdir(parents=True, exist_ok=True)

        train_loaders = {}
        val_loaders = {}
        test_loaders = {}

        # load the small dataset
        small_dataset = self.instance_type + "-small"
        small_directory = './result/generated_instances/' + self.instance_type + '/' + '-small' + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'

        small_regression_samples_directory = small_directory + 'regression_samples' + '/'
        train_loader, valid_loader, test_loader = self.load_dataset(dataset_directory=small_regression_samples_directory)
        train_loaders[small_dataset] = train_loader
        val_loaders[small_dataset] = valid_loader
        test_loaders[small_dataset] = test_loader


        model_gnn = GNNPolicy()
        train_dataset = small_dataset
        valid_dataset = small_dataset
        test_dataset = small_dataset

        optimizer = torch.optim.Adam(model_gnn.parameters(), lr=lr)
        k_init = []
        k_model = []
        loss = []
        epochs = []
        for epoch in range(n_epochs):
            print(f"Epoch {epoch}")

            if epoch == 0:
                optim = None
            else:
                optim = optimizer

            train_loader = train_loaders[train_dataset]
            train_loss = self.train(model_gnn, train_loader, optim)
            print(f"Train loss: {train_loss:0.6f}")


            valid_loader = val_loaders[valid_dataset]
            valid_loss = self.train(model_gnn, valid_loader, None)
            print(f"Valid loss: {valid_loss:0.6f}")

            test_loader = test_loaders[test_dataset]
            loss_ave, k_model_ave, k_init_ave = self.test(model_gnn, test_loader)

            loss.append(loss_ave)
            k_model.append(k_model_ave)
            k_init.append(k_init_ave)
            epochs.append(epoch)

        loss_np = np.array(loss).reshape(-1)
        k_model_np = np.array(k_model).reshape(-1)
        k_init_np = np.array(k_init).reshape(-1)
        epochs_np = np.array(epochs).reshape(-1)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(2, 1, figsize=(8, 6.4))
        fig.suptitle("Test Result: prediction of initial k")
        fig.subplots_adjust(top=0.5)
        ax[0].set_title(test_dataset + '-' + self.incumbent_mode, loc='right')
        ax[0].plot(epochs_np, loss_np)
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel("loss")
        ax[1].plot(epochs_np, k_model_np, label='k-prediction')

        ax[1].plot(epochs_np, k_init_np, label='k-label')
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel("k")
        ax[1].set_ylim([0, 1.1])
        ax[1].legend()
        plt.show()


    def execute_regression_k_prime(self, lr=0.0000001, n_epochs=20):

        saved_gnn_directory = './result/saved_models/'
        pathlib.Path(saved_gnn_directory).mkdir(parents=True, exist_ok=True)

        train_loaders = {}
        val_loaders = {}
        test_loaders = {}

        # load the small dataset
        small_dataset = self.instance_type + "-small"
        small_directory = './result/generated_instances/' + self.instance_type + '/' + '-small' + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'

        small_regression_samples_directory = small_directory + 'regression_samples_k_prime' + '/'
        train_loader, valid_loader, test_loader = self.load_dataset(dataset_directory=small_regression_samples_directory)
        train_loaders[small_dataset] = train_loader
        val_loaders[small_dataset] = valid_loader
        test_loaders[small_dataset] = test_loader


        model_gnn = GNNPolicy()
        train_dataset = small_dataset
        valid_dataset = small_dataset
        test_dataset = small_dataset

        optimizer = torch.optim.Adam(model_gnn.parameters(), lr=lr)
        k_init = []
        k_model = []
        loss = []
        epochs = []
        for epoch in range(n_epochs):
            print(f"Epoch {epoch}")

            if epoch == 0:
                optim = None
            else:
                optim = optimizer

            train_loader = train_loaders[train_dataset]
            train_loss = self.train(model_gnn, train_loader, optim)
            print(f"Train loss: {train_loss:0.6f}")


            valid_loader = val_loaders[valid_dataset]
            valid_loss = self.train(model_gnn, valid_loader, None)
            print(f"Valid loss: {valid_loss:0.6f}")

            test_loader = test_loaders[test_dataset]
            loss_ave, k_model_ave, k_init_ave = self.test(model_gnn, test_loader)

            loss.append(loss_ave)
            k_model.append(k_model_ave)
            k_init.append(k_init_ave)
            epochs.append(epoch)

        loss_np = np.array(loss).reshape(-1)
        k_model_np = np.array(k_model).reshape(-1)
        k_init_np = np.array(k_init).reshape(-1)
        epochs_np = np.array(epochs).reshape(-1)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(2, 1, figsize=(8, 6.4))
        fig.suptitle("Test Result: prediction of initial k")
        fig.subplots_adjust(top=0.5)
        ax[0].set_title(test_dataset + '-' + self.incumbent_mode, loc='right')
        ax[0].plot(epochs_np, loss_np)
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel("loss")
        ax[1].plot(epochs_np, k_model_np, label='k-prediction')

        ax[1].plot(epochs_np, k_init_np, label='k-label')
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel("k")
        ax[1].set_ylim([0, 1.1])
        ax[1].legend()
        plt.show()

        torch.save(model_gnn.state_dict(),
                   saved_gnn_directory + 'trained_params_mean_' + train_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '_k_prime.pth')


    def evaluate_lb_per_instance(self, node_time_limit, total_time_limit, index_instance,
                                 reset_k_at_2nditeration=False):
        """
        evaluate a single MIP instance by two algorithms: lb-baseline and lb-pred_k
        :param node_time_limit:
        :param total_time_limit:
        :param index_instance:
        :return:
        """
        device = self.device
        gc.collect()

        filename = f'{self.directory_transformedmodel}{self.instance_type}-{str(index_instance)}_transformed.cip'
        firstsol_filename = f'{self.directory_sol}{self.incumbent_mode}-{self.instance_type}-{str(index_instance)}_transformed.sol'

        MIP_model = Model()
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


        observation, _, _, done, _ = self.env.reset(instance)
        device = self.device
        graph = BipartiteNodeData(observation.constraint_features,
                                   observation.edge_features.indices,
                                   observation.edge_features.values,
                                   observation.variable_features,
                                   device=device)
        graph = graph.to(device)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = observation.constraint_features.shape[0] + \
                           observation.variable_features.shape[
                               0]


        initial_obj = MIP_model.getSolObjVal(incumbent)
        print("Initial obj before LB: {}".format(initial_obj))

        binary_supports = binary_support(MIP_model, incumbent)
        print('binary support: ', binary_supports)

        k_model = self.regression_model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
                             graph.variable_features)

        k_pred = k_model.item() * n_binvars
        print('GNN prediction: ', k_model.item())


        if self.is_symmetric == False:
            k_pred = k_model.item() * binary_supports

        del k_model
        del graph
        del observation

        # create a copy of MIP
        MIP_model.resetParams()
        MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
            problemName='Baseline', origcopy=False)
        MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
            problemName='GNN',
            origcopy=False)
        MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
            problemName='GNN+reset',
            origcopy=False)

        print('MIP copies are created')

        MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent,
                                                MIP_copy_vars)
        MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent,
                                                  MIP_copy_vars2)
        MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent,
                                                  MIP_copy_vars3)

        print('incumbent solution is copied to MIP copies')
        MIP_model.freeProb()
        del MIP_model
        del incumbent


        # execute local branching baseline heuristic by Fischetti and Lodi
        lb_model = LocalBranching(MIP_model=MIP_model_copy, MIP_sol_bar=sol_MIP_copy, k=self.k_baseline,
                                   node_time_limit=node_time_limit,
                                   total_time_limit=total_time_limit)
        status, obj_best, elapsed_time, lb_bits, times, objs, _, _ = lb_model.mdp_localbranch(
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=False,
            policy=None,
            optimizer=None,
            device=device
        )
        print("Instance:", MIP_model_copy.getProbName())
        print("Status of LB: ", status)
        print("Best obj of LB: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        MIP_model_copy.freeProb()
        del sol_MIP_copy
        del MIP_model_copy


        # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
        lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=k_pred,
                                   node_time_limit=node_time_limit,
                                   total_time_limit=total_time_limit)
        status, obj_best, elapsed_time, lb_bits_regression_reset, times_regression_reset, objs_regression_reset, _, _ = lb_model3.mdp_localbranch(
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=reset_k_at_2nditeration,
            policy=None,
            optimizer=None,
            device=device
        )

        print("Instance:", MIP_model_copy3.getProbName())
        print("Status of LB: ", status)
        print("Best obj of LB: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        MIP_model_copy3.freeProb()
        del sol_MIP_copy3
        del MIP_model_copy3

        # execute local branching with 1. first k predicted by GNN; 2. from 2nd iteration of lb, continue lb algorithm with no further injection

        lb_model2 = LocalBranching(MIP_model=MIP_model_copy2, MIP_sol_bar=sol_MIP_copy2, k=k_pred,
                                   node_time_limit=node_time_limit,
                                   total_time_limit=total_time_limit)
        status, obj_best, elapsed_time, lb_bits_regression_noreset, times_regression_noreset, objs_regression_noreset, _, _ = lb_model2.mdp_localbranch(
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=False,
            policy=None,
            optimizer=None,
            device=device
        )

        print("Instance:", MIP_model_copy2.getProbName())
        print("Status of LB: ", status)
        print("Best obj of LB: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        MIP_model_copy2.freeProb()
        del sol_MIP_copy2
        del MIP_model_copy2

        data = [objs, times, objs_regression_noreset, times_regression_noreset, objs_regression_reset, times_regression_reset]
        filename = f'{self.directory_lb_test}lb-test-{instance_name}.pkl'  # instance 100-199
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)
        del data
        del objs
        del times
        del objs_regression_noreset
        del times_regression_noreset
        del objs_regression_reset
        del times_regression_reset
        del lb_model
        del lb_model2
        del lb_model3

        index_instance += 1
        del instance
        return index_instance
    def evaluate_lb_per_instance_k_prime(self, node_time_limit, total_time_limit, index_instance,
                                 reset_k_at_2nditeration=False):
        """
        evaluate a single MIP instance by two algorithms: lb-baseline and lb-pred_k
        :param node_time_limit:
        :param total_time_limit:
        :param index_instance:
        :return:
        """
        device = self.device
        gc.collect()

        filename = f'{self.directory_transformedmodel}{self.instance_type}-{str(index_instance)}_transformed.cip'
        firstsol_filename = f'{self.directory_sol}{self.incumbent_mode}-{self.instance_type}-{str(index_instance)}_transformed.sol'

        MIP_model = Model()
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


        instance = ecole.scip.Model.from_pyscipopt(MIP_model)
        observation, _, _, done, _ = self.env.reset(instance)
        device = self.device
        graph = BipartiteNodeData(observation.constraint_features,
                                   observation.edge_features.indices,
                                   observation.edge_features.values,
                                   observation.variable_features,
                                   device=device)
        graph = graph.to(device)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = observation.constraint_features.shape[0] + \
                           observation.variable_features.shape[
                               0]


        # create a copy of MIP
        MIP_model.resetParams()
        MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
            problemName='Baseline', origcopy=False)
        MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
            problemName='GNN+reset',
            origcopy=False)

        print('MIP copies are created')

        MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent,
                                                MIP_copy_vars)
        MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent,
                                                  MIP_copy_vars3)

        print('incumbent solution is copied to MIP copies')

        # solve the root node and get the LP solution, compute k_prime
        k_prime = self.compute_k_prime(MIP_model, incumbent)

        initial_obj = MIP_model.getSolObjVal(incumbent)
        print("Initial obj before LB: {}".format(initial_obj))

        binary_supports = binary_support(MIP_model, incumbent)
        print('binary support: ', binary_supports)

        k_model = self.regression_model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
                                            graph.variable_features)

        k_pred = k_model.item() * k_prime
        print('GNN prediction: ', k_model.item())

        if self.is_symmetric == False:
            k_pred = k_model.item() * k_prime

        del k_model
        del graph
        del observation

        MIP_model.freeProb()
        del MIP_model
        del incumbent


        # execute local branching baseline heuristic by Fischetti and Lodi
        lb_model = LocalBranching(MIP_model=MIP_model_copy, MIP_sol_bar=sol_MIP_copy, k=self.k_baseline,
                                   node_time_limit=node_time_limit,
                                   total_time_limit=total_time_limit)
        status, obj_best, elapsed_time, lb_bits, times, objs, _, _ = lb_model.mdp_localbranch(
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=False,
            policy=None,
            optimizer=None,
            device=device
        )
        print("Instance:", MIP_model_copy.getProbName())
        print("Status of LB: ", status)
        print("Best obj of LB: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        MIP_model_copy.freeProb()
        del sol_MIP_copy
        del MIP_model_copy


        # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
        lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=k_pred,
                                   node_time_limit=node_time_limit,
                                   total_time_limit=total_time_limit)
        status, obj_best, elapsed_time, lb_bits_regression_reset, times_regression_reset, objs_regression_reset, _, _ = lb_model3.mdp_localbranch(
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=reset_k_at_2nditeration,
            policy=None,
            optimizer=None,
            device=device
        )

        print("Instance:", MIP_model_copy3.getProbName())
        print("Status of LB: ", status)
        print("Best obj of LB: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        MIP_model_copy3.freeProb()
        del sol_MIP_copy3
        del MIP_model_copy3


        data = [objs, times, objs_regression_reset, times_regression_reset]
        filename = f'{self.directory_lb_test}lb-test-{instance_name}.pkl'  # instance 100-199
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)
        del data
        del objs
        del times
        del objs_regression_reset
        del times_regression_reset
        del lb_model
        del lb_model3

        index_instance += 1
        del instance
        return index_instance

    def evaluate_localbranching(self, test_instance_size='-small', train_instance_size='-small', total_time_limit=60,
                                node_time_limit=30, reset_k_at_2nditeration=False):

        self.train_dataset = self.instance_type + train_instance_size
        self.evaluation_dataset = self.instance_type + test_instance_size

        direc = './data/generated_instances/' + self.instance_type + '/' + test_instance_size + '/'
        self.directory_transformedmodel = direc + 'transformedmodel' + '/'
        self.directory_sol = direc + self.incumbent_mode + '/'

        self.k_baseline = 20

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_baseline = self.k_baseline / 2
        total_time_limit = total_time_limit
        node_time_limit = node_time_limit

        self.saved_gnn_directory = './result/saved_models/'
        self.regression_model_gnn = GNNPolicy()
        self.regression_model_gnn.load_state_dict(torch.load(
            self.saved_gnn_directory + 'trained_params_mean_' + self.train_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '.pth'))
        self.regression_model_gnn.to(self.device)

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        self.directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        pathlib.Path(self.directory_lb_test).mkdir(parents=True, exist_ok=True)

        index_instance = 161

        while index_instance < 165:
            index_instance = self.evaluate_lb_per_instance(node_time_limit=node_time_limit,
                                                           total_time_limit=total_time_limit,
                                                           index_instance=index_instance,
                                                           reset_k_at_2nditeration=reset_k_at_2nditeration)

    def solve2opt_evaluation(self, test_instance_size='-small'):

        self.evaluation_dataset = self.instance_type + test_instance_size
        directory_opt = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + 'opt_solution' + '/'
        pathlib.Path(directory_opt).mkdir(parents=True, exist_ok=True)

        self.generator = generator_switcher(self.evaluation_dataset)
        self.generator.seed(self.seed)

        index_instance = 0
        while index_instance < 200:

            instance = next(self.generator)
            MIP_model = instance.as_pyscipopt()
            MIP_model.setProbName(self.instance_type + test_instance_size + '-' + str(index_instance))
            instance_name = MIP_model.getProbName()
            print(' \n')
            print(instance_name)

            n_vars = MIP_model.getNVars()
            n_binvars = MIP_model.getNBinVars()
            print("N of variables: {}".format(n_vars))
            print("N of binary vars: {}".format(n_binvars))
            print("N of constraints: {}".format(MIP_model.getNConss()))

            valid, MIP_model, incumbent_solution = self.initialize_MIP(MIP_model)

            if valid:
                if index_instance > 99:
                    MIP_model.resetParams()
                    MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
                        problemName='Baseline', origcopy=False)

                    MIP_model_copy.setParam('presolving/maxrounds', 0)
                    MIP_model_copy.setParam('presolving/maxrestarts', 0)
                    MIP_model_copy.setParam("display/verblevel", 0)
                    MIP_model_copy.optimize()
                    status = MIP_model_copy.getStatus()
                    if status == 'optimal':
                        obj = MIP_model_copy.getObjVal()
                        time = MIP_model_copy.getSolvingTime()
                        data = [obj, time]

                        filename = f'{directory_opt}{instance_name}-optimal-obj-time.pkl'
                        with gzip.open(filename, 'wb') as f:
                            pickle.dump(data, f)
                        del data
                    else:
                        print('Warning: solved problem ' + instance_name + ' is not optimal!')

                    print("instance:", MIP_model_copy.getProbName(),
                          "status:", MIP_model_copy.getStatus(),
                          "best obj: ", MIP_model_copy.getObjVal(),
                          "solving time: ", MIP_model_copy.getSolvingTime())

                    MIP_model_copy.freeProb()
                    del MIP_copy_vars
                    del MIP_model_copy

                index_instance += 1

            else:
                print('This instance is not valid for evaluation')

            MIP_model.freeProb()
            del MIP_model
            del incumbent_solution
            del instance

    def primal_integral(self, test_instance_size, total_time_limit=60, node_time_limit=30):

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/'
            directory_lb_test_2 = directory_2 + 'lb-from-' +  'rootsol' + '-t_node' + str(30) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/'
            directory_lb_test_2 = directory_2 + 'lb-from-' + 'firstsol' + '-t_node' + str(30) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        primal_int_baselines = []
        primal_int_preds = []
        primal_int_preds_reset = []
        primal_gap_final_baselines = []
        primal_gap_final_preds = []
        primal_gap_final_preds_reset = []
        steplines_baseline = []
        steplines_pred = []
        steplines_pred_reset = []

        for i in range(100, 200):

            instance_name = self.instance_type + '-' + str(i) + '_transformed'  # instance 100-199

            filename = f'{directory_lb_test}lb-test-{instance_name}.pkl'

            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs, times, objs_pred, times_pred, objs_pred_reset, times_pred_reset = data  # objs contains objs of a single instance of a lb test


            a = [objs.min(), objs_pred.min(), objs_pred_reset.min()]
            obj_opt = np.amin(a)

            # compute primal gap for baseline localbranching run
            # if times[-1] < total_time_limit:
            times = np.append(times, total_time_limit)
            objs = np.append(objs, objs[-1])

            gamma_baseline = np.zeros(len(objs))
            for j in range(len(objs)):
                if objs[j] == 0 and obj_opt == 0:
                    gamma_baseline[j] = 0
                elif objs[j] * obj_opt < 0:
                    gamma_baseline[j] = 1
                else:
                    gamma_baseline[j] = np.abs(objs[j] - obj_opt) / np.maximum(np.abs(objs[j]), np.abs(obj_opt))

            # compute the primal gap of last objective
            primal_gap_final_baseline = np.abs(objs[-1] - obj_opt) / np.abs(obj_opt)
            primal_gap_final_baselines.append(primal_gap_final_baseline)

            # create step line
            stepline_baseline = interp1d(times, gamma_baseline, 'previous')
            steplines_baseline.append(stepline_baseline)

            # compute primal integral
            primal_int_baseline = 0
            for j in range(len(objs) - 1):
                primal_int_baseline += gamma_baseline[j] * (times[j + 1] - times[j])
            primal_int_baselines.append(primal_int_baseline)


            # lb-gnn
            # if times_pred[-1] < total_time_limit:
            times_pred = np.append(times_pred, total_time_limit)
            objs_pred = np.append(objs_pred, objs_pred[-1])

            gamma_pred = np.zeros(len(objs_pred))
            for j in range(len(objs_pred)):
                if objs_pred[j] == 0 and obj_opt == 0:
                    gamma_pred[j] = 0
                elif objs_pred[j] * obj_opt < 0:
                    gamma_pred[j] = 1
                else:
                    gamma_pred[j] = np.abs(objs_pred[j] - obj_opt) / np.maximum(np.abs(objs_pred[j]), np.abs(obj_opt))

            primal_gap_final_pred = np.abs(objs_pred[-1] - obj_opt) / np.abs(obj_opt)
            primal_gap_final_preds.append(primal_gap_final_pred)

            stepline_pred = interp1d(times_pred, gamma_pred, 'previous')
            steplines_pred.append(stepline_pred)


            # compute primal interal
            primal_int_pred = 0
            for j in range(len(objs_pred) - 1):
                primal_int_pred += gamma_pred[j] * (times_pred[j + 1] - times_pred[j])
            primal_int_preds.append(primal_int_pred)

            # lb-gnn-reset
            times_pred_reset = np.append(times_pred_reset, total_time_limit)
            objs_pred_reset = np.append(objs_pred_reset, objs_pred_reset[-1])

            gamma_pred_reset = np.zeros(len(objs_pred_reset))
            for j in range(len(objs_pred_reset)):
                if objs_pred_reset[j] == 0 and obj_opt == 0:
                    gamma_pred_reset[j] = 0
                elif objs_pred_reset[j] * obj_opt < 0:
                    gamma_pred_reset[j] = 1
                else:
                    gamma_pred_reset[j] = np.abs(objs_pred_reset[j] - obj_opt) / np.maximum(np.abs(objs_pred_reset[j]), np.abs(obj_opt))

            primal_gap_final_pred_reset = np.abs(objs_pred_reset[-1] - obj_opt) / np.abs(obj_opt)
            primal_gap_final_preds_reset.append(primal_gap_final_pred_reset)

            stepline_pred_reset = interp1d(times_pred_reset, gamma_pred_reset, 'previous')
            steplines_pred_reset.append(stepline_pred_reset)

            # compute primal interal
            primal_int_pred_reset = 0
            for j in range(len(objs_pred_reset) - 1):
                primal_int_pred_reset += gamma_pred_reset[j] * (times_pred_reset[j + 1] - times_pred_reset[j])
            primal_int_preds_reset.append(primal_int_pred_reset)


        primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_preds = np.array(primal_int_preds).reshape(-1)
        primal_int_preds_reset = np.array(primal_int_preds_reset).reshape(-1)

        primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_preds = np.array(primal_gap_final_preds).reshape(-1)
        primal_gap_final_preds_reset = np.array(primal_gap_final_preds_reset).reshape(-1)

        # average primal integral over test dataset
        primal_int_base_ave = primal_int_baselines.sum() / len(primal_int_baselines)
        primal_int_pred_ave = primal_int_preds.sum() / len(primal_int_preds)
        primal_int_pred_ave_reset = primal_int_preds_reset.sum() / len(primal_int_preds_reset)

        primal_gap_final_baselines = primal_gap_final_baselines.sum() / len(primal_gap_final_baselines)
        primal_gap_final_preds = primal_gap_final_preds.sum() / len(primal_gap_final_preds)
        primal_gap_final_preds_reset = primal_gap_final_preds_reset.sum() / len(primal_gap_final_preds_reset)

        print(self.instance_type + self.instance_size)
        print(self.incumbent_mode + 'Solution')
        print('baseline primal integral: ', primal_int_base_ave)
        print('k_pred primal integral: ', primal_int_pred_ave)
        print('k_pred_reset primal integral: ', primal_int_pred_ave_reset)
        print('\n')
        print('baseline primal gap: ',primal_gap_final_baselines)
        print('k_pred primal gap: ', primal_gap_final_preds)
        print('k_pred_reset primal gap: ', primal_gap_final_preds_reset)

        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)
        primalgaps_baseline = None
        for n, stepline_baseline in enumerate(steplines_baseline):
            primal_gap = stepline_baseline(t)
            if n==0:
                primalgaps_baseline = primal_gap
            else:
                primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        primalgap_baseline_ave = np.average(primalgaps_baseline, axis=0)

        primalgaps_pred = None
        for n, stepline_pred in enumerate(steplines_pred):
            primal_gap = stepline_pred(t)
            if n == 0:
                primalgaps_pred = primal_gap
            else:
                primalgaps_pred = np.vstack((primalgaps_pred, primal_gap))
        primalgap_pred_ave = np.average(primalgaps_pred, axis=0)

        primalgaps_pred_reset = None
        for n, stepline_pred_reset in enumerate(steplines_pred_reset):
            primal_gap = stepline_pred_reset(t)
            if n == 0:
                primalgaps_pred_reset = primal_gap
            else:
                primalgaps_pred_reset = np.vstack((primalgaps_pred_reset, primal_gap))
        primalgap_pred_ave_reset = np.average(primalgaps_pred_reset, axis=0)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle("Normalized primal gap")
        ax.set_title(self.instance_type + '-' + self.incumbent_mode, loc='right')
        ax.plot(t, primalgap_baseline_ave, label='lb-baseline')
        ax.plot(t, primalgap_pred_ave, label='lb-gnn')
        ax.plot(t, primalgap_pred_ave_reset, '--', label='lb-gnn-reset')
        ax.set_xlabel('time /s')
        ax.set_ylabel("normalized primal gap")
        ax.legend()
        plt.show()
