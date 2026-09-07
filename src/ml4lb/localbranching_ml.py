import pyscipopt
from pyscipopt import Model
import ecole
import numpy as np
import random
import pathlib
import gzip
import pickle
import matplotlib.pyplot as plt
from ml4lb.utilities import lbconstraint_modes, instancetypes, incumbent_modes, t_reward_types, generator_switcher, binary_support, copy_sol, mean_filter, haming_distance_solutions, haming_distance_solutions_asym, mean_shift, k_0_bank
from ml4lb.localbranching import addLBConstraint, addLBConstraintAsymmetric
from ml4lb.ecole_extend.environment_extend import SimpleConfiguring, SimpleConfiguringEnablecuts
from ml4lb.models import GraphDataset, GNNPolicy, BipartiteNodeData
import torch.nn.functional as F
import torch_geometric
import torch
from torch.utils.data import DataLoader, ConcatDataset
from scipy.interpolate import interp1d
from ml4lb.localbranching import LocalBranching
from ml4lb.event import StopWhenFirstLPSolvedEventHandler

import gc

from ml4lb.models_rl import SimplePolicy, AgentReinforce
from ml4lb.dataset import InstanceDataset, custom_collate, InstanceDataset_2

'''
This file implements the classes/methods for training and testing the ML models (both the regression
model for predicting the initial neighborhood size k and the RL models for adapting k and t) for
local branching. Instances and incumbent solutions are loaded from files on disk.
'''


class MlLocalbranch:
    """Base class shared by the ML-for-LB training and evaluation classes.

    Holds the experiment configuration (dataset, instance size, LB constraint
    mode, incumbent mode, seed, device), sets up the ecole environment for
    computing incumbents/features, and provides dataset loading and
    primal-integral helpers.
    """

    def __init__(self, instance_type, instance_size, lbconstraint_mode, incumbent_mode='firstsol', seed=100, enable_gpu=False):
        self.instance_type = instance_type
        self.instance_size = instance_size
        self.incumbent_mode = incumbent_mode
        self.lbconstraint_mode = lbconstraint_mode
        self.seed = seed
        print('seed: {}'.format(str(seed)))
        self.directory = './result/generated_instances/' + self.instance_type + '/' + self.instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'

        self.initialize_ecole_env()
        self.env.seed(self.seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.enable_gpu = enable_gpu
        if self.enable_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        print(self.device)

        self.k_baseline = 20
        self.k_prime_ratio_baseline = 0.0

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

    def compute_k_prime(self, MIP_model, incumbent):

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
        print("* Model status: %s" % status)
        print("* Solve stage: %s" % stage)
        print("* LP status: %s" % lp_status)
        print('* number of sol : ', n_sols)

        sol_lp = MIP_model.createLPSol()

        k_prime = haming_distance_solutions(MIP_model, incumbent, sol_lp)
        if not self.is_symmetric:
            k_prime = haming_distance_solutions_asym(MIP_model, incumbent, sol_lp)
        k_prime = np.ceil(k_prime)

        return k_prime

    def load_mip_dataset(self, instances_directory=None, sols_directory=None, incumbent_mode=None):
        instance_filename = f'{self.instance_type}-*_transformed.cip'
        sol_filename = f'{incumbent_mode}-{self.instance_type}-*_transformed.sol'

        train_instances_directory = instances_directory + 'train/'
        instance_files = [str(path) for path in sorted(pathlib.Path(train_instances_directory).glob(instance_filename), key=lambda path: int(path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        instance_train_files = instance_files[:int(7/8 * len(instance_files))]
        instance_valid_files = instance_files[int(7/8 * len(instance_files)):]

        test_instances_directory = instances_directory + 'test/'
        instance_test_files = [str(path) for path in sorted(pathlib.Path(test_instances_directory).glob(instance_filename),
                                                       key=lambda path: int(
                                                           path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        train_sols_directory = sols_directory + 'train/'
        sol_files = [str(path) for path in sorted(pathlib.Path(train_sols_directory).glob(sol_filename), key=lambda path: int(path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        sol_train_files = sol_files[:int(7/8 * len(sol_files))]
        sol_valid_files = sol_files[int(7/8 * len(sol_files)):]

        test_sols_directory = sols_directory + 'test/'
        sol_test_files = [str(path) for path in sorted(pathlib.Path(test_sols_directory).glob(sol_filename),
                                                  key=lambda path: int(path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        train_dataset = InstanceDataset(mip_files=instance_train_files, sol_files=sol_train_files)
        valid_dataset = InstanceDataset(mip_files=instance_valid_files, sol_files=sol_valid_files)
        test_dataset = InstanceDataset(mip_files=instance_test_files, sol_files=sol_test_files)

        return train_dataset, valid_dataset, test_dataset

    def load_test_mip_dataset(self, instances_directory=None, sols_directory=None, incumbent_mode=None):
        instance_filename = f'{self.instance_type}-*_transformed.cip'
        sol_filename = f'{incumbent_mode}-{self.instance_type}-*_transformed.sol'

        test_instances_directory = instances_directory + 'test/'
        instance_test_files = [str(path) for path in sorted(pathlib.Path(test_instances_directory).glob(instance_filename),
                                                       key=lambda path: int(
                                                           path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        test_sols_directory = sols_directory + 'test/'
        sol_test_files = [str(path) for path in sorted(pathlib.Path(test_sols_directory).glob(sol_filename),
                                                  key=lambda path: int(path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        test_dataset = InstanceDataset(mip_files=instance_test_files, sol_files=sol_test_files)

        return test_dataset

    def load_results_file_list(self, instances_directory=None):
        instance_filename = f'{self.instance_type}-*_transformed.cip'

        test_instances_directory = instances_directory
        instance_test_files = [str(path) for path in
                               sorted(pathlib.Path(test_instances_directory).glob(instance_filename),
                                      key=lambda path: int(
                                          path.stem.replace('-', '_').rsplit("_", 2)[1]))]


        return instance_test_files

    def compute_primal_integral(self, times, objs, obj_opt, total_time_limit=60):

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
        primal_gap_final = np.abs(objs[-1] - obj_opt) / np.abs(obj_opt) * 100

        # create step line
        stepline = interp1d(times, gamma_baseline, 'previous')


        # compute primal integral
        primal_integral = 0
        for j in range(len(objs) - 1):
            primal_integral += gamma_baseline[j] * (times[j + 1] - times[j])

        return primal_integral, primal_gap_final, stepline

    def compute_primal_integral_2(self, times, objs, obj_opt, total_time_limit=60):

        times = np.append(times, total_time_limit)
        objs = np.append(objs, objs[-1])

        primal_integral_array = np.zeros(len(objs))
        gamma_baseline = np.zeros(len(objs))
        for j in range(len(objs)):
            if objs[j] == 0 and obj_opt == 0:
                gamma_baseline[j] = 0
            elif objs[j] * obj_opt < 0:
                gamma_baseline[j] = 1
            else:
                gamma_baseline[j] = np.abs(objs[j] - obj_opt) / np.maximum(np.abs(objs[j]), np.abs(obj_opt))

        # compute the primal gap of last objective
        primal_gap_final = np.abs(objs[-1] - obj_opt) / np.abs(obj_opt) * 100

        # compute primal integral
        primal_integral_array[0] = 0
        primal_integral = 0
        for j in range(len(objs) - 1):
            primal_integral += gamma_baseline[j] * (times[j + 1] - times[j])
            primal_integral_array[j+1] = primal_integral

        # create step line
        gamma_stepline = interp1d(times, gamma_baseline, 'previous')
        primal_integral_stepline = interp1d(times, primal_integral_array)

        return primal_integral, primal_gap_final, gamma_stepline, primal_integral_stepline # , gamma_baseline

        # gamma_baseline


class RegressionInitialK_KPrime(MlLocalbranch):
    """Data collection, training and evaluation of the regression model that
    predicts the initial neighborhood size as a fraction of k' (the binary
    support of the incumbent solution).

    Provides sample generation (generate_k_samples_k_prime,
    generate_regression_samples_k_prime), model training
    (execute_regression_k_prime, execute_regression_mergedatasets) and the
    evaluation of the LB heuristics lb-baseline, lb-sr and lb-srm of
    Section 5 (evaluate_localbranching_k_prime), together with the
    primal-integral post-processing of the stored results.
    """

    def __init__(self, instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed=100, enable_gpu=False):
        super().__init__(instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed, enable_gpu)

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False

    def sample_k_per_instance_k_prime(self, t_limit, index_instance):
        """Measure the LB performance curve over neighborhood sizes for one instance.

        For the instance with the given index, LB probing runs are executed
        for the neighborhood-size ratios alpha = 0.00, 0.01, ..., 1.00 (a
        shorter grid for CA and MIS-rootsol): each run solves one LB sub-MIP
        whose neighborhood size is ceil(alpha * k'), where k' is the Hamming
        distance between the incumbent and the root-node LP solution (see
        compute_k_prime). The best objective and the solving time of every
        run are recorded and saved as '<instance name>.npz' (arrays:
        neigh_sizes, objs, t) into self.k_samples_directory; these raw curves
        are turned into regression labels by generate_regression_samples_k_prime.

        :param t_limit: time limit (s) of each LB probing run.
        :param index_instance: index of the instance to measure.
        :return: index of the next instance (index_instance + 1).
        """
        # Load the instance and its stored incumbent solution.
        filename = f'{self.directory_transformedmodel}{self.instance_type}-{str(index_instance)}_transformed.cip'
        sol_filename = f'{self.directory_sol}{self.incumbent_mode}-{self.instance_type}-{str(index_instance)}_transformed.sol'

        MIP_model = Model()
        MIP_model.readProblem(filename)
        instance_name = MIP_model.getProbName()
        print(instance_name)
        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

        incumbent = MIP_model.readSolFile(sol_filename)

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

        # Per-probe records: neighborhood ratio, best objective, solving time,
        # binary support of the best solution and SCIP status.
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

        # Number of probing runs, i.e. of sampled ratios alpha = 0.01 * i.
        # A shorter grid is used for datasets whose sub-MIPs are expensive.
        nsample = 101
        if self.instance_type == instancetypes[2]:
            nsample = 60
        if self.instance_type == instancetypes[1] and self.incumbent_mode == incumbent_modes[1]:
            nsample = 60

        # k' = Hamming distance between the incumbent and the root-node LP
        # solution (asymmetric variant over the incumbent support); the
        # sampled neighborhood sizes are expressed as fractions of k'.
        k_base = n_binvars
        if self.is_symmetric == False:
            k_base = n_supportbinvars
        k_prime = self.compute_k_prime(MIP_model, incumbent)
        phi_prime = k_prime / k_base
        print('phi_prime :', phi_prime)

        MIP_model.freeProb()

        # Probing loop: one LB sub-MIP per sampled ratio alpha.
        for i in range(nsample):

            # Reload a fresh copy of the instance with its incumbent.
            subMIP_model = MIP_model

            subMIP_model.readProblem(filename)
            sol_subMIP_model = subMIP_model.readSolFile(sol_filename)
            feas = subMIP_model.checkSol(sol_subMIP_model)
            try:
                subMIP_model.addSol(sol_subMIP_model, False)
            except:
                print('Error: the root solution of ' + instance_name + ' is not feasible!')

            # Add the LB constraint with neighborhood size ceil(alpha * k').
            alpha = 0.01 * (i)

            if self.lbconstraint_mode == 'asymmetric':
                neigh_size = np.ceil(alpha * k_prime)
                subMIP_model, constraint_lb = addLBConstraintAsymmetric(subMIP_model, sol_subMIP_model, neigh_size)
            else:
                neigh_size = np.ceil(alpha * k_prime)
                subMIP_model, constraint_lb = addLBConstraint(subMIP_model, sol_subMIP_model, neigh_size)

            print('Neigh size:', alpha)

            # Solve the LB sub-MIP for t_limit seconds and record the outcome.
            subMIP_model.resetParams()
            subMIP_model.setParam('limits/time', t_limit)
            subMIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.FAST)
            subMIP_model.setPresolve(pyscipopt.SCIP_PARAMSETTING.FAST)
            subMIP_model.setParam("display/verblevel", 0)
            subMIP_model.optimize()

            status_subMIP = subMIP_model.getStatus()
            print("Solve status :",status_subMIP)
            best_obj = subMIP_model.getSolObjVal(subMIP_model.getBestSol())
            solving_time = subMIP_model.getSolvingTime()  # total time used for solving (including presolving) the current problem
            n_node = subMIP_model.getNTotalNodes()
            firstlp_time = subMIP_model.getFirstLpTime()  # time for solving first LP rexlaxation at the root node
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


            subMIP_model.freeTransform()
            subMIP_model.resetParams()
            subMIP_model.delCons(constraint_lb)
            subMIP_model.releasePyCons(constraint_lb)
            del constraint_lb
            print('Number of solutions: ', subMIP_model.getNSols())
            subMIP_model.freeProb()


        for i in range(len(t)):
            print('Neighsize: {:.4f}'.format(neigh_sizes[i]),
                  'Best obj: {:.4f}'.format(objs[i]),
                  'Binary supports:{}'.format(n_supportbins[i]),
                  'Solving time: {:.4f}'.format(t[i]),
                  'Status: {}'.format(statuss[i])
                  )

        neigh_sizes = np.array(neigh_sizes).reshape(-1)
        t = np.array(t).reshape(-1)
        objs = np.array(objs).reshape(-1)

        # Save the raw performance curves of this instance; they are the
        # input of generate_regression_samples_k_prime.
        saved_name = f'{self.instance_type}-{str(index_instance)}_transformed'
        f = self.k_samples_directory + saved_name
        np.savez(f, neigh_sizes=neigh_sizes, objs=objs, t=t)

        # For logging only: compute the best ratio phi_0_star of this instance
        # exactly as generate_regression_samples_k_prime will do (normalize,
        # smooth, and minimize the combined time/objective score).
        t = t / t_limit
        objs_abs = objs
        objs = (objs_abs - np.min(objs_abs))
        objs = objs / np.max(objs)

        t = mean_filter(t, 5)
        objs = mean_filter(objs, 5)

        alpha = 1 / 2
        perf_score = alpha * t + (1 - alpha) * objs
        phi_bests = neigh_sizes[np.where(perf_score == perf_score.min())]
        phi_init = phi_bests[0]
        if phi_init > self.phi_max:
            self.phi_max = phi_init
        print('phi_0_star:', phi_init)
        print('phi_0_max:', self.phi_max)

        index_instance += 1

        return index_instance

    def sample_k_integrality_grip_per_instance_k_prime(self, t_limit, index_instance):
        """Diagnostic variant of sample_k_per_instance_k_prime (not part of the pipeline).

        Probes a coarse grid of ratios (alpha = 0.0, 0.1, ...) and, for each
        probe, additionally re-solves the LP relaxation of the sub-MIP to
        measure the 'relaxation grip' (fraction of integer variables with an
        integral LP value) and the RINS fixing ratio, and plots the curves
        per instance. Unlike sample_k_per_instance_k_prime it does NOT save
        .npz sample files, so it cannot feed the regression pipeline; it was
        used for the analysis of the LB neighborhoods only.

        :param t_limit: time limit (s) of each LB probing run.
        :param index_instance: index of the instance to analyze.
        :return: index of the next instance (index_instance + 1).
        """
        filename = f'{self.directory_transformedmodel}{self.instance_type}-{str(index_instance)}_transformed.cip'
        sol_filename = f'{self.directory_sol}{self.incumbent_mode}-{self.instance_type}-{str(index_instance)}_transformed.sol'

        MIP_model = Model()
        MIP_model.readProblem(filename)
        instance_name = MIP_model.getProbName()
        print(instance_name)
        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

        incumbent = MIP_model.readSolFile(sol_filename)

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
        RINS_fixing_ratios = []
        n_nodes = []
        firstlp_times = []
        n_lps = []
        presolve_times = []

        nsample = 11
        if self.instance_type == instancetypes[2]:
            nsample = 60
        if self.instance_type == instancetypes[1] and self.incumbent_mode == incumbent_modes[1]:
            nsample = 60


        k_base = n_binvars
        if self.is_symmetric == False:
            k_base = n_supportbinvars
        # solve the root node and get the LP solution
        k_prime = self.compute_k_prime(MIP_model, incumbent)
        phi_prime = k_prime / k_base
        print('phi_prime :', phi_prime)

        MIP_model.freeProb()
        subMIP_model2 = Model()

        for i in range(nsample):


            subMIP_model = MIP_model


            subMIP_model.readProblem(filename)
            sol_subMIP_model = subMIP_model.readSolFile(sol_filename)
            feas = subMIP_model.checkSol(sol_subMIP_model)
            try:
                subMIP_model.addSol(sol_subMIP_model, False)
            except:
                print('Error: the root solution of ' + instance_name + ' is not feasible!')

            # add LB constraint to subMIP model
            alpha = 0.1 * (i)

            if self.lbconstraint_mode == 'asymmetric':
                neigh_size = np.ceil(alpha * k_prime)
                subMIP_model, constraint_lb = addLBConstraintAsymmetric(subMIP_model, sol_subMIP_model, neigh_size)
            else:
                neigh_size = np.ceil(alpha * k_prime)
                subMIP_model, constraint_lb = addLBConstraint(subMIP_model, sol_subMIP_model, neigh_size)

            print('\n Neigh size:', alpha)
            print('solve LB subMIP')


            subMIP_model.resetParams()
            subMIP_model.setParam('limits/time', t_limit)
            subMIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.FAST)
            subMIP_model.setPresolve(pyscipopt.SCIP_PARAMSETTING.FAST)
            subMIP_model.setParam("display/verblevel", 0)

            subMIP_model.optimize()

            status_subMIP = subMIP_model.getStatus()
            lp_status = subMIP_model.getLPSolstat()

            print("Solve status :",status_subMIP)
            print("LP status: %s" % lp_status)

            best_obj = subMIP_model.getSolObjVal(subMIP_model.getBestSol())
            solving_time = subMIP_model.getSolvingTime()  # total time used for solving (including presolving) the current problem
            n_node = subMIP_model.getNTotalNodes()
            firstlp_time = subMIP_model.getFirstLpTime()  # time for solving first LP rexlaxation at the root node
            print('sovling time: ', solving_time)
            print('first LP time', firstlp_time)

            presolve_time = subMIP_model.getPresolvingTime()
            n_lp = subMIP_model.getNLPs()
            print('number of LP sol : ', n_lp)

            best_sol = subMIP_model.getBestSol()

            vars_subMIP = subMIP_model.getVars()
            n_binvars_subMIP = subMIP_model.getNBinVars()
            n_supportbins_subMIP = 0
            for i in range(n_binvars_subMIP):
                val = subMIP_model.getSolVal(best_sol, vars_subMIP[i])
                assert subMIP_model.isFeasIntegral(val), "Error: Value of a binary varialbe is not integral!"
                if subMIP_model.isFeasEQ(val, 1.0):
                    n_supportbins_subMIP += 1

            # test of integrality grip
            print('re-solve LP relaxation')

            subMIP_model.freeTransform()


            subMIP_model2, subMIP_model2_vars, success = subMIP_model.createCopy(problemName='SolveLPRelax', origcopy=True)
            subMIP_model2, sol_subMIP2_init = copy_sol(subMIP_model, subMIP_model2, sol_subMIP_model, subMIP_model2_vars)

            n_bins = subMIP_model2.getNBinVars()
            n_integers = subMIP_model2.getNIntVars()
            n_all_integer_vars = n_bins + n_integers

            vars_subMIP2 = subMIP_model2.getVars()
            n_vars_subMIP2 = subMIP_model2.getNVars()

            vartypes = np.empty(n_vars_subMIP2, dtype=object)
            for i in range(n_vars_subMIP2):
                vartype = vars_subMIP2[i].vtype()
                vartypes[i] = vartype
                if vartype == 'CONTINUOUS':
                    continue
                else:
                    subMIP_model2.chgVarType(vars_subMIP2[i],'C')

            subMIP_model2.resetParams()
            subMIP_model2.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
            subMIP_model2.setParam('presolving/maxrounds', 0)
            subMIP_model2.setParam('presolving/maxrestarts', 0)

            subMIP_model2.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
            subMIP_model2.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
            subMIP_model2.setIntParam("lp/solvefreq", 0)
            subMIP_model2.setParam("limits/nodes", 1)
            subMIP_model2.setParam('limits/time', 3600)
            subMIP_model2.setParam("display/verblevel", 0)

            subMIP_model2.setParam("lp/disablecutoff", 1)

            stage = subMIP_model2.getStage()
            n_sols = subMIP_model2.getNSols()
            print('before re-solving LP:')
            print('* number of LP sol : ', subMIP_model2.getNLPs())
            print('* number of sol : ', n_sols)
            print("* Solve stage: %s" % stage)


            if alpha == 0:
                stopWhenFirstLPSolvedEventHandler = StopWhenFirstLPSolvedEventHandler()
                subMIP_model2.includeEventhdlr(stopWhenFirstLPSolvedEventHandler, 'stop solving after first LP is solved', 'store the optimal LP solution if one is found')

            subMIP_model2.optimize()

            status = subMIP_model2.getStatus()
            lp_status = subMIP_model2.getLPSolstat()
            firstlp_time_2 = subMIP_model2.getFirstLpTime()
            stage = subMIP_model2.getStage()
            n_sols = subMIP_model2.getNSols()
            time = subMIP_model2.getSolvingTime()
            n_lp_2 = subMIP_model2.getNLPs()

            print('after solving subMIP2:')
            print('* solving time: ', time)
            print('*  first LP time: ', firstlp_time_2)
            print("* Model status: %s" % status)
            print("* Solve stage: %s" % stage)
            print("* LP status: %s" % lp_status)
            print('* number of LP sol : ', n_lp_2)
            print('* number of sol : ', n_sols)

            if n_lp_2 <= 1:
                print('Optimal LP sol is found after solving the first LP.')
                n_lp_integral_vars = 0
                n_rins_fixing = 0
                for i in range(n_all_integer_vars):
                    # check the integrality of LP solution for all the integer variables
                    lp_val = vars_subMIP2[i].getLPSol()
                    if subMIP_model2.isFeasIntegral(lp_val):
                        n_lp_integral_vars += 1

                    # check if LP val and incumbent val are equal for all the integer variables
                    val_incumbent_init = subMIP_model2.getSolVal(sol_subMIP2_init, vars_subMIP2[i])
                    if subMIP_model2.isFeasEQ(lp_val, val_incumbent_init):
                        n_rins_fixing += 1

                relax_grip = n_lp_integral_vars / n_all_integer_vars
                rins_fixing_ratio = n_rins_fixing / n_all_integer_vars

                print('num binary vars: ', n_bins)
                print('num all integer vars: ', n_all_integer_vars)
                print('num lp_integral: ', n_lp_integral_vars)
                print('num RINS_fixing: ', n_rins_fixing)
                print('relaxation grip :', relax_grip)
                print('RINS fixing ratio :', rins_fixing_ratio)

                relax_grips.append(relax_grip)
                RINS_fixing_ratios.append(rins_fixing_ratio)
            else:
                print('No LP sol is found after solving the first LP!')


            # restore the original type of interger and binary variables
            subMIP_model2.freeTransform()
            for i in range(n_vars_subMIP2):
                vartype = vartypes[i]
                if vartype == 'CONTINUOUS':
                    continue
                else:
                    subMIP_model2.chgVarType(vars_subMIP2[i], vartype)
            del vartypes

            neigh_sizes.append(alpha)
            objs.append(best_obj)
            t.append(solving_time)
            n_supportbins.append(n_supportbins_subMIP)
            statuss.append(status_subMIP)


            n_nodes.append(n_node)
            firstlp_times.append(firstlp_time)
            presolve_times.append(presolve_time)
            n_lps.append(n_lp_2)

            subMIP_model.freeTransform()
            subMIP_model.resetParams()
            subMIP_model.delCons(constraint_lb)
            subMIP_model.releasePyCons(constraint_lb)
            del constraint_lb
            print('Number of solutions: ', subMIP_model.getNSols())
            subMIP_model.freeProb()
            del subMIP_model

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
                  'Status: {}'.format(statuss[i])
                  )

        neigh_sizes = np.array(neigh_sizes).reshape(-1)
        t = np.array(t).reshape(-1)
        objs = np.array(objs).reshape(-1)
        relax_grips = np.array(relax_grips).reshape(-1)
        RINS_fixing_ratios = np.array(RINS_fixing_ratios).reshape(-1)


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
        phi_bests = neigh_sizes[np.where(perf_score == perf_score.min())]
        if len(phi_bests) > 0:
            phi_init = phi_bests[0]
            if phi_init > self.phi_max:
                self.phi_max = phi_init
            print('phi_0_star:', phi_init)
            print('phi_0_max:', self.phi_max)

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
        ax[2].set_ylabel("Cost")
        if len(neigh_sizes) == len(relax_grips):
            ax[3].set_title('all_int_vars:' + str(n_all_integer_vars) + ', bin_vars' + str(n_bins), fontsize=14)
            ax[3].plot(neigh_sizes, relax_grips, label='Relaxation grip')
            ax[3].plot(neigh_sizes, RINS_fixing_ratios, '--', label='incumbent and LP consistency')
            ax[3].set_ylabel("ratio")
            ax[3].legend()
        plt.show()

        index_instance += 1

        return index_instance

    def generate_k_samples_k_prime(self, t_limit, instance_size='-small'):
        """Stage 1 of the regression pipeline: measure the LB performance curves.

        For every instance of the training/test set (indices 0-199), LB
        probing runs are executed over a grid of neighborhood-size ratios
        alpha in [0, 1]: each probe solves one LB sub-MIP of neighborhood
        size ceil(alpha * k') around the stored incumbent, where k' is the
        Hamming distance between the incumbent and the root-node LP solution
        (see sample_k_per_instance_k_prime for the details).

        The raw performance curves (ratios, best objectives, solving times)
        of each instance are saved as .npz files under
        '<result dir>/k_samples_k_prime/seed<seed>/'. This is the expensive,
        solver-heavy stage; the curves are distilled into regression samples
        by generate_regression_samples_k_prime (stage 2), which can be
        re-run cheaply without repeating this stage.

        :param t_limit: time limit (s) of each LB probing run; pass the same
            value to generate_regression_samples_k_prime, which uses it to
            normalize the recorded times.
        :param instance_size: size suffix of the instance set ('-small').
        """
        # Output directory of the raw .npz sample files.
        self.k_samples_directory = self.directory + 'k_samples_k_prime' + '/' + 'seed' + str(self.seed) + '/'
        pathlib.Path(self.k_samples_directory).mkdir(parents=True, exist_ok=True)

        direc = './data/generated_instances/' + self.instance_type + '/' + instance_size + '/'

        index_instance = 0
        self.phi_max = 0  # largest best-ratio observed so far (logging only)

        # Measure every instance; indices 0-159 come from the train split,
        # 160-199 from the test split of the instance set.
        while index_instance < 200:
            if index_instance < 160:
                self.directory_transformedmodel = direc + 'transformedmodel' + '/' + 'train/'
                self.directory_sol = direc + self.incumbent_mode + '/' + 'train/'
            else:
                self.directory_transformedmodel = direc + 'transformedmodel' + '/' + 'test/'
                self.directory_sol = direc + self.incumbent_mode + '/' + 'test/'

            index_instance = self.sample_k_per_instance_k_prime(t_limit, index_instance)

    def two_examples(self):

        plt.clf()
        plt.rcParams.update({'font.size': 14})
        fig, ax = plt.subplots(2, 1, figsize=(6, 4))
        ax[0].set_xlabel(r'$\ r $')
        ax[0].set_ylabel("Objective")
        ax[1].set_ylabel("Time")
        t_limit = 3
        for i in range(2):

            directory = './result/generated_instances/' + instancetypes[i] + '/' + self.instance_size + '/' + lbconstraint_modes[1-i] + '/' + 'firstsol' + '/'

            k_samples_directory = directory + 'k_samples' + '/'

            instance_name = instancetypes[i] + '-' + str(i)
            data = np.load(k_samples_directory + instance_name + '.npz')
            k = data['neigh_sizes']
            t = data['t']
            objs_abs = data['objs']

            objs = objs_abs
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
            print('phi_0_star :', k_init)

            if instancetypes[i] == 'setcovering':
                ins_name = 'sc'
            elif instancetypes[i] == 'independentset':
                ins_name = 'mis'
            instance_name = ins_name + '-' + str(i)

            ax[0].plot(k, objs, label=instance_name)
            ax[1].plot(k, t, label=instance_name)
        ax[0].legend()
        ax[0].grid()
        ax[1].legend()
        ax[1].grid()
        plt.show()

    def generate_regression_samples_k_prime(self, t_limit, instance_size='-small'):
        """Stage 2 of the regression pipeline: build the (features, label) samples.

        For every instance (indices 100-199), the raw performance curves
        saved by generate_k_samples_k_prime (stage 1) are distilled into one
        supervised sample; no LB sub-MIP is solved here:

        - label: normalize the objective curve to [0, 1] and the time curve
          by t_limit, smooth both with a mean filter, combine them into the
          performance score (1/2 * time + 1/2 * objective), and take the
          ratio phi_0_star that minimizes it - the best neighborhood-size
          ratio of this instance;
        - features: the bipartite constraint-variable graph observation of
          the instance with its incumbent, extracted with the ecole
          environment (root node only).

        Each pair [observation, phi_0_star] is pickled into
        '<result dir>/regression_samples_k_prime/{train,test}/seed<seed>/'
        (train: indices 100-159, test: 160-199); these files are loaded by
        GraphDataset (ml4lb/models.py) when training the GNN in stage 3.

        :param t_limit: time limit (s) used for the probing runs in stage 1;
            it must match, since the recorded times are normalized by it.
        :param instance_size: size suffix of the instance set ('-small').
        """
        # Input directory (stage 1 curves) and output directories (samples).
        self.k_samples_directory = self.directory + 'k_samples_k_prime' + '/' + 'seed' + str(self.seed) + '/'
        self.regression_samples_directory_train = self.directory + 'regression_samples_k_prime' + '/train/' + 'seed' + str(self.seed) + '/'
        self.regression_samples_directory_test = self.directory + 'regression_samples_k_prime' + '/test/' + 'seed' + str(self.seed) + '/'
        pathlib.Path(self.regression_samples_directory_train).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.regression_samples_directory_test).mkdir(parents=True, exist_ok=True)

        direc = './data/generated_instances/' + self.instance_type + '/' + instance_size + '/'
        self.directory_transformedmodel = direc + 'transformedmodel' + '/train/'
        self.directory_sol = direc + self.incumbent_mode + '/train/'

        index_instance = 100
        list_phi = []
        while index_instance < 200:

            # Load the instance and its stored incumbent solution.
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

            # Load the raw performance curves measured in stage 1.
            data = np.load(self.k_samples_directory + instance_name + '.npz')
            k = data['neigh_sizes']
            t = data['t']
            objs_abs = data['objs']

            objs = objs_abs
            # Label computation: normalize the objective and solving time ...
            t = t / t_limit
            objs = (objs_abs - np.min(objs_abs))
            objs = objs / np.max(objs)

            # ... smooth both curves ...
            t = mean_filter(t, 5)
            objs = mean_filter(objs, 5)

            # ... and take the ratio minimizing the combined performance
            # score as the regression label phi_0_star of this instance.
            alpha = 1 / 2
            perf_score = alpha * t + (1 - alpha) * objs
            k_bests = k[np.where(perf_score == perf_score.min())]
            k_init = k_bests[0]
            print('phi_0_star :', k_init)
            list_phi.append(k_init)

            # Feature extraction: encode the instance plus incumbent as the
            # bipartite graph observation of the ecole environment.
            instance = ecole.scip.Model.from_pyscipopt(MIP_model)

            observation, _, _, done, _ = self.env.reset(instance)

            # Save the supervised sample [features, label]; instances
            # 100-159 form the training set, 160-199 the test set.
            data_sample = [observation, k_init]
            saved_name = f'{self.instance_type}-{str(index_instance)}_transformed'
            if index_instance < 160:
                filename = f'{self.regression_samples_directory_train}regression-{saved_name}.pkl'
            else:
                filename = f'{self.regression_samples_directory_test}regression-{saved_name}.pkl'
            with gzip.open(filename, 'wb') as f:
                pickle.dump(data_sample, f)

            index_instance += 1

        list_phi = np.array(list_phi).reshape(-1)
        phi_mean = list_phi.mean()
        print('phi_0_star mean :', phi_mean)

    def test_lp(self, t_limit, instance_size='-small'):

        self.k_samples_directory = self.directory + 'k_samples' + '/'
        self.regression_samples_directory = self.directory + 'regression_samples' + '/'
        pathlib.Path(self.regression_samples_directory).mkdir(parents=True, exist_ok=True)

        direc = './data/generated_instances/' + self.instance_type + '/' + instance_size + '/'
        self.directory_transformedmodel = direc + 'transformedmodel' + '/'
        self.directory_sol = direc + self.incumbent_mode + '/'

        index_instance = 1
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
            print('relaxation grip :', 1 - nlpcands / n_bins)

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
        print('average phi_star :', ave_phi_star)
        print('average phi_prime: ', ave_phi_prime)

    def generate_dataset(self, dataset_directory=None, filename=None):
        self.regression_samples_directory = dataset_directory
        train_directory = self.regression_samples_directory + 'train/'
        test_directory = self.regression_samples_directory + 'test/'
        sample_files = [str(path) for path in pathlib.Path(train_directory).glob(filename)]
        train_files = sample_files[:int(7/8 * len(sample_files))]
        valid_files = sample_files[int(7/8 * len(sample_files)):]

        test_files = [str(path) for path in pathlib.Path(test_directory).glob(filename)]

        train_dataset = GraphDataset(train_files)
        valid_dataset = GraphDataset(valid_files)
        test_dataset = GraphDataset(test_files)

        return train_dataset, valid_dataset, test_dataset

    def load_dataset(self, train_dataset, valid_dataset, test_dataset):

        train_loader = torch_geometric.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        valid_loader = torch_geometric.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)
        test_loader = torch_geometric.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

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
                k_model = gnn_model(batch.constraint_features, batch.edge_index, batch.edge_attr,
                                    batch.variable_features)
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

        print('phi labels :')
        print(k_init_list)
        print('phi predictions :')
        print(k_model_list)

        return loss_ave, k_model_ave, k_init_ave

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
        filename = 'regression-' + self.instance_type + '-*.pkl'

        train_dataset, valid_dataset, test_dataset = self.generate_dataset(dataset_directory=small_regression_samples_directory, filename=filename)
        train_loader, valid_loader, test_loader = self.load_dataset(train_dataset, valid_dataset, test_dataset)
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

    def execute_regression_mergedatasets(self, lr=0.0000001, n_epochs=20):

        saved_gnn_directory = './result/saved_models/regression/'
        pathlib.Path(saved_gnn_directory).mkdir(parents=True, exist_ok=True)

        train_loaders = {}
        val_loaders = {}
        test_loaders = {}

        # load the small dataset
        small_dataset =  "setcover-independentset-combinatorialauction"
        small_directory_setcover = './result/generated_instances/' + 'setcovering' + '/' + '-small' + '/' + 'asymmetric' + '/' + 'firstsol' + '/'
        small_regression_samples_directory = small_directory_setcover + 'regression_samples_k_prime' + '/'
        filename = 'regression-' + 'setcovering' + '-*.pkl'
        train_dataset_0, valid_dataset_0, test_dataset_0 = self.generate_dataset(
            dataset_directory=small_regression_samples_directory, filename=filename)

        small_directory_setcover = './result/generated_instances/' + 'setcovering' + '/' + '-small' + '/' + 'asymmetric' + '/' + 'rootsol' + '/'
        small_regression_samples_directory = small_directory_setcover + 'regression_samples_k_prime' + '/'
        filename = 'regression-' + 'setcovering' + '-*.pkl'
        train_dataset_1, valid_dataset_1, test_dataset_1 = self.generate_dataset(
            dataset_directory=small_regression_samples_directory, filename=filename)

        small_directory_independentset = './result/generated_instances/' + 'independentset' + '/' + '-small' + '/' + 'symmetric' + '/' + 'firstsol' + '/'
        small_regression_samples_directory = small_directory_independentset + 'regression_samples_k_prime' + '/'
        filename = 'regression-' + 'independentset' + '-*.pkl'
        train_dataset_2, valid_dataset_2, test_dataset_2 = self.generate_dataset(
            dataset_directory=small_regression_samples_directory, filename=filename)

        small_directory_independentset = './result/generated_instances/' + 'independentset' + '/' + '-small' + '/' + 'symmetric' + '/' + 'rootsol' + '/'
        small_regression_samples_directory = small_directory_independentset + 'regression_samples_k_prime' + '/'
        filename = 'regression-' + 'independentset' + '-*.pkl'
        train_dataset_3, valid_dataset_3, test_dataset_3 = self.generate_dataset(
            dataset_directory=small_regression_samples_directory, filename=filename)

        small_directory_combina = './result/generated_instances/' + 'combinatorialauction' + '/' + '-small' + '/' + 'symmetric' + '/' + 'firstsol' + '/'
        small_regression_samples_directory = small_directory_combina + 'regression_samples_k_prime' + '/'
        filename = 'regression-' + 'combinatorialauction' + '-*.pkl'
        train_dataset_4, valid_dataset_4, test_dataset_4 = self.generate_dataset(
            dataset_directory=small_regression_samples_directory, filename=filename)

        small_directory_combina = './result/generated_instances/' + 'combinatorialauction' + '/' + '-small' + '/' + 'symmetric' + '/' + 'rootsol' + '/'
        small_regression_samples_directory = small_directory_combina + 'regression_samples_k_prime' + '/'
        filename = 'regression-' + 'combinatorialauction' + '-*.pkl'
        train_dataset_5, valid_dataset_5, test_dataset_5 = self.generate_dataset(
            dataset_directory=small_regression_samples_directory, filename=filename)


        train_data = torch.utils.data.ConcatDataset([train_dataset_0, train_dataset_1, train_dataset_2, train_dataset_3, train_dataset_4, train_dataset_5]) #train_dataset_3, train_dataset_6, train_dataset_7
        valid_data = torch.utils.data.ConcatDataset([valid_dataset_0, valid_dataset_1, valid_dataset_2, valid_dataset_3, valid_dataset_4, valid_dataset_5]) #  valid_dataset_6, valid_dataset_7
        test_data = torch.utils.data.ConcatDataset([test_dataset_0, test_dataset_1,  test_dataset_2, test_dataset_3, test_dataset_4, test_dataset_5])# test_dataset_6, test_dataset_7]

        train_loader, valid_loader, test_loader = self.load_dataset(train_data, valid_data, test_data)
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
        for epoch in range(0, n_epochs):
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

            torch.save(model_gnn.state_dict(),
                       saved_gnn_directory + 'trained_params_mean_' + train_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '_k_prime' + '_epoch' + str(epoch) + '.pth')

            if epoch % 50 == 0 :
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
                   saved_gnn_directory + 'trained_params_mean_' + train_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '_k_prime' + '.pth')

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

        # variable features: only incumbent solution
        variable_features = observation.variable_features[:, -1:]
        # ensure the graph is created on the same device as the model
        device = self.device
        graph = BipartiteNodeData(observation.constraint_features,
                                  observation.edge_features.indices,
                                  observation.edge_features.values,
                                  variable_features,
                                  device=device)
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = observation.constraint_features.shape[0] + \
                          observation.variable_features.shape[
                              0]

        graph = graph.to(device)

        # variable features: all the variable features


        # create a copy of MIP
        MIP_model.resetParams()
        MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
            problemName='GNN+reset',
            origcopy=False)

        print('MIP copies are created')

        MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent,
                                                  MIP_copy_vars3)

        print('incumbent solution is copied to MIP copies')

        # solve the root node and get the LP solution, compute k_prime
        k_prime = self.compute_k_prime(MIP_model, incumbent)
        print('k_prime: ', k_prime)

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

        k_pred = max(k_pred, self.k_prime_ratio_baseline * k_prime)
        k_pred = max(k_pred, 10)
        k_pred = np.ceil(k_pred)

        del k_model
        del graph
        del observation

        MIP_model.freeProb()
        del MIP_model
        del incumbent


        # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
        lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=k_pred,
                                   node_time_limit=node_time_limit,
                                   total_time_limit=total_time_limit)
        status, obj_best, elapsed_time, lb_bits_regression_reset, times_regression_reset_, objs_regression_reset_, _, _ = lb_model3.mdp_localbranch(
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

        objs_regression_reset = np.array(lb_model3.primal_objs).reshape(-1)
        times_regression_reset = np.array(lb_model3.primal_times).reshape(-1)

        MIP_model_copy3.freeProb()
        del sol_MIP_copy3
        del MIP_model_copy3


        data = [objs_regression_reset, times_regression_reset] # objs, times,
        saved_name = f'{self.instance_type}-{str(index_instance)}_transformed'
        filename = f'{self.directory_lb_test}lb-test-{saved_name}.pkl'  # instance 100-199
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)
        del objs_regression_reset
        del times_regression_reset
        del lb_model3

        index_instance += 1
        del instance
        return index_instance

    def evaluate_lb_per_instance_k_prime_merged(self, node_time_limit, total_time_limit, index_instance,
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
        if not feas:
            print('Error: the initial solution of ' + instance_name + ' is not feasible!')
        else:
            print('The initial solution of ' + instance_name + ' is feasible!')
        try:
            MIP_model.addSol(incumbent, False)
        except:
            print('Error: the initial solution of ' + instance_name + ' is not feasible!')

        instance = ecole.scip.Model.from_pyscipopt(MIP_model)
        observation, _, _, done, _ = self.env.reset(instance)

        # variable features: only incumbent solution
        variable_features = observation.variable_features[:, -1:]
        device = self.device
        graph = BipartiteNodeData(observation.constraint_features,
                                  observation.edge_features.indices,
                                  observation.edge_features.values,
                                  variable_features,
                                  device=device)
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = observation.constraint_features.shape[0] + \
                          observation.variable_features.shape[
                              0]

        graph = graph.to(device)

        # variable features: all the variable features


        # create a copy of MIP
        MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
            problemName='GNN+reset',
            origcopy=False)

        print('MIP copies are created')

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

        k_pred = max(k_pred, self.k_prime_ratio_baseline * k_prime)
        k_pred = max(k_pred, 10)
        k_pred = np.ceil(k_pred)

        del k_model
        del graph
        del observation

        MIP_model.freeProb()
        del MIP_model
        del incumbent


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

        objs_regression_reset = np.array(lb_model3.primal_objs).reshape(-1)
        times_regression_reset = np.array(lb_model3.primal_times).reshape(-1)

        print("Instance:", MIP_model_copy3.getProbName())
        print("Status of LB: ", status)
        print("Best obj of LB: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        MIP_model_copy3.freeProb()
        del sol_MIP_copy3
        del MIP_model_copy3


        data = [objs_regression_reset, times_regression_reset]
        saved_name = f'{self.instance_type}-{str(index_instance)}_transformed'
        filename = f'{self.directory_lb_test}lb-test-{saved_name}.pkl'  # instance 100-199
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)
        del objs_regression_reset
        del times_regression_reset
        del lb_model3

        index_instance += 1
        del instance
        return index_instance

    def evaluate_lb_per_instance_baseline(self, node_time_limit, total_time_limit, index_instance,
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

        instance = ecole.scip.Model.from_pyscipopt(MIP_model)

        # variable features: all the variable features

        # We must tell pytorch geometric how many nodes there are, for indexing purposes


        # create a copy of MIP
        MIP_model.resetParams()


        MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
            problemName='Baseline', origcopy=False)

        print('MIP copies are created')

        MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent,
                                                MIP_copy_vars)

        print('incumbent solution is copied to MIP copies')


        initial_obj = MIP_model.getSolObjVal(incumbent)
        print("Initial obj before LB: {}".format(initial_obj))


        MIP_model.freeProb()
        del MIP_model
        del incumbent

        sol = MIP_model_copy.getBestSol()
        initial_obj = MIP_model_copy.getSolObjVal(sol)
        print("Initial obj before LB: {}".format(initial_obj))

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

        objs = np.array(lb_model.primal_objs).reshape(-1)
        times = np.array(lb_model.primal_times).reshape(-1)

        print("Instance:", MIP_model_copy.getProbName())
        print("Status of LB: ", status)
        print("Best obj of LB: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        MIP_model_copy.freeProb()
        del sol_MIP_copy
        del MIP_model_copy


        data = [objs, times]
        saved_name = f'{self.instance_type}-{str(index_instance)}_transformed'
        filename = f'{self.directory_lb_test}lb-test-{saved_name}.pkl'  # instance 100-199
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)
        del objs
        del times
        del lb_model

        index_instance += 1
        del instance
        return index_instance

    def evaluate_lb_per_instance_baseline_k0_average(self, node_time_limit, total_time_limit, index_instance,
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

        instance = ecole.scip.Model.from_pyscipopt(MIP_model)

        # variable features: all the variable features

        # We must tell pytorch geometric how many nodes there are, for indexing purposes


        # create a copy of MIP
        MIP_model.resetParams()


        MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
            problemName='Baseline', origcopy=False)

        print('MIP copies are created')

        MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent,
                                                MIP_copy_vars)

        print('incumbent solution is copied to MIP copies')

        # solve the root node and get the LP solution, compute k_prime
        k_prime = self.compute_k_prime(MIP_model, incumbent)
        k0_average =  self.k_prime_ratio_baseline * k_prime

        k0_average = max(k0_average, 10)
        k0_average = np.ceil(k0_average)

        initial_obj = MIP_model.getSolObjVal(incumbent)
        print("Initial obj before LB: {}".format(initial_obj))


        MIP_model.freeProb()
        del MIP_model
        del incumbent

        sol = MIP_model_copy.getBestSol()
        initial_obj = MIP_model_copy.getSolObjVal(sol)
        print("Initial obj before LB: {}".format(initial_obj))

        # execute local branching baseline heuristic by Fischetti and Lodi
        lb_model = LocalBranching(MIP_model=MIP_model_copy, MIP_sol_bar=sol_MIP_copy, k=k0_average,
                                  node_time_limit=node_time_limit,
                                  total_time_limit=total_time_limit)
        status, obj_best, elapsed_time, lb_bits, times, objs, _, _ = lb_model.mdp_localbranch(
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=reset_k_at_2nditeration,
            policy=None,
            optimizer=None,
            device=device
        )

        objs = np.array(lb_model.primal_objs).reshape(-1)
        times = np.array(lb_model.primal_times).reshape(-1)

        print("Instance:", MIP_model_copy.getProbName())
        print("Status of LB: ", status)
        print("Best obj of LB: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        MIP_model_copy.freeProb()
        del sol_MIP_copy
        del MIP_model_copy


        data = [objs, times]
        saved_name = f'{self.instance_type}-{str(index_instance)}_transformed'
        filename = f'{self.directory_lb_test}lb-test-{saved_name}.pkl'  # instance 100-199
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)
        del objs
        del times
        del lb_model

        index_instance += 1
        del instance
        return index_instance

    def evaluate_localbranching_k_prime(self, test_instance_size='-small', train_instance_size='-small', total_time_limit=60,
                                node_time_limit=30, reset_k_at_2nditeration=False, merged=False, baseline=False, regression_model_path=''):

        self.train_dataset = self.instance_type + train_instance_size
        self.evaluation_dataset = self.instance_type + test_instance_size

        direc = './data/generated_instances/' + self.instance_type + '/' + test_instance_size + '/'
        self.directory_transformedmodel = direc + 'transformedmodel' + '/test/'
        self.directory_sol = direc + self.incumbent_mode + '/test/'

        self.k_baseline = 20

        if self.instance_type in ['setcovering', 'independentset', 'combinatorialauction']:
            k0_ratio_average = k_0_bank[self.instance_type+'-'+self.incumbent_mode]
            self.k_prime_ratio_baseline = k0_ratio_average

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_baseline = self.k_baseline / 2
        total_time_limit = total_time_limit
        node_time_limit = node_time_limit

        self.saved_gnn_directory = './result/saved_models/'
        self.regression_model_gnn = GNNPolicy()
        if not baseline:
            if not merged:
                self.regression_model_gnn.load_state_dict(torch.load(
                    self.saved_gnn_directory + 'trained_params_mean_' + self.train_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '_k_prime.pth'))
            else:
                self.regression_model_gnn.load_state_dict(torch.load(
                    regression_model_path))

        self.regression_model_gnn.to(self.device)

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'k_prime/'

        if baseline:
            self.directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/seed'+ str(self.seed) + '/'
        else:
            if not merged:
                self.directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
                    node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/seed'+ str(self.seed) + '/'
            else:
                self.directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
                    node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/seed'+ str(self.seed) + '/'
        pathlib.Path(self.directory_lb_test).mkdir(parents=True, exist_ok=True)

        index_instance = 160
        index_max =200

        if self.instance_type == instancetypes[3]:
            index_instance = 80
            index_max = 115
        elif self.instance_type == instancetypes[4]:
            index_instance = 0
            index_max = 30

        if self.instance_type == 'combinatorialauction' and test_instance_size == '-large':
            index_instance = 0
            index_max = 40


        while index_instance < index_max:

            if self.instance_type == 'miplib_39binary' and index_instance == 18:
                index_instance = 19

            if baseline:
                index_instance = self.evaluate_lb_per_instance_baseline(node_time_limit=node_time_limit,
                                                                       total_time_limit=total_time_limit,
                                                                       index_instance=index_instance,
                                                                       reset_k_at_2nditeration=reset_k_at_2nditeration
                                                                       )
            else:
                if not merged:
                    index_instance = self.evaluate_lb_per_instance_k_prime(node_time_limit=node_time_limit,
                                                                           total_time_limit=total_time_limit,
                                                                           index_instance=index_instance,
                                                                           reset_k_at_2nditeration=reset_k_at_2nditeration
                                                                           )
                else:
                    index_instance = self.evaluate_lb_per_instance_k_prime_merged(node_time_limit=node_time_limit,
                                                                           total_time_limit=total_time_limit,
                                                                           index_instance=index_instance,
                                                                           reset_k_at_2nditeration=reset_k_at_2nditeration
                                                                           )

    def evaluate_localbranching_baseline_k0_average(self, test_instance_size='-small', train_instance_size='-small', total_time_limit=60,
                                node_time_limit=30, reset_k_at_2nditeration=False, merged=False):

        self.train_dataset = self.instance_type + train_instance_size
        self.evaluation_dataset = self.instance_type + test_instance_size

        direc = './data/generated_instances/' + self.instance_type + '/' + test_instance_size + '/'
        self.directory_transformedmodel = direc + 'transformedmodel' + '/test/'
        self.directory_sol = direc + self.incumbent_mode + '/test/'

        self.k_baseline = 20
        if not merged:
            k0_ratio_average = k_0_bank[self.instance_type+'-'+self.incumbent_mode]
        else:
            k0_ratio_average = k_0_bank['merged']

        self.k_prime_ratio_baseline = k0_ratio_average

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
        total_time_limit = total_time_limit
        node_time_limit = node_time_limit

        self.saved_gnn_directory = './result/saved_models/'
        self.regression_model_gnn = GNNPolicy()

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'k_prime/'

        if not merged:
            self.directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline_k0_average/seed'+ str(self.seed) + '/'
        else:
            self.directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(
                total_time_limit) + 's' + test_instance_size + '_baseline_k0_average_merged/seed' + str(self.seed) + '/'
        pathlib.Path(self.directory_lb_test).mkdir(parents=True, exist_ok=True)

        index_instance = 160
        index_max =200

        if self.instance_type == instancetypes[3]:
            index_instance = 80
            index_max = 115
        elif self.instance_type == instancetypes[4]:
            index_instance = 0
            index_max = 30

        if self.instance_type == 'combinatorialauction' and test_instance_size == '-large':
            index_instance = 0
            index_max = 40


        while index_instance < index_max:

            if self.instance_type == 'miplib_39binary' and index_instance == 18:
                index_instance = 19

            index_instance = self.evaluate_lb_per_instance_baseline_k0_average(node_time_limit=node_time_limit,
                                                                        total_time_limit=total_time_limit,
                                                                       index_instance=index_instance,
                                                                       reset_k_at_2nditeration=reset_k_at_2nditeration
                                                                       )

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

    def primal_integral_k_prime_012(self, test_instance_size, total_time_limit=60, node_time_limit=30):

        # input:
        # k_prime: gnn_prime without merged

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'


        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'

        # k_prime trained by data without merge
        directory_lb_test_k_prime = directory +'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        # k_prime trained by data with merge
        directory_lb_test_k_prime_merged = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
        directory_lb_test_baseline = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'

        primal_int_baselines = []
        primal_int_preds = []
        primal_int_preds_reset = []
        primal_gap_final_baselines = []
        primal_gap_final_preds = []
        primal_gap_final_preds_reset = []
        steplines_baseline = []
        steplines_pred = []
        steplines_pred_reset = []

        primal_int_regression_k_primes = []
        primal_gap_final_regression_k_primes = []
        steplines_regression_k_primes = []

        primal_int_regression_k_primes_merged = []
        primal_gap_final_regression_k_primes_merged = []
        steplines_regression_k_primes_merged = []

        index_instance = 160
        index_max = 200
        if self.instance_type == 'combinatorialauction' and test_instance_size == '-large':
            index_instance = 0
            index_max = 40


        for i in range(index_instance, index_max):  # # if not (i == 161 or i == 170):

            instance_name = self.instance_type + '-' + str(i) + '_transformed'  # instance 100-199


            # test from k_prime
            filename = f'{directory_lb_test_k_prime}lb-test-{instance_name}.pkl'
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_k_prime, times_k_prime = data  # objs contains objs of a single instance of a lb test

            # test from k_prime_merged
            filename = f'{directory_lb_test_k_prime_merged}lb-test-{instance_name}.pkl'
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_k_prime_merged, times_k_prime_merged = data  # objs contains objs of a single instance of a lb test

            # test from baseline
            filename = f'{directory_lb_test_baseline}lb-test-{instance_name}.pkl'
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs, times = data  # objs contains objs of a single instance of a lb test

            filename = f'{directory_lb_test_k_prime_2}lb-test-{instance_name}.pkl'
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_k_prime_2, times_k_prime_2 = data  # objs contains objs of a single instance of a lb test

            filename = f'{directory_lb_test_k_prime_merged_2}lb-test-{instance_name}.pkl'
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_k_prime_merged_2, times_k_prime_merged_2 = data  # objs contains objs of a single instance of a lb test

            filename = f'{directory_lb_test_baseline_2}lb-test-{instance_name}.pkl'
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_2, times_k_2 = data  # objs contains objs of a single instance of a lb test


            objs = np.array(objs).reshape(-1)
            times = np.array(times).reshape(-1)

            objs_2 = np.array(objs_2).reshape(-1)

            objs_k_prime = np.array(objs_k_prime).reshape(-1)
            times_k_prime = np.array(times_k_prime).reshape(-1)

            objs_k_prime_2 = np.array(objs_k_prime_2).reshape(-1)

            objs_k_prime_merged = np.array(objs_k_prime_merged).reshape(-1)
            times_k_prime_merged = np.array(times_k_prime_merged).reshape(-1)

            objs_k_prime_merged_2 = np.array(objs_k_prime_merged_2).reshape(-1)

            a = [objs.min(), objs_2.min(), objs_k_prime.min(), objs_k_prime_2.min(), objs_k_prime_merged.min(), objs_k_prime_merged_2.min()]
            obj_opt = np.amin(a)



            # lb-regression-k-prime
            # if times_regression[-1] < total_time_limit:


            # baseline

            primal_int_baseline, primal_gap_final_baseline, stepline_baseline = self.compute_primal_integral(
                times=times, objs=objs, obj_opt=obj_opt, total_time_limit=total_time_limit)

            primal_gap_final_baselines.append(primal_gap_final_baseline)
            steplines_baseline.append(stepline_baseline)
            primal_int_baselines.append(primal_int_baseline)

            # regression_k_prime
            primal_int_regression_k_prime, primal_gap_final_regression_k_prime, stepline_regression_k_prime = self.compute_primal_integral(
                times=times_k_prime, objs=objs_k_prime, obj_opt=obj_opt, total_time_limit=total_time_limit)

            primal_gap_final_regression_k_primes.append(primal_gap_final_regression_k_prime)
            steplines_regression_k_primes.append(stepline_regression_k_prime)
            primal_int_regression_k_primes.append(primal_int_regression_k_prime)

            # regression_k_prime_merged
            primal_int_regression_k_prime_merged, primal_gap_final_regression_k_prime_merged, stepline_regression_k_prime_merged = self.compute_primal_integral(
                times=times_k_prime_merged, objs=objs_k_prime_merged, obj_opt=obj_opt, total_time_limit=total_time_limit)

            primal_gap_final_regression_k_primes_merged.append(primal_gap_final_regression_k_prime_merged)
            steplines_regression_k_primes_merged.append(stepline_regression_k_prime_merged)
            primal_int_regression_k_primes_merged.append(primal_int_regression_k_prime_merged)


        primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_preds = np.array(primal_int_preds).reshape(-1)
        primal_int_preds_reset = np.array(primal_int_preds_reset).reshape(-1)
        primal_int_regression_k_primes = np.array(primal_int_regression_k_primes).reshape(-1)
        primal_int_regression_k_primes_merged = np.array(primal_int_regression_k_primes_merged).reshape(-1)

        primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_preds = np.array(primal_gap_final_preds).reshape(-1)
        primal_gap_final_preds_reset = np.array(primal_gap_final_preds_reset).reshape(-1)
        primal_gap_final_regression_k_primes = np.array(primal_gap_final_regression_k_primes).reshape(-1)
        primal_gap_final_regression_k_primes_merged = np.array(primal_gap_final_regression_k_primes_merged).reshape(-1)

        # average primal integral over test dataset
        primal_int_base_ave = primal_int_baselines.sum() / len(primal_int_baselines)
        primal_int_pred_ave = primal_int_preds.sum() / len(primal_int_preds)
        primal_int_pred_ave_reset = primal_int_preds_reset.sum() / len(primal_int_preds_reset)
        primal_int_regression_k_prime_ave = primal_int_regression_k_primes.sum() / len(primal_int_regression_k_primes)
        primal_int_regression_k_prime_merged_ave = primal_int_regression_k_primes_merged.sum() / len(primal_int_regression_k_primes_merged)

        primal_gap_final_baselines_ave = primal_gap_final_baselines.sum() / len(primal_gap_final_baselines)
        primal_gap_final_preds = primal_gap_final_preds.sum() / len(primal_gap_final_preds)
        primal_gap_final_preds_reset = primal_gap_final_preds_reset.sum() / len(primal_gap_final_preds_reset)
        primal_gap_final_regression_k_primes_ave = primal_gap_final_regression_k_primes.sum() / len(primal_gap_final_regression_k_primes)
        primal_gap_final_regression_k_primes_merged_ave = primal_gap_final_regression_k_primes_merged.sum() / len(
            primal_gap_final_regression_k_primes_merged)

        print(self.instance_type + test_instance_size)
        print(self.incumbent_mode + 'Solution')
        print('baseline primal integral: ', primal_int_base_ave)
        print('k_regre_prime primal integral: ', primal_int_regression_k_prime_ave)
        print('k_regre_prime_merged primal integral: ', primal_int_regression_k_prime_merged_ave)
        print('\n')
        print('baseline primal gap: ', primal_gap_final_baselines_ave)
        print('k_regre_prime primal gap: ', primal_gap_final_regression_k_primes_ave)
        print('k_regre_prime_merged primal gap: ', primal_gap_final_regression_k_primes_merged_ave)

        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)
        primalgaps_baseline = None
        for n, stepline_baseline in enumerate(steplines_baseline):
            primal_gap = stepline_baseline(t)
            if n == 0:
                primalgaps_baseline = primal_gap
            else:
                primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        primalgap_baseline_ave = np.average(primalgaps_baseline, axis=0)


        primalgaps_regression_k_prime = None
        for n, stepline in enumerate(steplines_regression_k_primes):
            primal_gap = stepline(t)
            if n == 0:
                primalgaps_regression_k_prime = primal_gap
            else:
                primalgaps_regression_k_prime = np.vstack((primalgaps_regression_k_prime, primal_gap))
        primalgap_regression_k_prime_ave = np.average(primalgaps_regression_k_prime, axis=0)

        primalgaps_regression_k_prime_merged = None
        for n, stepline in enumerate(steplines_regression_k_primes_merged):
            primal_gap = stepline(t)
            if n == 0:
                primalgaps_regression_k_prime_merged = primal_gap
            else:
                primalgaps_regression_k_prime_merged = np.vstack((primalgaps_regression_k_prime_merged, primal_gap))
        primalgap_regression_k_prime_merged_ave = np.average(primalgaps_regression_k_prime_merged, axis=0)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle("Normalized primal gap")
        ax.set_title(self.instance_type + '-' + test_instance_size + '-' + self.incumbent_mode, loc='right')
        ax.plot(t, primalgap_baseline_ave, label='lb-baseline')
        ax.plot(t, primalgap_regression_k_prime_ave, label='lb-regression-k-prime-homo')
        ax.plot(t, primalgap_regression_k_prime_merged_ave, label='lb-regression-k-prime-merged')
        ax.set_xlabel('time /s')
        ax.set_ylabel("normalized primal gap")
        ax.legend()
        plt.show()

    def primal_integral_k_prime_3_sepa(self, test_instance_size, total_time_limit=60, node_time_limit=30):

        # input:
        # k_prime: gnn_prime without merged

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'


        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'

        # k_prime trained by data without merge
        directory_lb_test_k_prime = directory +'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        # k_prime trained by data with merge
        directory_lb_test_k_prime_merged = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
        directory_lb_test_baseline = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'

        primal_int_baselines = []
        primal_int_preds = []
        primal_int_preds_reset = []
        primal_gap_final_baselines = []
        primal_gap_final_preds = []
        primal_gap_final_preds_reset = []
        steplines_baseline = []
        steplines_pred = []
        steplines_pred_reset = []

        primal_int_regression_k_primes = []
        primal_gap_final_regression_k_primes = []
        steplines_regression_k_primes = []

        primal_int_regression_k_primes_merged = []
        primal_gap_final_regression_k_primes_merged = []
        steplines_regression_k_primes_merged = []

        for i in range(80, 115):
        # if not (i == 161 or i == 170):

            instance_name = self.instance_type + '-' + str(i) + '_transformed'  # instance 100-199


            # test from k_prime_merged
            filename = f'{directory_lb_test_k_prime_merged}lb-test-{instance_name}.pkl'
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_k_prime_merged, times_k_prime_merged = data  # objs contains objs of a single instance of a lb test

            # test from baseline
            filename = f'{directory_lb_test_baseline}lb-test-{instance_name}.pkl'
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs, times = data  # objs contains objs of a single instance of a lb test


            filename = f'{directory_lb_test_k_prime_merged_2}lb-test-{instance_name}.pkl'
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_k_prime_merged_2, times_k_prime_merged_2 = data  # objs contains objs of a single instance of a lb test

            filename = f'{directory_lb_test_baseline_2}lb-test-{instance_name}.pkl'
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_2, times_k_2 = data  # objs contains objs of a single instance of a lb test

            objs = np.array(objs).reshape(-1)
            times = np.array(times).reshape(-1)

            objs_2 = np.array(objs_2).reshape(-1)


            objs_k_prime_merged = np.array(objs_k_prime_merged).reshape(-1)
            times_k_prime_merged = np.array(times_k_prime_merged).reshape(-1)

            objs_k_prime_merged_2 = np.array(objs_k_prime_merged_2).reshape(-1)

            a = [objs.min(), objs_2.min(), objs_k_prime_merged.min(), objs_k_prime_merged_2.min()]
            obj_opt = np.amin(a)



            # lb-regression-k-prime
            # if times_regression[-1] < total_time_limit:


            # baseline

            primal_int_baseline, primal_gap_final_baseline, stepline_baseline = self.compute_primal_integral(
                times=times, objs=objs, obj_opt=obj_opt, total_time_limit=total_time_limit)

            primal_gap_final_baselines.append(primal_gap_final_baseline)
            steplines_baseline.append(stepline_baseline)
            primal_int_baselines.append(primal_int_baseline)


            # regression_k_prime_merged
            primal_int_regression_k_prime_merged, primal_gap_final_regression_k_prime_merged, stepline_regression_k_prime_merged = self.compute_primal_integral(
                times=times_k_prime_merged, objs=objs_k_prime_merged, obj_opt=obj_opt, total_time_limit=total_time_limit)

            primal_gap_final_regression_k_primes_merged.append(primal_gap_final_regression_k_prime_merged)
            steplines_regression_k_primes_merged.append(stepline_regression_k_prime_merged)
            primal_int_regression_k_primes_merged.append(primal_int_regression_k_prime_merged)


        primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_preds = np.array(primal_int_preds).reshape(-1)
        primal_int_preds_reset = np.array(primal_int_preds_reset).reshape(-1)
        primal_int_regression_k_primes = np.array(primal_int_regression_k_primes).reshape(-1)
        primal_int_regression_k_primes_merged = np.array(primal_int_regression_k_primes_merged).reshape(-1)

        primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_preds = np.array(primal_gap_final_preds).reshape(-1)
        primal_gap_final_preds_reset = np.array(primal_gap_final_preds_reset).reshape(-1)
        primal_gap_final_regression_k_primes = np.array(primal_gap_final_regression_k_primes).reshape(-1)
        primal_gap_final_regression_k_primes_merged = np.array(primal_gap_final_regression_k_primes_merged).reshape(-1)

        # average primal integral over test dataset
        primal_int_base_ave = primal_int_baselines.sum() / len(primal_int_baselines)
        primal_int_pred_ave = primal_int_preds.sum() / len(primal_int_preds)
        primal_int_pred_ave_reset = primal_int_preds_reset.sum() / len(primal_int_preds_reset)
        primal_int_regression_k_prime_ave = primal_int_regression_k_primes.sum() / len(primal_int_regression_k_primes)
        primal_int_regression_k_prime_merged_ave = primal_int_regression_k_primes_merged.sum() / len(primal_int_regression_k_primes_merged)

        primal_gap_final_baselines_ave = primal_gap_final_baselines.sum() / len(primal_gap_final_baselines)
        primal_gap_final_preds = primal_gap_final_preds.sum() / len(primal_gap_final_preds)
        primal_gap_final_preds_reset = primal_gap_final_preds_reset.sum() / len(primal_gap_final_preds_reset)
        primal_gap_final_regression_k_primes_ave = primal_gap_final_regression_k_primes.sum() / len(primal_gap_final_regression_k_primes)
        primal_gap_final_regression_k_primes_merged_ave = primal_gap_final_regression_k_primes_merged.sum() / len(
            primal_gap_final_regression_k_primes_merged)

        print(self.instance_type + self.instance_size)
        print(self.incumbent_mode + 'Solution')
        print('baseline primal integral: ', primal_int_base_ave)
        print('k_regre_prime_merged primal integral: ', primal_int_regression_k_prime_merged_ave)
        print('\n')
        print('baseline primal gap: ', primal_gap_final_baselines_ave)
        print('k_regre_prime_merged primal gap: ', primal_gap_final_regression_k_primes_merged_ave)

        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)
        primalgaps_baseline = None
        for n, stepline_baseline in enumerate(steplines_baseline):
            primal_gap = stepline_baseline(t)
            if n == 0:
                primalgaps_baseline = primal_gap
            else:
                primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        primalgap_baseline_ave = np.average(primalgaps_baseline, axis=0)


        primalgaps_regression_k_prime_merged = None
        for n, stepline in enumerate(steplines_regression_k_primes_merged):
            primal_gap = stepline(t)
            if n == 0:
                primalgaps_regression_k_prime_merged = primal_gap
            else:
                primalgaps_regression_k_prime_merged = np.vstack((primalgaps_regression_k_prime_merged, primal_gap))
        primalgap_regression_k_prime_merged_ave = np.average(primalgaps_regression_k_prime_merged, axis=0)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle("Normalized primal gap")
        ax.set_title(self.instance_type + '-' + test_instance_size + '-' + self.incumbent_mode, loc='right')
        ax.plot(t, primalgap_baseline_ave, label='lb-baseline')
        ax.plot(t, primalgap_regression_k_prime_merged_ave, label='lb-regression-k-prime-merged')
        ax.set_xlabel('time /s')
        ax.set_ylabel("normalized primal gap")
        ax.legend()
        plt.show()

    def primal_integral_k_prime_miplib_bianry39(self, test_instance_size, total_time_limit=60, node_time_limit=30):

        # input:

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'


        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'

        # k_prime trained by data without merge
        directory_lb_test_k_prime = directory +'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        # k_prime trained by data with merge
        directory_lb_test_k_prime_merged = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
        directory_lb_test_baseline = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'

        primal_int_baselines = []
        primal_int_preds = []
        primal_int_preds_reset = []
        primal_gap_final_baselines = []
        primal_gap_final_preds = []
        primal_gap_final_preds_reset = []
        steplines_baseline = []
        steplines_pred = []
        steplines_pred_reset = []

        primal_int_regression_k_primes = []
        primal_gap_final_regression_k_primes = []
        steplines_regression_k_primes = []

        primal_int_regression_k_primes_merged = []
        primal_gap_final_regression_k_primes_merged = []
        steplines_regression_k_primes_merged = []

        for i in range(0, 30):
            if not (i == 18):

                instance_name = self.instance_type + '-' + str(i) + '_transformed'  # instance 100-199


                # test from k_prime_merged
                filename = f'{directory_lb_test_k_prime_merged}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_k_prime_merged, times_k_prime_merged = data  # objs contains objs of a single instance of a lb test

                # test from baseline
                filename = f'{directory_lb_test_baseline}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs, times = data  # objs contains objs of a single instance of a lb test


                filename = f'{directory_lb_test_k_prime_merged_2}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_k_prime_merged_2, times_k_prime_merged_2 = data  # objs contains objs of a single instance of a lb test

                filename = f'{directory_lb_test_baseline_2}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_2, times_k_2 = data  # objs contains objs of a single instance of a lb test

                objs = np.array(objs).reshape(-1)
                times = np.array(times).reshape(-1)

                objs_2 = np.array(objs_2).reshape(-1)


                objs_k_prime_merged = np.array(objs_k_prime_merged).reshape(-1)
                times_k_prime_merged = np.array(times_k_prime_merged).reshape(-1)

                objs_k_prime_merged_2 = np.array(objs_k_prime_merged_2).reshape(-1)

                a = [objs.min(), objs_2.min(), objs_k_prime_merged.min(), objs_k_prime_merged_2.min()]
                obj_opt = np.amin(a)



                # lb-regression-k-prime
                # if times_regression[-1] < total_time_limit:

                # baseline
                primal_int_baseline, primal_gap_final_baseline, stepline_baseline = self.compute_primal_integral(
                    times=times, objs=objs, obj_opt=obj_opt, total_time_limit=total_time_limit)

                primal_gap_final_baselines.append(primal_gap_final_baseline)
                steplines_baseline.append(stepline_baseline)
                primal_int_baselines.append(primal_int_baseline)


                # regression_k_prime_merged
                primal_int_regression_k_prime_merged, primal_gap_final_regression_k_prime_merged, stepline_regression_k_prime_merged = self.compute_primal_integral(
                    times=times_k_prime_merged, objs=objs_k_prime_merged, obj_opt=obj_opt, total_time_limit=total_time_limit)

                primal_gap_final_regression_k_primes_merged.append(primal_gap_final_regression_k_prime_merged)
                steplines_regression_k_primes_merged.append(stepline_regression_k_prime_merged)
                primal_int_regression_k_primes_merged.append(primal_int_regression_k_prime_merged)


        primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_preds = np.array(primal_int_preds).reshape(-1)
        primal_int_preds_reset = np.array(primal_int_preds_reset).reshape(-1)
        primal_int_regression_k_primes = np.array(primal_int_regression_k_primes).reshape(-1)
        primal_int_regression_k_primes_merged = np.array(primal_int_regression_k_primes_merged).reshape(-1)

        primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_preds = np.array(primal_gap_final_preds).reshape(-1)
        primal_gap_final_preds_reset = np.array(primal_gap_final_preds_reset).reshape(-1)
        primal_gap_final_regression_k_primes = np.array(primal_gap_final_regression_k_primes).reshape(-1)
        primal_gap_final_regression_k_primes_merged = np.array(primal_gap_final_regression_k_primes_merged).reshape(-1)

        # average primal integral over test dataset
        primal_int_base_ave = primal_int_baselines.sum() / len(primal_int_baselines)
        primal_int_pred_ave = primal_int_preds.sum() / len(primal_int_preds)
        primal_int_pred_ave_reset = primal_int_preds_reset.sum() / len(primal_int_preds_reset)
        primal_int_regression_k_prime_ave = primal_int_regression_k_primes.sum() / len(primal_int_regression_k_primes)
        primal_int_regression_k_prime_merged_ave = primal_int_regression_k_primes_merged.sum() / len(primal_int_regression_k_primes_merged)

        primal_gap_final_baselines_ave = primal_gap_final_baselines.sum() / len(primal_gap_final_baselines)
        primal_gap_final_preds = primal_gap_final_preds.sum() / len(primal_gap_final_preds)
        primal_gap_final_preds_reset = primal_gap_final_preds_reset.sum() / len(primal_gap_final_preds_reset)
        primal_gap_final_regression_k_primes_ave = primal_gap_final_regression_k_primes.sum() / len(primal_gap_final_regression_k_primes)
        primal_gap_final_regression_k_primes_merged_ave = primal_gap_final_regression_k_primes_merged.sum() / len(
            primal_gap_final_regression_k_primes_merged)

        print(self.instance_type + self.instance_size)
        print(self.incumbent_mode + 'Solution')
        print('baseline primal integral: ', primal_int_base_ave)
        print('k_regre_prime_merged primal integral: ', primal_int_regression_k_prime_merged_ave)
        print('\n')
        print('baseline primal gap: ', primal_gap_final_baselines_ave)
        print('k_regre_prime primal gap: ', primal_gap_final_regression_k_primes_merged_ave)

        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)
        primalgaps_baseline = None
        for n, stepline_baseline in enumerate(steplines_baseline):
            primal_gap = stepline_baseline(t)
            if n == 0:
                primalgaps_baseline = primal_gap
            else:
                primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        primalgap_baseline_ave = np.average(primalgaps_baseline, axis=0)


        primalgaps_regression_k_prime_merged = None
        for n, stepline in enumerate(steplines_regression_k_primes_merged):
            primal_gap = stepline(t)
            if n == 0:
                primalgaps_regression_k_prime_merged = primal_gap
            else:
                primalgaps_regression_k_prime_merged = np.vstack((primalgaps_regression_k_prime_merged, primal_gap))
        primalgap_regression_k_prime_merged_ave = np.average(primalgaps_regression_k_prime_merged, axis=0)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle("Normalized primal gap")
        ax.set_title(self.instance_type + '-' + test_instance_size + '-' + self.incumbent_mode, loc='right')
        ax.plot(t, primalgap_baseline_ave, label='lb-baseline')
        ax.plot(t, primalgap_regression_k_prime_merged_ave, label='lb-regression-k-prime-merged')
        ax.set_xlabel('time /s')
        ax.set_ylabel("normalized primal gap")
        ax.legend()
        plt.show()


class RlLocalbranch(MlLocalbranch):
    """Training and evaluation of the RL (REINFORCE) policies for local branching.

    Provides the training loops for the k-policy (train_agent_policy_k) and
    the t-policy (train_agent_policy_t), the evaluation of the RL-guided LB
    heuristics lb-rl, lb-srmrl and lb-srmrl-adapt-t of Section 5
    (evaluate_localbranching_rlactive, evaluate_localbranching_rlactive_policy_kt),
    and the primal-integral post-processing that prints and plots the
    results (primal_integral, primal_integral_03, primal_gap_integral_hybrid_03).
    """

    def __init__(self, instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed=100, enable_gpu=False):
        super().__init__(instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed, enable_gpu)
        self.alpha = 0.01
        self.gamma = 0.99
        self.eps = np.finfo(np.float32).eps.item()

    def load_mip_dataset(self, instances_directory=None, sols_directory=None, incumbent_mode=None):
        instance_filename = f'{self.instance_type}-*_transformed.cip'
        sol_filename = f'{incumbent_mode}-{self.instance_type}-*_transformed.sol'

        train_instances_directory = instances_directory + 'train/'
        instance_files = [str(path) for path in sorted(pathlib.Path(train_instances_directory).glob(instance_filename), key=lambda path: int(path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        if self.instance_type == instancetypes[3]:
            instance_train_files = instance_files[:int(4 / 80 * len(instance_files))]
        elif self.instance_type == instancetypes[4]:
            instance_train_files = instance_files[:int(5 / 29 * len(instance_files))]
        else:
            instance_train_files = instance_files[:int(1/80 * len(instance_files))]
        instance_valid_files = instance_files[int(7/8 * len(instance_files)):]

        test_instances_directory = instances_directory + 'test/'
        instance_test_files = [str(path) for path in sorted(pathlib.Path(test_instances_directory).glob(instance_filename),
                                                       key=lambda path: int(
                                                           path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        train_sols_directory = sols_directory + 'train/'
        sol_files = [str(path) for path in sorted(pathlib.Path(train_sols_directory).glob(sol_filename), key=lambda path: int(path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        if self.instance_type == instancetypes[3]:
            sol_train_files = sol_files[:int(4 / 80 * len(sol_files))]
        elif self.instance_type == instancetypes[4]:
            sol_train_files = sol_files[:int(5 / 29 * len(sol_files))]
        else:
            sol_train_files = sol_files[:int(1/80 * len(sol_files))]
        sol_valid_files = sol_files[int(7/8 * len(sol_files)):]

        test_sols_directory = sols_directory + 'test/'
        sol_test_files = [str(path) for path in sorted(pathlib.Path(test_sols_directory).glob(sol_filename),
                                                  key=lambda path: int(path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        train_dataset = InstanceDataset_2(mip_files=instance_train_files, sol_files=sol_train_files)
        valid_dataset = InstanceDataset_2(mip_files=instance_valid_files, sol_files=sol_valid_files)
        test_dataset = InstanceDataset_2(mip_files=instance_test_files, sol_files=sol_test_files)

        return train_dataset, valid_dataset, test_dataset

    def mdp_localbranch(self, localbranch=None, is_symmetric=True, reset_k_at_2nditeration=False, agent_k=None, optimizer_k=None, agent_t=None, optimizer_t=None, device=None, enable_adapt_t=False, t_reward_type=t_reward_types[0]):

        localbranch.total_time_available = localbranch.total_time_limit
        localbranch.first = False
        localbranch.diversify = False
        localbranch.t_node = localbranch.default_node_time_limit
        localbranch.div = 0
        localbranch.is_symmetric = is_symmetric
        localbranch.reset_k_at_2nditeration = reset_k_at_2nditeration
        lb_bits = 0
        t_list = []
        obj_list = []
        lb_bits_list = []
        k_list = []

        lb_bits_list.append(lb_bits)
        t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
        obj_list.append(localbranch.MIP_obj_best)
        k_list.append(localbranch.k)

        k_action = localbranch.actions['unchange']
        t_action = localbranch.actions['unchange']

        # initialize the env to state_0
        lb_bits += 1
        state, reward_k, reward_time , done, _ = localbranch.step_localbranch(k_action=k_action, t_action=t_action, lb_bits=lb_bits)
        localbranch.MIP_obj_init = localbranch.MIP_obj_best
        lb_bits_list.append(lb_bits)
        t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
        obj_list.append(localbranch.MIP_obj_best)
        k_list.append(localbranch.k)


        if (not done) and reset_k_at_2nditeration:
            lb_bits += 1
            localbranch.default_k = 20
            if not localbranch.is_symmetric:
                localbranch.default_k = 10
            localbranch.k = localbranch.default_k
            localbranch.diversify = False
            localbranch.first = False

            state, reward_k, reward_time, done, _ = localbranch.step_localbranch(k_action=k_action, t_action=t_action,
                                                                   lb_bits=lb_bits)

            localbranch.MIP_obj_init = localbranch.MIP_obj_best
            lb_bits_list.append(lb_bits)
            t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
            obj_list.append(localbranch.MIP_obj_best)
            k_list.append(localbranch.k)

        while not done:  # and localbranch.div < localbranch.div_max
            lb_bits += 1

            k_vanilla, t_action = localbranch.policy_vanilla(state)


            k_action = k_vanilla
            if agent_k is not None:
                k_action = agent_k.select_action(state)

            if agent_t is not None:
                t_action = agent_t.select_action(state)


            # execute one iteration of LB, get the state and rewards

            state, reward_k, reward_time, done, _ = localbranch.step_localbranch(k_action=k_action, t_action=t_action, lb_bits=lb_bits, enable_adapt_t=enable_adapt_t)

            if agent_k is not None:
                agent_k.rewards.append(reward_k)
            if agent_t is not None:
                if t_reward_type == t_reward_types[1]:
                    reward_t = reward_k + reward_time
                elif t_reward_type == t_reward_types[0]:
                    reward_t = reward_k
                elif t_reward_type == t_reward_types[2]:
                    reward_t = reward_time

                agent_t.rewards.append(reward_t)

            lb_bits_list.append(lb_bits)
            t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
            obj_list.append(localbranch.MIP_obj_best)
            k_list.append(localbranch.k)

        print(
            'K_final: {:.0f}'.format(localbranch.k),
            'div_final: {:.0f}'.format(localbranch.div)
        )

        print('try to solve right branch')
        localbranch.solve_rightbranch()
        print('right branch is solved/arriving at time limit.')
        t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
        obj_list.append(localbranch.MIP_obj_best)
        k_list.append(localbranch.k)

        status = localbranch.MIP_model.getStatus()

        elapsed_time = localbranch.total_time_limit - localbranch.total_time_available

        lb_bits_list = np.array(lb_bits_list).reshape(-1)
        times_list = np.array(t_list).reshape(-1)
        objs_list = np.array(obj_list).reshape(-1)
        k_list = np.array(k_list).reshape(-1)


        del localbranch.subMIP_sol_best
        del localbranch.MIP_sol_bar
        del localbranch.MIP_sol_best


        return status, localbranch.MIP_obj_best, elapsed_time, lb_bits_list, times_list, objs_list, agent_k, agent_t

    def train_rl_policy_per_instance(self, MIP_model, incumbent_solution, node_time_limit, total_time_limit, index_instance,
                                     reset_k_at_2nditeration=False, agent_k=None, optimizer_k=None, agent_t=None, optimizer_t=None,
                                     device=None, t_reward_type = t_reward_types[0], enable_adapt_t=False):
        """
        evaluate a single MIP instance by two algorithms: lb-baseline and lb-pred_k
        :param node_time_limit:
        :param total_time_limit:
        :param index_instance:
        :return:
        """
        gc.collect()


        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("Instance: ", MIP_model.getProbName())
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

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
        graph = graph.to(device)


        initial_obj = MIP_model.getSolObjVal(incumbent_solution)
        print("Initial obj before LB: {}".format(initial_obj))

        binary_supports = binary_support(MIP_model, incumbent_solution)
        print('binary support: ', binary_supports)

        del graph
        del observation

        # create a copy of MIP
        MIP_model.resetParams()

        MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
            problemName='GNN_reset',
            origcopy=False)

        print('MIP copies are created')

        MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent_solution,
                                                  MIP_copy_vars3)

        print('incumbent solution is copied to MIP copies')
        MIP_model.freeProb()
        del MIP_model
        del incumbent_solution

        # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
        lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=self.k_baseline, # k_pred
                                   node_time_limit=node_time_limit,
                                   total_time_limit=total_time_limit,
                                   is_symmetric=self.is_symmetric)
        status, obj_best, elapsed_time, lb_bits_pred_reset, times_pred_reset_, objs_pred_reset_, agent_k, agent_t = self.mdp_localbranch(
            localbranch=lb_model3,
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=reset_k_at_2nditeration,
            agent_k=agent_k,
            optimizer_k=optimizer_k,
            agent_t=agent_t,
            optimizer_t=optimizer_t,
            device=device,
            t_reward_type=t_reward_type,
            enable_adapt_t=enable_adapt_t
        )

        print("Instance:", MIP_model_copy3.getProbName())
        print("Status of LB: ", status)
        print("Best obj of LB: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        objs_pred_reset = lb_model3.primal_objs
        times_pred_reset = lb_model3.primal_times

        objs_pred_reset = np.array(objs_pred_reset).reshape(-1)
        times_pred_reset = np.array(times_pred_reset).reshape(-1)


        data = [objs_pred_reset, times_pred_reset]
        primal_integral, primal_gap_final, stepline = self.compute_primal_integral(times_pred_reset, objs_pred_reset, total_time_limit)

        del sol_MIP_copy3
        del MIP_model_copy3

        del objs_pred_reset
        del times_pred_reset

        del lb_model3
        del stepline

        index_instance += 1
        del instance
        return index_instance, agent_k, agent_t, primal_integral, primal_gap_final

    def update_agent(self, agent, optimizer):

        R = 0
        policy_losses = []
        returns = []
        # calculate the return
        for r in agent.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0,R)
        returns = torch.tensor(returns)
        returns = returns.to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        # calculate loss
        with torch.set_grad_enabled(optimizer is not None):
            for log_prob, Return in zip(agent.log_probs, returns):
                policy_losses.append(-log_prob * Return)

            # optimize policy network
            if (optimizer is not None) and (not len(policy_losses) == 0):
                optimizer.zero_grad()
                policy_losses = torch.cat(policy_losses).sum()
                policy_losses.backward()
                optimizer.step()

        del agent.rewards[:]
        del agent.log_probs[:]
        return agent, optimizer, R


    def train_agent_policy_k(self, train_instance_size='-small', train_incumbent_mode=incumbent_modes[0], total_time_limit=60, node_time_limit=10,
                             reset_k_at_2nditeration=False, lr=0.001, n_epochs=20, epsilon=0, use_checkpoint=False):

        train_instance_type = self.instance_type
        train_data = train_instance_type + train_instance_size
        direc = './data/generated_instances/' + train_instance_type + '/' + train_instance_size + '/'

        instances_directory = direc + 'transformedmodel' + '/'
        sols_directory = direc + 'firstsol' + '/'
        train_dataset_first, valid_dataset_first, test_dataset_first = self.load_mip_dataset(instances_directory=instances_directory, sols_directory=sols_directory, incumbent_mode='firstsol')
        sols_directory = direc + 'rootsol' + '/'
        train_dataset_root, valid_dataset_root, test_dataset_root = self.load_mip_dataset(instances_directory=instances_directory,
                                                                           sols_directory=sols_directory, incumbent_mode='rootsol')
        train_datasets = [train_dataset_first, train_dataset_root]
        firstroot_dataset = ConcatDataset(train_datasets)

        if train_incumbent_mode == incumbent_modes[0]:
            train_dataset = train_dataset_first
        elif train_incumbent_mode == incumbent_modes[1]:
            train_dataset = train_dataset_root
        else:
            train_dataset = firstroot_dataset

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, collate_fn=custom_collate)
        size_trainset = len(train_loader.dataset)
        print(size_trainset)

        device = self.device
        self.regression_dataset = train_instance_type + '-small'


        self.k_baseline = 20

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_baseline = self.k_baseline / 2
        total_time_limit = total_time_limit
        node_time_limit = node_time_limit

        self.saved_model_directory = './result/saved_models/'

        self.saved_rlmodels_k_policy_directory = self.saved_model_directory + 'rl/reinforce/k_policy/' + train_instance_type + '/' + 't_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's/' + 'seed' + str(self.seed) + '/'
        pathlib.Path(self.saved_rlmodels_k_policy_directory).mkdir(parents=True, exist_ok=True)

        train_directory = './result/generated_instances/' + self.instance_type + '/' + train_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        self.reinforce_train_directory = train_directory + 'rl/' + 'reinforce/train/k_policy/data/'
        pathlib.Path(self.reinforce_train_directory).mkdir(parents=True, exist_ok=True)

        rl_policy_k = SimplePolicy(7, 4)
        rl_policy_k.train()


        optim_k = torch.optim.Adam(rl_policy_k.parameters(), lr=lr)

        greedy_k = False
        rl_policy_k = rl_policy_k.to(device)
        agent_k = AgentReinforce(rl_policy_k, device, greedy_k, optim_k, epsilon)

        returns_k = []
        epochs = []
        primal_integrals = []
        primal_gaps = []
        data = None
        epochs_np = None
        returns_np = None
        primal_integrals_np = None
        primal_gaps_np = None
        epoch_init = 0
        epoch_start = epoch_init
        epoch_end = epoch_start+n_epochs+1

        if use_checkpoint:
            checkpoint = torch.load(

                self.saved_rlmodels_k_policy_directory + 'checkpoint_trained_reward3_simplepolicy_rl4lb_reinforce_trainset_' + train_instance_type + train_instance_size + '_0.1trainset_lr' + str(lr) + '_epochs' + str(45) + '.pth'

            )
            rl_policy_k.load_state_dict(checkpoint['model_state_dict'])
            optim_k.load_state_dict(checkpoint['optimizer_state_dict'])
            data = checkpoint['loss_data']
            epochs, returns_k, primal_integrals, primal_gaps = data
            rl_policy_k.train()

            epoch_start = checkpoint['epoch'] + 1
            epoch_end = epoch_start + n_epochs
            optimizer_k = optim_k

        for epoch in range(epoch_start,epoch_end):
            del data
            print(f"Epoch {epoch}")
            if epoch == epoch_init:
                optimizer_k = None
            elif epoch == epoch_init + 1:
                optimizer_k = optim_k

            index_instance = 0
            return_epoch = 0
            primal_integral_epoch = 0
            primal_gap_epoch = 0

            i = 0

            # while index_instance < size_trainset:
            for batch in (train_loader):

                print("instance: ", i)
                MIP_model = Model()
                print("create a new SCIP model")

                mip_file = batch['mipfile'][0]
                sol_file = batch['solfile'][0]

                MIP_model.readProblem(mip_file)

                incumbent_solution = MIP_model.readSolFile(sol_file)
                assert MIP_model.checkSol(
                    incumbent_solution), 'Warning: The initial incumbent of instance {} is not feasible!'.format(
                    MIP_model.getProbName())
                try:
                    MIP_model.addSol(incumbent_solution, False)
                    print('The initial incumbent of {} is successfully added to MIP model'.format(
                        MIP_model.getProbName()))
                except:
                    print('Error: the initial incumbent of {} is not successfully added to MIP model'.format(
                        MIP_model.getProbName()))

                # train_previous rl_policy
                index_instance, agent_k, _ , primal_integral, primal_gap_final = self.train_rl_policy_per_instance(MIP_model=MIP_model,
                                                                                                               incumbent_solution=incumbent_solution,
                                                                                                               node_time_limit=node_time_limit,
                                                                                                               total_time_limit=total_time_limit,
                                                                                                               index_instance=index_instance,
                                                                                                               reset_k_at_2nditeration=reset_k_at_2nditeration,
                                                                                                               agent_k=agent_k,
                                                                                                               optimizer_k=optimizer_k,
                                                                                                               device=device
                                                                                                               )

                agent_k, optimizer_k, R = self.update_agent(agent_k, optimizer_k)
                return_epoch += R

                primal_integral_epoch += primal_integral
                primal_gap_epoch += primal_gap_final

                i += 1

            return_epoch = return_epoch/size_trainset
            primal_integral_epoch = primal_integral_epoch/size_trainset
            primal_gap_epoch = primal_gap_epoch/size_trainset

            returns_k.append(return_epoch)
            epochs.append(epoch)
            primal_integrals.append(primal_integral_epoch)
            primal_gaps.append(primal_gap_epoch)

            print(f"Return: {return_epoch:0.6f}")
            print(f"Primal ingtegral: {primal_integral_epoch:0.6f}")

            data = [epochs, returns_k, primal_integrals, primal_gaps]

            if epoch > 0:
                filename = f'{self.reinforce_train_directory}lb-rl-checkpoint-reward3-simplepolicy-0.1trainset-lr{str(lr)}-epochs{str(epoch)}.pkl'  # instance 10% of testset

                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)

                # save checkpoint
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': rl_policy_k.state_dict(),
                            'optimizer_state_dict': optimizer_k.state_dict(),
                            'loss_data':data,
                            },

                    self.saved_rlmodels_k_policy_directory + 'checkpoint_trained_reward3_simplepolicy_rl4lb_reinforce_trainset_' + train_instance_type + train_instance_size + '_0.1trainset_lr' + str(lr) + '_epochs' + str(epoch) + '.pth'

                           )

                epochs_np = np.array(epochs).reshape(-1)
                returns_np = np.array(returns_k).reshape(-1)
                primal_integrals_np = np.array(primal_integrals).reshape(-1)
                primal_gaps_np = np.array(primal_gaps).reshape(-1)

                plt.close('all')
                plt.clf()
                fig, ax = plt.subplots(3, 1, figsize=(8, 6.4))
                fig.suptitle(train_data)
                fig.subplots_adjust()
                ax[0].set_title('lr= ' + str(lr) + ', epsilon=' + str(epsilon) + ', t_limit=' + str(total_time_limit),
                                loc='right')
                ax[0].plot(epochs_np, returns_np, label='loss')
                ax[0].set_xlabel('epoch')
                ax[0].set_ylabel("return t")

                ax[1].plot(epochs_np, primal_integrals_np, label='primal ingegral')
                ax[1].set_xlabel('epoch')
                ax[1].set_ylabel("primal integral")
                ax[1].legend()

                ax[2].plot(epochs_np, primal_gaps_np, label='primal gap')
                ax[2].set_xlabel('epoch')
                ax[2].set_ylabel("primal gap")
                ax[2].legend()


                plt.savefig(
                    './result/plots/plot_train_rl_reinforce_train_k_policy_' + train_instance_type + '_' + train_instance_size + '_' + train_incumbent_mode + '_'  + 'total_timelimit' + str(
                        total_time_limit) + 's' + '_lr ' + str(lr) + '.png')

                plt.show()


    def train_agent_policy_t(self, train_instance_size='-small', train_incumbent_mode=incumbent_modes[0], total_time_limit=60, node_time_limit=10,
                             reset_k_at_2nditeration=False, lr_k=0.01, lr_t=0.01, n_epochs=20, epsilon=0, use_checkpoint=False, rl_k_policy_path ='', t_reward_type = t_reward_types[0], enable_adapt_t=False):

        train_instance_type = self.instance_type
        train_data =  train_instance_type + train_instance_size
        direc = './data/generated_instances/' + train_instance_type + '/' + train_instance_size + '/'

        instances_directory = direc + 'transformedmodel' + '/'
        sols_directory = direc + 'firstsol' + '/'
        train_dataset_first, valid_dataset_first, test_dataset_first = self.load_mip_dataset(instances_directory=instances_directory, sols_directory=sols_directory, incumbent_mode='firstsol')
        sols_directory = direc + 'rootsol' + '/'
        train_dataset_root, valid_dataset_root, test_dataset_root = self.load_mip_dataset(instances_directory=instances_directory,
                                                                           sols_directory=sols_directory, incumbent_mode='rootsol')
        train_datasets = [train_dataset_first, train_dataset_root]
        firstroot_dataset = ConcatDataset(train_datasets)

        if train_incumbent_mode == incumbent_modes[0]:
            train_dataset = train_dataset_first
        elif train_incumbent_mode == incumbent_modes[1]:
            train_dataset = train_dataset_root
        else:
            train_dataset = firstroot_dataset

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, collate_fn=custom_collate)
        size_trainset = len(train_loader.dataset)
        print(size_trainset)

        device = self.device


        self.k_baseline = 20

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_baseline = self.k_baseline / 2
        total_time_limit = total_time_limit
        node_time_limit = node_time_limit

        self.saved_model_directory = './result/saved_models/'

        self.saved_rlmodels_k_policy_directory = self.saved_model_directory + 'rl/reinforce/k_policy/' + train_instance_type + '/' + 't_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's/'
        pathlib.Path( self.saved_rlmodels_k_policy_directory).mkdir(parents=True, exist_ok=True)

        self.saved_rlmodels_t_policy_directory = self.saved_model_directory + 'rl/reinforce/t_policy/' + train_instance_type + '/' + 't_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's/'
        pathlib.Path(self.saved_rlmodels_t_policy_directory).mkdir(parents=True, exist_ok=True)

        train_directory = './result/generated_instances/' + train_instance_type + '/' + train_instance_size + '/' + self.lbconstraint_mode + '/' + train_incumbent_mode + '/'
        self.reinforce_train_t_policy_directory = train_directory + 'rl/' + 'reinforce/train/t_policy/data/' + 't_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's/'
        pathlib.Path(self.reinforce_train_t_policy_directory).mkdir(parents=True, exist_ok=True)

        rl_policy_k = SimplePolicy(7, 4)
        checkpoint = torch.load(
            rl_k_policy_path)
        rl_policy_k.load_state_dict(checkpoint['model_state_dict'])
        rl_policy_k.eval()

        rl_policy_t = SimplePolicy(7, 4)
        rl_policy_t.train()

        optim_k = None
        optim_t = torch.optim.Adam(rl_policy_t.parameters(), lr=lr_t)

        greedy_k = False
        greedy_t  = False
        rl_policy_k = rl_policy_k.to(device)
        rl_policy_t = rl_policy_t.to(device)
        agent_k = AgentReinforce(rl_policy_k, device, greedy_k, optim_k, epsilon)
        agent_t = AgentReinforce(rl_policy_t, device, greedy_t, optim_t, epsilon)

        returns_k = []
        returns_t = []
        epochs = []
        primal_integrals = []
        primal_gaps = []
        data = None
        epochs_np = None
        returns_np = None
        primal_integrals_np = None
        primal_gaps_np = None
        epoch_init = 0
        epoch_start = epoch_init
        epoch_end = epoch_start+n_epochs+1

        if use_checkpoint:
            checkpoint = torch.load(

                self.saved_rlmodels_t_policy_directory + 'checkpoint_trained_reward_t_simplepolicy_rl4lb_reinforce_trainset_' + train_instance_type + train_instance_size + '_0.1trainset_lr' + str(lr_t) + '_epochs' + str(45) + '.pth'

            )
            rl_policy_t.load_state_dict(checkpoint['model_state_dict'])
            optim_t.load_state_dict(checkpoint['optimizer_state_dict'])
            data = checkpoint['loss_data']
            epochs, returns_k, returns_t, primal_integrals, primal_gaps = data
            rl_policy_t.train()

            epoch_start = checkpoint['epoch'] + 1
            epoch_end = epoch_start + n_epochs
            optimizer_t = optim_t

        for epoch in range(epoch_start,epoch_end):
            del data
            print(f"Epoch {epoch}")
            if epoch == epoch_init:
                optimizer_k = None
                optimizer_t = None
            elif epoch == epoch_init + 1:
                optimizer_k = optim_k
                optimizer_t = optim_t

            index_instance = 0
            return_epoch_k = 0
            return_epoch_t = 0
            primal_integral_epoch = 0
            primal_gap_epoch = 0

            # while index_instance < size_trainset:
            i = 0
            for batch in (train_loader):

                # option 2: only have the directory of MIP model and incumbent in the dataloader, load the MIP model here below:
                print("instance: ", i)
                MIP_model = Model()
                print("create a new SCIP model")

                mip_file = batch['mipfile'][0]
                sol_file = batch['solfile'][0]

                MIP_model.readProblem(mip_file)

                incumbent_solution = MIP_model.readSolFile(sol_file)
                assert MIP_model.checkSol(
                    incumbent_solution), 'Warning: The initial incumbent of instance {} is not feasible!'.format(
                    MIP_model.getProbName())
                try:
                    MIP_model.addSol(incumbent_solution, False)
                    print('The initial incumbent of {} is successfully added to MIP model'.format(
                        MIP_model.getProbName()))
                except:
                    print('Error: the initial incumbent of {} is not successfully added to MIP model'.format(
                        MIP_model.getProbName()))

                # train_previous rl_policy
                index_instance, agent_k, agent_t, primal_integral, primal_gap_final = self.train_rl_policy_per_instance(
                    MIP_model=MIP_model,
                    incumbent_solution=incumbent_solution,
                    node_time_limit=node_time_limit,
                    total_time_limit=total_time_limit,
                    index_instance=index_instance,
                    reset_k_at_2nditeration=reset_k_at_2nditeration,
                    agent_k=agent_k,
                    optimizer_k=optimizer_k,
                    agent_t=agent_t,
                    optimizer_t=optimizer_t,
                    device=device,
                    t_reward_type=t_reward_type,
                    enable_adapt_t=enable_adapt_t
                                                                                                               )


                agent_k, optimizer_k, R_k = self.update_agent(agent_k, optimizer_k)
                return_epoch_k += R_k

                agent_t, optimizer_t, R_t = self.update_agent(agent_t, optimizer_t)
                return_epoch_t += R_t

                primal_integral_epoch += primal_integral
                primal_gap_epoch += primal_gap_final

                i += 1

            return_epoch_k = return_epoch_k/size_trainset
            return_epoch_t = return_epoch_t / size_trainset
            primal_integral_epoch = primal_integral_epoch/size_trainset
            primal_gap_epoch = primal_gap_epoch/size_trainset

            returns_k.append(return_epoch_k)
            returns_t.append(return_epoch_t)
            epochs.append(epoch)
            primal_integrals.append(primal_integral_epoch)
            primal_gaps.append(primal_gap_epoch)

            print(f"Return policy k: {return_epoch_k:0.6f}")
            print(f"Return policy t: {return_epoch_t:0.6f}")
            print(f"Primal ingtegral: {primal_integral_epoch:0.6f}")

            data = [epochs, returns_k, returns_t, primal_integrals, primal_gaps]

            if epoch > 0:
                if enable_adapt_t:
                    filename = f'{self.reinforce_train_t_policy_directory}lb-rl-checkpoint-enable_vanilla_t_policy-t_policy-simplepolicy-{t_reward_type}-0.1trainset-lr{str(lr_t)}.pkl'
                else:
                    filename = f'{self.reinforce_train_t_policy_directory}lb-rl-checkpoint-t_policy-simplepolicy-{t_reward_type}-0.1trainset-lr{str(lr_t)}.pkl'  # instance 10% of testset

                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)

                # save checkpoint
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': rl_policy_t.state_dict(),
                            'optimizer_state_dict': optimizer_t.state_dict(),
                            'loss_data':data,
                            },

                    self.saved_rlmodels_t_policy_directory + 'checkpoint_rl4lb_trained_-t_policy-simplepolicy-' + t_reward_type + '_reinforce_0.1trainset_' + train_instance_type + train_instance_size + '_' + train_incumbent_mode + '_total_timelimit' + str(total_time_limit) + 's' + '_lr' + str(lr_t) +  '.pth' # + '_epochs' + str(epoch) +

                           )

            if epoch % 5 ==0:
                epochs_np = np.array(epochs).reshape(-1)
                returns_np = np.array(returns_t).reshape(-1)
                primal_integrals_np = np.array(primal_integrals).reshape(-1)
                primal_gaps_np = np.array(primal_gaps).reshape(-1)

                plt.close('all')
                plt.clf()
                fig, ax = plt.subplots(3, 1, figsize=(8, 6.4))
                fig.suptitle(train_data)
                fig.subplots_adjust()
                ax[0].set_title('lr= ' + str(lr_t) + ', epsilon=' + str(epsilon) + ', t_limit=' + str(total_time_limit), loc='right')
                ax[0].plot(epochs_np, returns_np, label='loss')
                ax[0].set_xlabel('epoch')
                ax[0].set_ylabel("return t")

                ax[1].plot(epochs_np, primal_integrals_np, label='primal ingegral')
                ax[1].set_xlabel('epoch')
                ax[1].set_ylabel("primal integral")
                ax[1].legend()

                ax[2].plot(epochs_np, primal_gaps_np, label='primal gap')
                ax[2].set_xlabel('epoch')
                ax[2].set_ylabel("primal gap")
                ax[2].legend()
                if enable_adapt_t:
                    plt.savefig('./result/plots/plot_train_rl_reinforce_enable_vanilla_t_train_t_policy_' + train_instance_type + '_' + train_instance_size + '_' + train_incumbent_mode + '_' + t_reward_type + 'total_timelimit' + str(total_time_limit) + 's' + '_lr ' + str(lr_t) + '_treward_case4r-1.png')
                else:
                    plt.savefig('./result/plots/plot_train_rl_reinforce_train_t_policy_' + train_instance_type + '_' + train_instance_size + '_' + train_incumbent_mode + '_' + t_reward_type + 'total_timelimit' + str(total_time_limit) + 's' + '_lr ' + str(lr_t) + '_treward_case4r-1.png')

                plt.show()

    def evaluate_lb_per_instance(self, node_time_limit, total_time_limit, index_instance, reset_k_at_2nditeration=False, agent=None,
                             ):
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

        incumbent_solution = MIP_model.readSolFile(firstsol_filename)

        feas = MIP_model.checkSol(incumbent_solution)
        try:
            MIP_model.addSol(incumbent_solution, False)
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

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = observation.constraint_features.shape[0] + \
                          observation.variable_features.shape[
                              0]
        graph = graph.to(device)


        initial_obj = MIP_model.getSolObjVal(incumbent_solution)
        print("Initial obj before LB: {}".format(initial_obj))

        binary_supports = binary_support(MIP_model, incumbent_solution)
        print('binary support: ', binary_supports)


        k_model = self.regression_model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
                            graph.variable_features)

        k_pred = k_model.item() * n_binvars
        print('GNN prediction: ', k_model.item())

        if self.is_symmetric == False:
            k_pred = k_model.item() * binary_supports

        k_pred = np.ceil(k_pred)

        del k_model
        del graph
        del observation

        # create a copy of MIP
        MIP_model.resetParams()
        MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
            problemName='noregression-rl',
            origcopy=False)
        MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
            problemName='regression-rl',
            origcopy=False)

        print('MIP copies are created')

        MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent_solution,
                                                  MIP_copy_vars2)
        MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent_solution,
                                                  MIP_copy_vars3)

        print('incumbent solution is copied to MIP copies')
        MIP_model.freeProb()
        del MIP_model
        del incumbent_solution


        # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
        lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=k_pred,
                                   node_time_limit=node_time_limit,
                                   total_time_limit=total_time_limit)
        status, obj_best, elapsed_time, lb_bits_pred_reset, times__regression_reinforce, objs_regression_reinforce, loss_instance, accu_instance = lb_model3.mdp_localbranch(
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=reset_k_at_2nditeration,
            policy=agent,
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

        # execute local branching with 1. first k predicted by GNN; 2. for 2nd iteration of lb, continue lb algorithm with no further injection
        lb_model2 = LocalBranching(MIP_model=MIP_model_copy2, MIP_sol_bar=sol_MIP_copy2, k=self.k_baseline,
                                   node_time_limit=node_time_limit,
                                   total_time_limit=total_time_limit)
        status, obj_best, elapsed_time, lb_bits_pred, times_noregression_reinforce, objs_noregression_reinforce, _, _ = lb_model2.mdp_localbranch(
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=False,
            policy=agent,
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

        data = [objs_noregression_reinforce, times_noregression_reinforce, objs_regression_reinforce, times__regression_reinforce]
        filename = f'{self.directory_lb_test}lb-test-{instance_name}.pkl'  # instance 100-199
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)

        del data
        del lb_model2
        del lb_model3

        index_instance += 1
        return index_instance

    def evaluate_localbranching(self, evaluation_instance_size='-small', total_time_limit=60, node_time_limit=30, reset_k_at_2nditeration=False, greedy=False):

        self.regression_dataset = self.instance_type + '-small'

        direc = './data/generated_instances/' + self.instance_type + '/' + evaluation_instance_size + '/'
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
            self.saved_gnn_directory + 'trained_params_mean_' + self.regression_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '.pth'))
        self.regression_model_gnn.to(self.device)

        evaluation_directory = './result/generated_instances/' + self.instance_type + '/' + evaluation_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/reinforce/'
        self.directory_lb_test = evaluation_directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + evaluation_instance_size + '/talored/'
        pathlib.Path(self.directory_lb_test).mkdir(parents=True, exist_ok=True)

        rl_policy = SimplePolicy(7, 4)

        self.saved_rlmodels_k_policy_directory = self.saved_gnn_directory + 'rl_noimitation/'
        checkpoint = torch.load(
            self.saved_rlmodels_k_policy_directory + 'checkpoint_noregression_noimitation_reward3_simplepolicy_rl4lb_reinforce_lr0.01_epsilon0.0_60s_talored4independentset-small-firstsol.pth'
        )
        rl_policy.load_state_dict(checkpoint['model_state_dict'])


        rl_policy.eval()

        greedy = greedy
        rl_policy = rl_policy.to(self.device)
        agent = AgentReinforce(rl_policy, self.device, greedy, None, 0.0)

        index_instance = 100
        while index_instance < 200:
            index_instance = self.evaluate_lb_per_instance(node_time_limit=node_time_limit, total_time_limit=total_time_limit, index_instance=index_instance, reset_k_at_2nditeration=reset_k_at_2nditeration,
                                                           agent=agent
                                                           )

    def evaluate_lb_per_instance_rlactive(self, MIP_model, incumbent, node_time_limit, total_time_limit, reset_k_at_2nditeration=False,
                                 agent1=None, agent2=None, enable_adapt_t=False
                                 ):
        """
        evaluate a single MIP instance by two algorithms: lb-baseline and lb-pred_k
        :param node_time_limit:
        :param total_time_limit:
        :param index_instance:
        :return:
        """
        incumbent_solution = incumbent

        device = self.device
        gc.collect()


        instance_name = MIP_model.getProbName()
        print(instance_name)
        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

        feas = MIP_model.checkSol(incumbent_solution)
        if not feas:
            print('Error: the initial solution of ' + instance_name + ' is not feasible!')
        else:
            print('The initial solution of ' + instance_name + ' is feasible!')

        try:
            MIP_model.addSol(incumbent_solution, False)
        except:
            print('Error: the root solution of ' + instance_name + ' is not feasible!')

        MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
            problemName='noregression-rl',
            origcopy=False)
        MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
            problemName='regression-rl',
            origcopy=False)

        print('MIP copies are created')

        MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent_solution,
                                                  MIP_copy_vars2)
        MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent_solution,
                                                  MIP_copy_vars3)
        print('incumbent solution is copied to MIP copies')


        instance = ecole.scip.Model.from_pyscipopt(MIP_model)
        observation, _, _, done, _ = self.env.reset(instance)


        # variable features: only incumbent solution
        variable_features = observation.variable_features[:, -1:]
        device = self.device
        graph = BipartiteNodeData(observation.constraint_features,
                                    observation.edge_features.indices,
                                    observation.edge_features.values,
                                    variable_features,
                                    device=device)
        graph = graph.to(device)


        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = observation.constraint_features.shape[0] + \
                          observation.variable_features.shape[
                              0]


        # solve the root node and get the LP solution, compute k_prime
        k_prime = self.compute_k_prime(MIP_model, incumbent)

        initial_obj = MIP_model.getSolObjVal(incumbent_solution)
        print("Initial obj before LB: {}".format(initial_obj))

        binary_supports = binary_support(MIP_model, incumbent_solution)
        print('binary support: ', binary_supports)


        k_model = self.regression_model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
                                            graph.variable_features)

        k_pred = k_model.item() * k_prime
        print('GNN prediction: ', k_model.item())

        if self.is_symmetric == False:
            k_pred = k_model.item() * k_prime

        k_pred = max(k_pred, self.k_prime_ratio_baseline * k_prime)
        k_pred = max(k_pred, 10)

        k_pred = np.ceil(k_pred)

        del k_model
        del graph
        del observation


        MIP_model.freeProb()
        del MIP_model
        del incumbent_solution


        # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
        lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=k_pred,
                                   node_time_limit=node_time_limit,
                                   total_time_limit=total_time_limit)

        status, obj_best, elapsed_time, lb_bits_pred_reset, times_regression_reinforce_, objs_regression_reinforce_, agent1, _ = self.mdp_localbranch(
            localbranch=lb_model3,
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=reset_k_at_2nditeration,
            agent_k=agent1,
            optimizer_k=None,
            device=device,
            enable_adapt_t=enable_adapt_t)
        print("Instance:", MIP_model_copy3.getProbName())
        print("Status of LB: ", status)
        print("Best obj of LB: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        objs_regression_reinforce = np.array(lb_model3.primal_objs).reshape(-1)
        times_regression_reinforce = np.array(lb_model3.primal_times).reshape(-1)

        MIP_model_copy3.freeProb()
        del sol_MIP_copy3
        del MIP_model_copy3

        # execute local branching with 1. first k predicted by GNN; 2. for 2nd iteration of lb, continue lb algorithm with no further injection
        lb_model2 = LocalBranching(MIP_model=MIP_model_copy2, MIP_sol_bar=sol_MIP_copy2, k=self.k_baseline,
                                   node_time_limit=node_time_limit,
                                   total_time_limit=total_time_limit)

        status, obj_best, elapsed_time, lb_bits_pred, times_noregression_reinforce_, objs_noregression_reinforce_, agent2, _ = self.mdp_localbranch(
            localbranch=lb_model2,
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=reset_k_at_2nditeration, #False
            agent_k=agent2,
            optimizer_k=None,
            device=device,
            enable_adapt_t=enable_adapt_t
        )

        objs_noregression_reinforce = np.array(lb_model2.primal_objs).reshape(-1)
        times_noregression_reinforce = np.array(lb_model2.primal_times).reshape(-1)


        print("Instance:", MIP_model_copy2.getProbName())
        print("Status of LB: ", status)
        print("Best obj of LB: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        MIP_model_copy2.freeProb()
        del sol_MIP_copy2
        del MIP_model_copy2

        data = [objs_noregression_reinforce, times_noregression_reinforce, objs_regression_reinforce, times_regression_reinforce]
        filename = f'{self.directory_lb_test}lb-test-{instance_name}.pkl'  # instance 100-199
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)

        del data
        del lb_model2
        del lb_model3

        return agent1, agent2

    def evaluate_localbranching_rlactive(self, evaluation_instance_size='-small', total_time_limit=60, node_time_limit=30,
                                reset_k_at_2nditeration=False, greedy=False, lr=None, regression_model_path='',
                    rl_model_path='', enable_adapt_t=False):

        self.regression_dataset = self.instance_type + '-small'

        direc = './data/generated_instances/' + self.instance_type + '/' + evaluation_instance_size + '/'
        directory_transformedmodel = direc + 'transformedmodel' + '/'
        directory_sol = direc + self.incumbent_mode + '/'

        incumbent_mode = self.incumbent_mode
        test_dataset = self.load_test_mip_dataset(directory_transformedmodel, directory_sol, incumbent_mode)

        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=custom_collate)

        self.k_baseline = 20

        if self.instance_type in ['setcovering', 'independentset', 'combinatorialauction']:
            k0_ratio_average = k_0_bank[self.instance_type+'-'+self.incumbent_mode]
            self.k_prime_ratio_baseline = k0_ratio_average

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_baseline = self.k_baseline / 2
        total_time_limit = total_time_limit
        node_time_limit = node_time_limit

        self.saved_model_directory = './result/saved_models/'
        self.regression_model_gnn = GNNPolicy()
        self.regression_model_gnn.load_state_dict(torch.load(
            regression_model_path))
        self.regression_model_gnn.to(self.device)

        evaluation_directory = './result/generated_instances/' + self.instance_type + '/' + evaluation_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/reinforce/test/old_models/'
        if enable_adapt_t:
            self.directory_lb_test = evaluation_directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(
            total_time_limit) + 's' + evaluation_instance_size + '/rlactive_t_node_baseline/seed'+ str(self.seed) + '/'
        else:
            self.directory_lb_test = evaluation_directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(
                total_time_limit) + 's' + evaluation_instance_size + '/rlactive/seed' + str(self.seed) + '/'
        pathlib.Path(self.directory_lb_test).mkdir(parents=True, exist_ok=True)

        rl_policy1 = SimplePolicy(7, 4)
        rl_policy2 = SimplePolicy(7, 4)


        checkpoint = torch.load(
            rl_model_path
        )
        rl_policy1.load_state_dict(checkpoint['model_state_dict'])
        rl_policy2.load_state_dict(checkpoint['model_state_dict'])


        rl_policy1.train()
        rl_policy2.train()

        optim1 = torch.optim.Adam(rl_policy1.parameters(), lr=lr)
        optim2 = torch.optim.Adam(rl_policy2.parameters(), lr=lr)

        optim1.load_state_dict(checkpoint['optimizer_state_dict'])
        optim2.load_state_dict(checkpoint['optimizer_state_dict'])

        # Move optimizer state to device to prevent device mismatch during step()
        for state in optim1.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        for state in optim2.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

        greedy = greedy
        rl_policy1 = rl_policy1.to(self.device)
        rl_policy2 = rl_policy2.to(self.device)
        agent1 = AgentReinforce(rl_policy1, self.device, greedy, optim1, 0.0)
        agent2 = AgentReinforce(rl_policy2, self.device, greedy, optim2, 0.0)

        for batch in (test_loader):
            MIP_model = batch['mip_model'][0]
            incumbent_solution = batch['incumbent_solution'][0]
            agent1, agent2 = self.evaluate_lb_per_instance_rlactive(
                MIP_model=MIP_model,
                incumbent=incumbent_solution,
                node_time_limit=node_time_limit,
                total_time_limit=total_time_limit,
                reset_k_at_2nditeration=reset_k_at_2nditeration,
                agent1=agent1,
                agent2=agent2,
                enable_adapt_t=enable_adapt_t
                                                            )

            agent1, optim1, R = self.update_agent(agent1, optim1)
            agent2, optim2, R = self.update_agent(agent2, optim2)

    def evaluate_lb_per_instance_rlactive_policy_kt(self, MIP_model, incumbent, node_time_limit, total_time_limit, reset_k_at_2nditeration=False,
                                 agent1=None, agent2=None, agent_t_1=None, agent_t_2=None,  t_reward_type = t_reward_types[2], enable_adapt_t=False
                                 ):
        """
        evaluate a single MIP instance by two algorithms: lb-baseline and lb-pred_k
        :param node_time_limit:
        :param total_time_limit:
        :param index_instance:
        :return:
        """
        incumbent_solution = incumbent

        device = self.device
        gc.collect()


        instance_name = MIP_model.getProbName()
        print(instance_name)
        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

        feas = MIP_model.checkSol(incumbent_solution)

        if not feas:
            print('Error: the initial solution of ' + instance_name + ' is not feasible!')
        else:
            print('The initial solution of ' + instance_name + ' is feasible!')

        try:
            MIP_model.addSol(incumbent_solution, False)
        except:
            print('Error: the root solution of ' + instance_name + ' is not feasible!')

        # create a copy of MIP
        MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
            problemName='noregression-rl',
            origcopy=False)
        MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
            problemName='regression-rl',
            origcopy=False)

        print('MIP copies are created')

        MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent_solution,
                                                  MIP_copy_vars2)
        MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent_solution,
                                                  MIP_copy_vars3)
        print('incumbent solution is copied to MIP copies')


        instance = ecole.scip.Model.from_pyscipopt(MIP_model)
        observation, _, _, done, _ = self.env.reset(instance)

        # variable features: only incumbent solution
        variable_features = observation.variable_features[:, -1:]
        device = self.device
        graph = BipartiteNodeData(observation.constraint_features,
                                  observation.edge_features.indices,
                                  observation.edge_features.values,
                                  variable_features,
                                  device=device)
        graph = graph.to(device)


        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = observation.constraint_features.shape[0] + \
                          observation.variable_features.shape[
                              0]


        # solve the root node and get the LP solution, compute k_prime
        k_prime = self.compute_k_prime(MIP_model, incumbent)

        initial_obj = MIP_model.getSolObjVal(incumbent_solution)
        print("Initial obj before LB: {}".format(initial_obj))

        binary_supports = binary_support(MIP_model, incumbent_solution)
        print('binary support: ', binary_supports)


        k_model = self.regression_model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
                                            graph.variable_features)

        k_pred = k_model.item() * k_prime
        print('GNN prediction: ', k_model.item())

        if self.is_symmetric == False:
            k_pred = k_model.item() * k_prime

        k_pred = np.ceil(k_pred)

        if k_pred < 10:
            k_pred = 10

        del k_model
        del graph
        del observation


        MIP_model.freeProb()
        del MIP_model
        del incumbent_solution


        # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
        lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=k_pred,
                                   node_time_limit=node_time_limit,
                                   total_time_limit=total_time_limit)

        status, obj_best, elapsed_time, lb_bits_pred_reset, times_regression_reinforce_, objs_regression_reinforce_, agent1, agent_t_1 = self.mdp_localbranch(
            localbranch=lb_model3,
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=reset_k_at_2nditeration,
            agent_k=agent1,
            optimizer_k=None,
            agent_t=agent_t_1,
            device=device,
            t_reward_type=t_reward_type,
            enable_adapt_t=enable_adapt_t)
        print("Instance:", MIP_model_copy3.getProbName())
        print("Status of LB: ", status)
        print("Best obj of LB: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        objs_regression_reinforce = np.array(lb_model3.primal_objs).reshape(-1)
        times_regression_reinforce = np.array(lb_model3.primal_times).reshape(-1)

        MIP_model_copy3.freeProb()
        del sol_MIP_copy3
        del MIP_model_copy3

        # execute local branching with 1. first k predicted by GNN; 2. for 2nd iteration of lb, continue lb algorithm with no further injection
        lb_model2 = LocalBranching(MIP_model=MIP_model_copy2, MIP_sol_bar=sol_MIP_copy2, k=self.k_baseline,
                                   node_time_limit=node_time_limit,
                                   total_time_limit=total_time_limit)

        status, obj_best, elapsed_time, lb_bits_pred, times_noregression_reinforce_, objs_noregression_reinforce_, agent2, agent_t_2= self.mdp_localbranch(
            localbranch=lb_model2,
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=reset_k_at_2nditeration, #False
            agent_k=agent2,
            optimizer_k=None,
            agent_t=agent_t_2,
            device=device,
            t_reward_type=t_reward_type,
            enable_adapt_t=enable_adapt_t
        )

        objs_noregression_reinforce = np.array(lb_model2.primal_objs).reshape(-1)
        times_noregression_reinforce = np.array(lb_model2.primal_times).reshape(-1)


        print("Instance:", MIP_model_copy2.getProbName())
        print("Status of LB: ", status)
        print("Best obj of LB: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        MIP_model_copy2.freeProb()
        del sol_MIP_copy2
        del MIP_model_copy2

        data = [objs_noregression_reinforce, times_noregression_reinforce, objs_regression_reinforce, times_regression_reinforce]
        filename = f'{self.directory_lb_test}lb-test-{instance_name}.pkl'  # instance 100-199
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)

        del data
        del lb_model2
        del lb_model3

        return agent1, agent2, agent_t_1, agent_t_2

    def evaluate_localbranching_rlactive_policy_kt(self, evaluation_instance_size='-small', total_time_limit=60, node_time_limit=30,
                                reset_k_at_2nditeration=False, greedy=False, lr=None, lr_t=None, regression_model_path='', rl_k_model_path='', rl_t_model_path='',  t_reward_type = t_reward_types[2], enable_adapt_t=False):

        self.regression_dataset = self.instance_type + '-small'

        direc = './data/generated_instances/' + self.instance_type + '/' + evaluation_instance_size + '/'
        directory_transformedmodel = direc + 'transformedmodel' + '/'
        directory_sol = direc + self.incumbent_mode + '/'

        incumbent_mode = self.incumbent_mode
        test_dataset = self.load_test_mip_dataset(directory_transformedmodel, directory_sol, incumbent_mode)

        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=custom_collate)

        self.k_baseline = 20

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_baseline = self.k_baseline / 2
        total_time_limit = total_time_limit
        node_time_limit = node_time_limit

        self.saved_model_directory = './result/saved_models/'
        self.regression_model_gnn = GNNPolicy()
        self.regression_model_gnn.load_state_dict(torch.load(
            regression_model_path))
        self.regression_model_gnn.to(self.device)

        evaluation_directory = './result/generated_instances/' + self.instance_type + '/' + evaluation_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/reinforce/test/old_models/'
        if enable_adapt_t:
            self.directory_lb_test = evaluation_directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(
            total_time_limit) + 's' + evaluation_instance_size + '/rlactive_t_node_baseline-rlpolicy-treward1/seed'+ str(self.seed) + '/' # rlactive_t_node_baseline-rlpolicy-treward1
        else:
            self.directory_lb_test = evaluation_directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(
                total_time_limit) + 's' + evaluation_instance_size + '/rlactive_t_node_rlpolicy-treward1/seed' + str(self.seed) + '/' # rlactive_t_node_rlpolicy-treward1
        pathlib.Path(self.directory_lb_test).mkdir(parents=True, exist_ok=True)

        print('The results are saved in:')
        print(self.directory_lb_test)

        rl_policy1 = SimplePolicy(7, 4)
        rl_policy2 = SimplePolicy(7, 4)

        rl_policy_t_1 = SimplePolicy(7, 4)
        rl_policy_t_2 = SimplePolicy(7, 4)


        checkpoint = torch.load(
            rl_k_model_path
        )
        rl_policy1.load_state_dict(checkpoint['model_state_dict'])
        rl_policy2.load_state_dict(checkpoint['model_state_dict'])

        checkpoint_t = torch.load(rl_t_model_path)
        rl_policy_t_1.load_state_dict(checkpoint_t['model_state_dict'])
        rl_policy_t_2.load_state_dict(checkpoint_t['model_state_dict'])


        rl_policy1.train()
        rl_policy2.train()
        rl_policy_t_1.eval() # .train()
        rl_policy_t_2.eval() # .train()

        optim1 = torch.optim.Adam(rl_policy1.parameters(), lr=lr)
        optim2 = torch.optim.Adam(rl_policy2.parameters(), lr=lr)
        optim_t_1 = None
        optim_t_2 = None

        optim1.load_state_dict(checkpoint['optimizer_state_dict'])
        optim2.load_state_dict(checkpoint['optimizer_state_dict'])

        # Move optimizer state to device to prevent device mismatch during step()
        for state in optim1.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        for state in optim2.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

        greedy = greedy
        rl_policy1 = rl_policy1.to(self.device)
        rl_policy2 = rl_policy2.to(self.device)
        rl_policy_t_1 = rl_policy_t_1.to(self.device)
        rl_policy_t_2 = rl_policy_t_2.to(self.device)

        agent1 = AgentReinforce(rl_policy1, self.device, greedy, optim1, 0.0)
        agent2 = AgentReinforce(rl_policy2, self.device, greedy, optim2, 0.0)

        agent_t_1 = AgentReinforce(rl_policy_t_1, self.device, greedy, optim_t_1, 0.0)
        agent_t_2 = AgentReinforce(rl_policy_t_2, self.device, greedy, optim_t_2, 0.0)

        for batch in (test_loader):
            MIP_model = batch['mip_model'][0]
            incumbent_solution = batch['incumbent_solution'][0]
            agent1, agent2, agent_t_1, agent_t_2 = self.evaluate_lb_per_instance_rlactive_policy_kt(
                MIP_model=MIP_model,
                incumbent=incumbent_solution,
                node_time_limit=node_time_limit,
                total_time_limit=total_time_limit,
                reset_k_at_2nditeration=reset_k_at_2nditeration,
                agent1=agent1,
                agent2=agent2,
                agent_t_1=agent_t_1,
                agent_t_2=agent_t_2,
                t_reward_type=t_reward_type,
                enable_adapt_t=enable_adapt_t
                                                            )

            agent1, optim1, R = self.update_agent(agent1, optim1)
            agent2, optim2, R = self.update_agent(agent2, optim2)
            agent_t_1, optim_t_1, R = self.update_agent(agent_t_1, optim_t_1)
            agent_t_2, optim_t_2, R = self.update_agent(agent_t_2, optim_t_2)

    def evaluate_lb_per_instance_scip_rl(self, MIP_model, incumbent, node_time_limit, total_time_limit, reset_k_at_2nditeration=False,
                                 agent1=None, agent2=None, enable_adapt_t=False
                                 ):
        """
        evaluate a single MIP instance by two algorithms: lb-baseline and lb-pred_k
        :param node_time_limit:
        :param total_time_limit:
        :param index_instance:
        :return:
        """
        incumbent_solution = incumbent

        device = self.device
        gc.collect()


        instance_name = MIP_model.getProbName()
        print(instance_name)
        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

        feas = MIP_model.checkSol(incumbent_solution)
        try:
            MIP_model.addSol(incumbent_solution, False)
        except:
            print('Error: the root solution of ' + instance_name + ' is not feasible!')

        # variable features: only incumbent solution
        variable_features = observation.variable_features[:, -1:]
        device = self.device
        graph = BipartiteNodeData(observation.constraint_features,
                                  observation.edge_features.indices,
                                  observation.edge_features.values,
                                  variable_features,
                                  device=device)
        graph = graph.to(device)


        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = observation.constraint_features.shape[0] + \
                          observation.variable_features.shape[
                              0]


        # solve the root node and get the LP solution, compute k_prime
        k_prime = self.compute_k_prime(MIP_model, incumbent)

        initial_obj = MIP_model.getSolObjVal(incumbent_solution)
        print("Initial obj before LB: {}".format(initial_obj))

        binary_supports = binary_support(MIP_model, incumbent_solution)
        print('binary support: ', binary_supports)


        k_model = self.regression_model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
                                            graph.variable_features)

        k_pred = k_model.item() * k_prime
        print('GNN prediction: ', k_model.item())

        if self.is_symmetric == False:
            k_pred = k_model.item() * k_prime

        k_pred = np.ceil(k_pred)

        if k_pred < 10:
            k_pred = 10

        del k_model
        del graph
        del observation

        # create a copy of MIP
        MIP_model.resetParams()
        MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
            problemName='noregression-rl',
            origcopy=False)
        MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
            problemName='regression-rl',
            origcopy=False)

        print('MIP copies are created')

        MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent_solution,
                                                  MIP_copy_vars2)
        MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent_solution,
                                                  MIP_copy_vars3)

        print('incumbent solution is copied to MIP copies')
        MIP_model.freeProb()
        del MIP_model
        del incumbent_solution


        # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
        lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=k_pred,
                                   node_time_limit=node_time_limit,
                                   total_time_limit=total_time_limit)

        status, obj_best, elapsed_time, lb_bits_pred_reset, times_regression_reinforce_, objs_regression_reinforce_, agent1, _ = self.mdp_localbranch(
            localbranch=lb_model3,
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=reset_k_at_2nditeration,
            agent_k=agent1,
            optimizer_k=None,
            device=device,
            enable_adapt_t=enable_adapt_t)
        print("Instance:", MIP_model_copy3.getProbName())
        print("Status of LB: ", status)
        print("Best obj of LB: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        objs_regression_reinforce = np.array(lb_model3.primal_objs).reshape(-1)
        times_regression_reinforce = np.array(lb_model3.primal_times).reshape(-1)

        MIP_model_copy3.freeProb()
        del sol_MIP_copy3
        del MIP_model_copy3

        # execute local branching with 1. first k predicted by GNN; 2. for 2nd iteration of lb, continue lb algorithm with no further injection
        lb_model2 = LocalBranching(MIP_model=MIP_model_copy2, MIP_sol_bar=sol_MIP_copy2, k=self.k_baseline,
                                   node_time_limit=node_time_limit,
                                   total_time_limit=total_time_limit)

        status, obj_best, elapsed_time, lb_bits_pred, times_noregression_reinforce_, objs_noregression_reinforce_, agent2, _ = self.mdp_localbranch(
            localbranch=lb_model2,
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=reset_k_at_2nditeration,
            agent_k=agent2,
            optimizer_k=None,
            device=device)

        objs_noregression_reinforce = np.array(lb_model2.primal_objs).reshape(-1)
        times_noregression_reinforce = np.array(lb_model2.primal_times).reshape(-1)


        print("Instance:", MIP_model_copy2.getProbName())
        print("Status of LB: ", status)
        print("Best obj of LB: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        MIP_model_copy2.freeProb()
        del sol_MIP_copy2
        del MIP_model_copy2

        data = [objs_noregression_reinforce, times_noregression_reinforce, objs_regression_reinforce, times_regression_reinforce]
        filename = f'{self.directory_lb_test}lb-test-{instance_name}.pkl'  # instance 100-199
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)

        del data
        del lb_model2
        del lb_model3

        return agent1, agent2


    def primal_integral(self, test_instance_size, total_time_limit=60, node_time_limit=30, mean_option='arithmetic'):

        print(mean_option)

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/reinforce/test/old_models/'
        directory_lb_test = directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/seed'+ str(self.seed) + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/' + 'rl/reinforce/test/old_models/'
            directory_lb_test_2 = directory_2 + 'evaluation-reinforce4lb-from-' +  'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/seed'+ str(self.seed) + '/'
        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/' + 'rl/reinforce/test/old_models/'
            directory_lb_test_2 = directory_2 + 'evaluation-reinforce4lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/seed'+ str(self.seed) + '/'


        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/seed'+ str(self.seed) + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/seed'+ str(self.seed) + '/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/seed'+ str(self.seed) + '/'


        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/seed'+ str(self.seed) + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/seed'+ str(self.seed) + '/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/seed'+ str(self.seed) + '/'

        # k_prime trained by data without merge
        directory_lb_test_k_prime = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/seed'+ str(self.seed) + '/'
        # k_prime trained by data with merge
        directory_lb_test_k_prime_merged = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/seed'+ str(self.seed) + '/'
        directory_lb_test_baseline = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/seed'+ str(self.seed) + '/'

        directory_lb_test_baseline_k0_average = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(
            total_time_limit) + 's' + test_instance_size + '_baseline_k0_average/seed' + str(123) + '/'

        primal_int_baselines = []
        primal_int_regressions_merged = []
        primal_int_regressions = []
        primal_int_regression_reinforces = []
        primal_int_reinforces = []
        primal_gap_final_baselines = []
        primal_gap_final_regressions = []
        primal_gap_final_regressions_merged = []
        primal_gap_final_regression_reinforces = []
        primal_gap_final_reinforces = []
        steplines_baseline = []
        steplines_regression = []
        steplines_regression_merged = []
        steplines_regression_reinforce = []
        steplines_reinforce = []

        primal_int_reinforces_talored = []
        primal_gap_final_reinforces_talored = []
        steplines_reinforce_talored = []

        index_mix = 160
        index_max = 200

        if self.instance_type == instancetypes[2] and test_instance_size == '-large':
            index_mix = 0
            index_max = 40

        for i in range(index_mix, index_max):
            instance_name = self.instance_type + '-' + str(i) + '_transformed' # instance 100-199

            filename = f'{directory_lb_test}lb-test-{instance_name}.pkl'

            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_reinforce, times_reinforce, objs_regresison_reinforce, times_regression_reinforce = data  # objs contains objs of a single instance of a lb test

            filename = f'{directory_lb_test_2}lb-test-{instance_name}.pkl'

            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_reinforce_2, times_reinforce_2, objs_regresison_reinforce_2, times_regression_reinforce_2 = data  # objs contains objs of a single instance of a lb test


            # test from k_prime
            filename = f'{directory_lb_test_k_prime}lb-test-{instance_name}.pkl'
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_k_prime, times_k_prime = data  # objs contains objs of a single instance of a lb test

            # test from k_prime_merged
            filename = f'{directory_lb_test_k_prime_merged}lb-test-{instance_name}.pkl'
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_k_prime_merged, times_k_prime_merged = data  # objs contains objs of a single instance of a lb test

            # test from baseline
            filename = f'{directory_lb_test_baseline}lb-test-{instance_name}.pkl'
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs, times = data  # objs contains objs of a single instance of a lb test

            # test from baseline
            filename = f'{directory_lb_test_baseline_k0_average}lb-test-{instance_name}.pkl'
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_lb_baseline_k0_average, times_lb_baseline_k0_average = data  # objs contains objs of a single instance of a lb test

            filename = f'{directory_lb_test_k_prime_2}lb-test-{instance_name}.pkl'
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_k_prime_2, times_k_prime_2 = data  # objs contains objs of a single instance of a lb test

            filename = f'{directory_lb_test_k_prime_merged_2}lb-test-{instance_name}.pkl'
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_k_prime_merged_2, times_k_prime_merged_2 = data  # objs contains objs of a single instance of a lb test

            filename = f'{directory_lb_test_baseline_2}lb-test-{instance_name}.pkl'
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_2, times_k_2 = data  # objs contains objs of a single instance of a lb test

            objs_reinforce = np.array(objs_reinforce).reshape(-1)
            times_reinforce = np.array(times_reinforce).reshape(-1)
            objs_regresison_reinforce = np.array(objs_regresison_reinforce).reshape(-1)
            times_regression_reinforce = np.array(times_regression_reinforce).reshape(-1)

            objs_reinforce_2 = np.array(objs_reinforce_2).reshape(-1)
            objs_regresison_reinforce_2 = np.array(objs_regresison_reinforce_2).reshape(-1)

            objs = np.array(objs).reshape(-1)
            times = np.array(times).reshape(-1)

            objs_2 = np.array(objs_2).reshape(-1)

            objs_k_prime = np.array(objs_k_prime).reshape(-1)
            times_k_prime = np.array(times_k_prime).reshape(-1)

            objs_k_prime_2 = np.array(objs_k_prime_2).reshape(-1)

            objs_k_prime_merged = np.array(objs_k_prime_merged).reshape(-1)
            times_k_prime_merged = np.array(times_k_prime_merged).reshape(-1)

            objs_k_prime_merged_2 = np.array(objs_k_prime_merged_2).reshape(-1)

            a = [objs_reinforce.min(), objs_regresison_reinforce.min(), objs_reinforce_2.min(), objs_regresison_reinforce_2.min(), objs.min(), objs_2.min(), objs_k_prime.min(), objs_k_prime_2.min(), objs_k_prime_merged.min(), objs_k_prime_merged_2.min()]
            obj_opt = np.amin(a)

            # lb-baseline:
            # compute primal gap for baseline localbranching run
            # if times[-1] < total_time_limit:
            primal_int_baseline, primal_gap_final_baseline, stepline_baseline = self.compute_primal_integral(times=times, objs=objs, obj_opt=obj_opt, total_time_limit=total_time_limit)
            primal_gap_final_baselines.append(primal_gap_final_baseline)
            steplines_baseline.append(stepline_baseline)
            primal_int_baselines.append(primal_int_baseline)

            # lb-regression
            # if times_regression[-1] < total_time_limit:

            primal_int_regression, primal_gap_final_regression, stepline_regression = self.compute_primal_integral(
                times=times_k_prime, objs=objs_k_prime, obj_opt=obj_opt, total_time_limit=total_time_limit)
            primal_gap_final_regressions.append(primal_gap_final_regression)
            steplines_regression.append(stepline_regression)
            primal_int_regressions.append(primal_int_regression)

            # lb-regression-merged
            # if times_regression[-1] < total_time_limit:

            primal_int_regression_merged, primal_gap_final_regression_merged, stepline_regression_merged = self.compute_primal_integral(
                times=times_k_prime_merged, objs=objs_k_prime_merged, obj_opt=obj_opt, total_time_limit=total_time_limit)
            primal_gap_final_regressions_merged.append(primal_gap_final_regression_merged)
            steplines_regression_merged.append(stepline_regression_merged)
            primal_int_regressions_merged.append(primal_int_regression_merged)


            # lb-regression-reinforce

            primal_int_regression_reinforce, primal_gap_final_regression_reinforce, stepline_regression_reinforce = self.compute_primal_integral(
                times=times_regression_reinforce, objs=objs_regresison_reinforce, obj_opt=obj_opt, total_time_limit=total_time_limit)
            primal_gap_final_regression_reinforces.append(primal_gap_final_regression_reinforce)
            steplines_regression_reinforce.append(stepline_regression_reinforce)
            primal_int_regression_reinforces.append(primal_int_regression_reinforce)

            # lb-reinforce

            primal_int_reinforce, primal_gap_final_reinforce, stepline_reinforce = self.compute_primal_integral(
                times=times_reinforce, objs=objs_reinforce, obj_opt=obj_opt,
                total_time_limit=total_time_limit)
            primal_gap_final_reinforces.append(primal_gap_final_reinforce)
            steplines_reinforce.append(stepline_reinforce)
            primal_int_reinforces.append(primal_int_reinforce)


            # lb-reinforce-talored, or lb-baseline_k0_average
            primal_int_reinforce_talored, primal_gap_final_reinforce_talored, stepline_reinforce_talored = self.compute_primal_integral(
                times=times_lb_baseline_k0_average, objs=objs_lb_baseline_k0_average, obj_opt=obj_opt,
                total_time_limit=total_time_limit)
            primal_gap_final_reinforces_talored.append(primal_gap_final_reinforce_talored)
            steplines_reinforce_talored.append(stepline_reinforce_talored)
            primal_int_reinforces_talored.append(primal_int_reinforce_talored)


        primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_regressions = np.array(primal_int_regressions).reshape(-1)
        primal_int_regressions_merged = np.array(primal_int_regressions_merged).reshape(-1)
        primal_int_regression_reinforces = np.array(primal_int_regression_reinforces).reshape(-1)
        primal_int_reinforces = np.array(primal_int_reinforces).reshape(-1)

        primal_int_reinforces_talored = np.array(primal_int_reinforces_talored).reshape(-1)

        primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_regressions = np.array(primal_gap_final_regressions).reshape(-1)
        primal_gap_final_regressions_merged = np.array(primal_gap_final_regressions_merged).reshape(-1)
        primal_gap_final_regression_reinforces = np.array(primal_gap_final_regression_reinforces).reshape(-1)
        primal_gap_final_reinforces = np.array(primal_gap_final_reinforces).reshape(-1)

        # # primal_gap_final_regression_reinforces_talored = np.array(primal_gap_final_regression_reinforces_talored).reshape(-1)
        primal_gap_final_reinforces_talored = np.array(primal_gap_final_reinforces_talored).reshape(-1)

        primal_int_base_ave = mean_shift(primal_int_baselines,
                                         mean_option=mean_option)
        primal_int_regression_ave = mean_shift(primal_int_regressions,
                                               mean_option=mean_option)
        primal_int_regression_merged_ave = mean_shift(primal_int_regressions_merged,
                                                      mean_option=mean_option)
        primal_int_regression_reinforce_ave = mean_shift(primal_int_regression_reinforces,
                                                         mean_option=mean_option)
        primal_int_reinforce_ave = mean_shift(primal_int_reinforces,
                                              mean_option=mean_option)

        primal_int_reinforce_talored_ave = mean_shift(primal_int_reinforces_talored,
                                                      mean_option=mean_option)

        primal_gap_final_baseline_ave = mean_shift(primal_gap_final_baselines,
                                                   mean_option=mean_option)
        primal_gap_final_regression_ave = mean_shift(primal_gap_final_regressions,
                                                     mean_option=mean_option)
        primal_gap_final_regression_merged_ave = mean_shift(primal_gap_final_regressions_merged,
                                                            mean_option=mean_option)
        primal_gap_final_regression_reinforce_ave = mean_shift(primal_gap_final_regression_reinforces,
                                                               mean_option=mean_option)
        primal_gap_final_reinforce_ave = mean_shift(primal_gap_final_reinforces,
                                                    mean_option=mean_option)

        primal_gap_final_reinforce_talored_ave = mean_shift(primal_gap_final_reinforces_talored,
                                                            mean_option=mean_option)

        print(self.instance_type + test_instance_size)
        print(self.incumbent_mode + 'Solution')
        print('baseline primal integral: ', np.round(primal_int_base_ave, 3))
        print('baseline k0 average primal integral: ', np.round(primal_int_reinforce_talored_ave, 3))
        print('regression primal integral: ', np.round(primal_int_regression_ave, 3))
        print('regression merged primal integral: ', np.round(primal_int_regression_merged_ave, 3))
        print('rl primal integral: ', np.round(primal_int_reinforce_ave, 3))
        print('regression-rl primal integral: ', np.round(primal_int_regression_reinforce_ave, 3))

        print('\n')
        print('baseline primal gap: ', np.round(primal_gap_final_baseline_ave, 3))
        print('baseline k0 average primal gap: ', np.round(primal_gap_final_reinforce_talored_ave, 3))
        print('regression primal gap: ', np.round(primal_gap_final_regression_ave, 3))
        print('regression primal merged gap: ', np.round(primal_gap_final_regression_merged_ave, 3))
        print('rl primal gap: ', np.round(primal_gap_final_reinforce_ave, 3))
        print('regression-rl primal gap: ', np.round(primal_gap_final_regression_reinforce_ave, 3))
        print('\n')
        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)

        primalgaps_baseline = None
        for n, stepline_baseline in enumerate(steplines_baseline):
            primal_gap = stepline_baseline(t)
            if n == 0:
                primalgaps_baseline = primal_gap
            else:
                primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        primalgap_baseline_ave = mean_shift(primalgaps_baseline, axis=0,
                                            mean_option=mean_option)

        primalgaps_regression = None
        for n, stepline_regression in enumerate(steplines_regression):
            primal_gap = stepline_regression(t)
            if n == 0:
                primalgaps_regression = primal_gap
            else:
                primalgaps_regression = np.vstack((primalgaps_regression, primal_gap))
        primalgap_regression_ave = mean_shift(primalgaps_regression, axis=0, mean_option=mean_option)

        primalgaps_regression_merged = None
        for n, stepline_regression in enumerate(steplines_regression_merged):
            primal_gap = stepline_regression(t)
            if n == 0:
                primalgaps_regression_merged = primal_gap
            else:
                primalgaps_regression_merged = np.vstack((primalgaps_regression_merged, primal_gap))
        primalgap_regression_merged_ave = mean_shift(primalgaps_regression_merged, axis=0,
                                                     mean_option=mean_option)

        primalgaps_regression_reinforce = None
        for n, stepline_regression_reinforce in enumerate(steplines_regression_reinforce):
            primal_gap = stepline_regression_reinforce(t)
            if n == 0:
                primalgaps_regression_reinforce = primal_gap
            else:
                primalgaps_regression_reinforce = np.vstack((primalgaps_regression_reinforce, primal_gap))
        primalgap_regression_reinforce_ave = mean_shift(primalgaps_regression_reinforce, axis=0,
                                                        mean_option=mean_option)

        primalgaps_reinforce = None
        for n, stepline_reinforce in enumerate(steplines_reinforce):
            primal_gap = stepline_reinforce(t)
            if n == 0:
                primalgaps_reinforce = primal_gap
            else:
                primalgaps_reinforce = np.vstack((primalgaps_reinforce, primal_gap))
        primalgap_reinforce_ave = mean_shift(primalgaps_reinforce, axis=0,
                                             mean_option=mean_option)

        primalgaps_reinforce_talored = None
        for n, stepline_reinforce in enumerate(steplines_reinforce_talored):
            primal_gap = stepline_reinforce(t)
            if n == 0:
                primalgaps_reinforce_talored = primal_gap
            else:
                primalgaps_reinforce_talored = np.vstack((primalgaps_reinforce_talored, primal_gap))
        primalgap_reinforce_talored_ave = np.average(primalgaps_reinforce_talored, axis=0)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle(self.instance_type + test_instance_size + '-' + self.incumbent_mode, fontsize=13)
        ax.plot(t, primalgap_baseline_ave, label='lb-base', color='tab:blue')
        if test_instance_size == '-small':
            ax.plot(t, primalgap_regression_ave, label='lb-sr', color ='tab:grey')
        ax.plot(t, primalgap_regression_merged_ave, label='lb-srm', color='tab:orange')
        ax.plot(t, primalgap_reinforce_ave, '--', label='lb-rl', color='tab:green')
        ax.plot(t, primalgap_regression_reinforce_ave,'--', label='lb-srmrl', color='tab:red')
        #
        ax.plot(t, primalgap_reinforce_talored_ave, ':', label='lb-base-k0-average', color='tab:green')

        ax.set_xlabel('time /s', fontsize=12)
        ax.set_ylabel("scaled primal gap", fontsize=12)
        ax.legend()
        ax.grid()
        plt.savefig('./result/plots/' + self.instance_type + '_' + test_instance_size + '_' + self.incumbent_mode + '_tnode' + str(node_time_limit) + 's' + '_ttotal' + str(total_time_limit) + 's_' + 'server' + '_oldlb_seed' + str(self.seed) + '_' + mean_option + '20240418_k0base_merged.png')
        plt.show()
        plt.clf()

    def primal_integral_03(self, test_instance_size, total_time_limit=60, node_time_limit=30, mean_option='arithmetic'):

        print(mean_option)

        direc = './data/generated_instances/' + self.instance_type + '/' + test_instance_size + '/'
        directory_transformedmodel = direc + 'transformedmodel' + '/test/'

        # set directory for the test result of RL-policy1
        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/reinforce/test/old_models/'
        directory_lb_test = directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/seed'+ str(self.seed) + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/' + 'rl/reinforce/test/old_models/'
            directory_lb_test_2 = directory_2 + 'evaluation-reinforce4lb-from-' +  'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/seed'+ str(self.seed) + '/'
        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/' + 'rl/reinforce/test/old_models/'
            directory_lb_test_2 = directory_2 + 'evaluation-reinforce4lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/seed'+ str(self.seed) + '/'


        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/seed'+ str(self.seed) + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/seed'+ str(self.seed) + '/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/seed'+ str(self.seed) + '/'


        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/seed'+ str(self.seed) + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/seed'+ str(self.seed) + '/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/seed'+ str(self.seed) + '/'

        # k_prime trained by data without merge
        directory_lb_test_k_prime = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/seed'+ str(self.seed) + '/'
        # k_prime trained by data with merge
        directory_lb_test_k_prime_merged = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/seed'+ str(self.seed) + '/'
        directory_lb_test_baseline = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/seed'+ str(self.seed) + '/'

        primal_int_baselines = []
        primal_int_regressions_merged = []
        primal_int_regressions = []
        primal_int_regression_reinforces = []
        primal_int_reinforces = []
        primal_gap_final_baselines = []
        primal_gap_final_regressions = []
        primal_gap_final_regressions_merged = []
        primal_gap_final_regression_reinforces = []
        primal_gap_final_reinforces = []
        steplines_baseline = []
        steplines_regression = []
        steplines_regression_merged = []
        steplines_regression_reinforce = []
        steplines_reinforce = []


        if self.instance_type == instancetypes[3]:
            index_mix = 80
            index_max = 115
        elif self.instance_type == instancetypes[4]:
            index_mix = 0
            index_max = 30

        for i in range(index_mix,index_max):

            if not (self.instance_type == instancetypes[4] and i == 18):

                instance_name = self.instance_type + '-' + str(i) + '_transformed' # instance 100-199

                mip_filename = f'{directory_transformedmodel}{instance_name}.cip'
                mip = Model()
                MIP_model = Model()
                MIP_model.readProblem(mip_filename)
                instance_name = MIP_model.getProbName()

                filename = f'{directory_lb_test}lb-test-{instance_name}.pkl'

                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_reinforce, times_reinforce, objs_regresison_reinforce, times_regression_reinforce = data  # objs contains objs of a single instance of a lb test

                filename = f'{directory_lb_test_2}lb-test-{instance_name}.pkl'

                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_reinforce_2, times_reinforce_2, objs_regresison_reinforce_2, times_regression_reinforce_2 = data  # objs contains objs of a single instance of a lb test


                instance_name = self.instance_type + '-' + str(i) + '_transformed'  # instance 100-199

                # test from k_prime_merged
                filename = f'{directory_lb_test_k_prime_merged}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_k_prime_merged, times_k_prime_merged = data  # objs contains objs of a single instance of a lb test

                # test from baseline
                filename = f'{directory_lb_test_baseline}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs, times = data  # objs contains objs of a single instance of a lb test


                filename = f'{directory_lb_test_k_prime_merged_2}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_k_prime_merged_2, times_k_prime_merged_2 = data  # objs contains objs of a single instance of a lb test

                filename = f'{directory_lb_test_baseline_2}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_2, times_k_2 = data  # objs contains objs of a single instance of a lb test

                objs_reinforce = np.array(objs_reinforce).reshape(-1)
                times_reinforce = np.array(times_reinforce).reshape(-1)
                objs_regresison_reinforce = np.array(objs_regresison_reinforce).reshape(-1)
                times_regression_reinforce = np.array(times_regression_reinforce).reshape(-1)

                objs_reinforce_2 = np.array(objs_reinforce_2).reshape(-1)
                objs_regresison_reinforce_2 = np.array(objs_regresison_reinforce_2).reshape(-1)

                objs = np.array(objs).reshape(-1)
                times = np.array(times).reshape(-1)

                objs_2 = np.array(objs_2).reshape(-1)


                objs_k_prime_merged = np.array(objs_k_prime_merged).reshape(-1)
                times_k_prime_merged = np.array(times_k_prime_merged).reshape(-1)

                objs_k_prime_merged_2 = np.array(objs_k_prime_merged_2).reshape(-1)

                a = [objs_reinforce.min(), objs_regresison_reinforce.min(), objs_reinforce_2.min(), objs_regresison_reinforce_2.min(), objs.min(), objs_2.min(), objs_k_prime_merged.min(), objs_k_prime_merged_2.min()]
                obj_opt = np.amin(a)

                # lb-baseline:
                # compute primal gap for baseline localbranching run
                # if times[-1] < total_time_limit:
                primal_int_baseline, primal_gap_final_baseline, stepline_baseline = self.compute_primal_integral(times=times, objs=objs, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_baselines.append(primal_gap_final_baseline)
                steplines_baseline.append(stepline_baseline)
                primal_int_baselines.append(primal_int_baseline)


                # lb-regression-merged
                # if times_regression[-1] < total_time_limit:

                primal_int_regression_merged, primal_gap_final_regression_merged, stepline_regression_merged = self.compute_primal_integral(
                    times=times_k_prime_merged, objs=objs_k_prime_merged, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_regressions_merged.append(primal_gap_final_regression_merged)
                steplines_regression_merged.append(stepline_regression_merged)
                primal_int_regressions_merged.append(primal_int_regression_merged)


                # lb-regression-reinforce

                primal_int_regression_reinforce, primal_gap_final_regression_reinforce, stepline_regression_reinforce = self.compute_primal_integral(
                    times=times_regression_reinforce, objs=objs_regresison_reinforce, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_regression_reinforces.append(primal_gap_final_regression_reinforce)
                steplines_regression_reinforce.append(stepline_regression_reinforce)
                primal_int_regression_reinforces.append(primal_int_regression_reinforce)

                # lb-reinforce

                primal_int_reinforce, primal_gap_final_reinforce, stepline_reinforce = self.compute_primal_integral(
                    times=times_reinforce, objs=objs_reinforce, obj_opt=obj_opt,
                    total_time_limit=total_time_limit)
                primal_gap_final_reinforces.append(primal_gap_final_reinforce)
                steplines_reinforce.append(stepline_reinforce)
                primal_int_reinforces.append(primal_int_reinforce)


        primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_regressions = np.array(primal_int_regressions).reshape(-1)
        primal_int_regressions_merged = np.array(primal_int_regressions_merged).reshape(-1)
        primal_int_regression_reinforces = np.array(primal_int_regression_reinforces).reshape(-1)
        primal_int_reinforces = np.array(primal_int_reinforces).reshape(-1)


        primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_regressions = np.array(primal_gap_final_regressions).reshape(-1)
        primal_gap_final_regressions_merged = np.array(primal_gap_final_regressions_merged).reshape(-1)
        primal_gap_final_regression_reinforces = np.array(primal_gap_final_regression_reinforces).reshape(-1)
        primal_gap_final_reinforces = np.array(primal_gap_final_reinforces).reshape(-1)


        # average primal integral over test dataset

        primal_int_base_ave = mean_shift(primal_int_baselines,
                                         mean_option=mean_option)
        primal_int_regression_ave = mean_shift(primal_int_regressions,
                                               mean_option=mean_option)
        primal_int_regression_merged_ave = mean_shift(primal_int_regressions_merged,
                                                      mean_option=mean_option)
        primal_int_regression_reinforce_ave = mean_shift(primal_int_regression_reinforces,
                                                         mean_option=mean_option)
        primal_int_reinforce_ave = mean_shift(primal_int_reinforces,
                                              mean_option=mean_option)


        primal_gap_final_baseline_ave = mean_shift(primal_gap_final_baselines,
                                                   mean_option=mean_option)
        primal_gap_final_regression_ave = mean_shift(primal_gap_final_regressions,
                                                     mean_option=mean_option)
        primal_gap_final_regression_merged_ave = mean_shift(primal_gap_final_regressions_merged,
                                                            mean_option=mean_option)
        primal_gap_final_regression_reinforce_ave = mean_shift(primal_gap_final_regression_reinforces,
                                                               mean_option=mean_option)
        primal_gap_final_reinforce_ave = mean_shift(primal_gap_final_reinforces,
                                                    mean_option=mean_option)


        print(self.instance_type + test_instance_size)
        print(self.incumbent_mode + 'Solution')
        print('baseline primal integral: ', np.round(primal_int_base_ave, 3))
        print('regression primal integral: ', np.round(primal_int_regression_ave, 3))
        print('regression merged primal integral: ', np.round(primal_int_regression_merged_ave, 3))
        print('rl primal integral: ', np.round(primal_int_reinforce_ave, 3))
        print('regression-rl primal integral: ', np.round(primal_int_regression_reinforce_ave, 3))

        print('\n')
        print('baseline primal gap: ', np.round(primal_gap_final_baseline_ave, 3))
        print('regression primal gap: ', np.round(primal_gap_final_regression_ave, 3))
        print('regression primal merged gap: ', np.round(primal_gap_final_regression_merged_ave, 3))
        print('rl primal gap: ', np.round(primal_gap_final_reinforce_ave, 3))
        print('regression-rl primal gap: ', np.round(primal_gap_final_regression_reinforce_ave, 3))
        print('\n')

        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)

        primalgaps_baseline = None
        for n, stepline_baseline in enumerate(steplines_baseline):
            primal_gap = stepline_baseline(t)
            if n == 0:
                primalgaps_baseline = primal_gap
            else:
                primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        primalgap_baseline_ave = mean_shift(primalgaps_baseline, axis=0,
                                            mean_option=mean_option)


        primalgaps_regression_merged = None
        for n, stepline_regression in enumerate(steplines_regression_merged):
            primal_gap = stepline_regression(t)
            if n == 0:
                primalgaps_regression_merged = primal_gap
            else:
                primalgaps_regression_merged = np.vstack((primalgaps_regression_merged, primal_gap))
        primalgap_regression_merged_ave = mean_shift(primalgaps_regression_merged, axis=0,
                                                     mean_option=mean_option)

        primalgaps_regression_reinforce = None
        for n, stepline_regression_reinforce in enumerate(steplines_regression_reinforce):
            primal_gap = stepline_regression_reinforce(t)
            if n == 0:
                primalgaps_regression_reinforce = primal_gap
            else:
                primalgaps_regression_reinforce = np.vstack((primalgaps_regression_reinforce, primal_gap))
        primalgap_regression_reinforce_ave = mean_shift(primalgaps_regression_reinforce, axis=0,
                                                        mean_option=mean_option)

        primalgaps_reinforce = None
        for n, stepline_reinforce in enumerate(steplines_reinforce):
            primal_gap = stepline_reinforce(t)
            if n == 0:
                primalgaps_reinforce = primal_gap
            else:
                primalgaps_reinforce = np.vstack((primalgaps_reinforce, primal_gap))
        primalgap_reinforce_ave = mean_shift(primalgaps_reinforce, axis=0,
                                             mean_option=mean_option)


        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle(self.instance_type + '-' + self.incumbent_mode, fontsize=13)
        ax.plot(t, primalgap_baseline_ave, label='lb-base', color='tab:blue')
        ax.plot(t, primalgap_regression_merged_ave, label='lb-srm', color='tab:orange')
        ax.plot(t, primalgap_reinforce_ave, '--', label='lb-rl', color='tab:green')
        ax.plot(t, primalgap_regression_reinforce_ave,'--', label='lb-srmrl', color='tab:red')

        ax.set_xlabel('time /s', fontsize=12)
        ax.set_ylabel("scaled primal gap", fontsize=12)
        ax.legend()
        ax.grid()
        plt.savefig('./result/plots/' + self.instance_type + '_' + test_instance_size + '_' + self.incumbent_mode + '_tnode' + str(node_time_limit) + 's' + '_ttotal' + str(total_time_limit) + 's' + '_server'+ '_oldlb_seed' + str(self.seed) + '_' + mean_option + '.png')
        plt.show()
        plt.clf()

    def primal_integral_hybrid_03(self, test_instance_size, total_time_limit=60, node_time_limit=30):

        direc = './data/generated_instances/' + self.instance_type + '/' + test_instance_size + '/'
        directory_transformedmodel = direc + 'transformedmodel' + '/test/'

        # set directory for the test result of RL-policy1
        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/reinforce/test/old_models/'
        directory_lb_test = directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/' + 'rl/reinforce/test/old_models/'
            directory_lb_test_2 = directory_2 + 'evaluation-reinforce4lb-from-' +  'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/'
        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/' + 'rl/reinforce/test/old_models/'
            directory_lb_test_2 = directory_2 + 'evaluation-reinforce4lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/'

        # set directory for the test result of RL-policy1-t_node_baseline
        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/reinforce/test/old_models/'
        directory_lb_test_hybrid = directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive_t_node_baseline/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/' + 'rl/reinforce/test/old_models/'
            directory_lb_test_hybrid_2 = directory_2 + 'evaluation-reinforce4lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(
                total_time_limit) + 's' + test_instance_size + '/rlactive_t_node_baseline/'
        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/' + 'rl/reinforce/test/old_models/'
            directory_lb_test_hybrid_2 = directory_2 + 'evaluation-reinforce4lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(
                total_time_limit) + 's' + test_instance_size + '/rlactive_t_node_baseline/'


        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'


        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'

        # k_prime trained by data without merge
        directory_lb_test_k_prime = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        # k_prime trained by data with merge
        directory_lb_test_k_prime_merged = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
        directory_lb_test_baseline = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'

        primal_int_baselines = []
        primal_int_regressions_merged = []
        primal_int_regressions = []
        primal_int_regression_reinforces = []
        primal_int_reinforces = []
        primal_gap_final_baselines = []
        primal_gap_final_regressions = []
        primal_gap_final_regressions_merged = []
        primal_gap_final_regression_reinforces = []
        primal_gap_final_reinforces = []
        steplines_baseline = []
        steplines_regression = []
        steplines_regression_merged = []
        steplines_regression_reinforce = []
        steplines_reinforce = []

        primal_int_regression_reinforces_talored = []
        primal_int_reinforces_talored = []
        primal_gap_final_regression_reinforces_talored = []
        primal_gap_final_reinforces_talored = []
        steplines_regression_reinforce_talored = []
        steplines_reinforce_talored = []

        if self.instance_type == instancetypes[3]:
            index_mix = 80
            index_max = 115
        elif self.instance_type == instancetypes[4]:
            index_mix = 0
            index_max = 30

        for i in range(index_mix, index_max):

            if not (self.instance_type == instancetypes[4] and i == 18):

                instance_name = self.instance_type + '-' + str(i) + '_transformed' # instance 100-199

                mip_filename = f'{directory_transformedmodel}{instance_name}.cip'
                mip = Model()
                MIP_model = Model()
                MIP_model.readProblem(mip_filename)
                instance_name = MIP_model.getProbName()

                filename = f'{directory_lb_test}lb-test-{instance_name}.pkl'

                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_reinforce, times_reinforce, objs_regresison_reinforce, times_regression_reinforce = data  # objs contains objs of a single instance of a lb test

                filename = f'{directory_lb_test_2}lb-test-{instance_name}.pkl'

                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_reinforce_2, times_reinforce_2, objs_regresison_reinforce_2, times_regression_reinforce_2 = data  # objs contains objs of a single instance of a lb test

                # load data of hybrid algorithm adapting t_node
                filename = f'{directory_lb_test_hybrid}lb-test-{instance_name}.pkl'

                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_reinforce_hybrid, times_reinforce_hybrid, objs_regresison_reinforce_hybrid, times_regression_reinforce_hybrid = data  # objs contains objs of a single instance of a lb test


                instance_name = self.instance_type + '-' + str(i) + '_transformed'  # instance 100-199
                # test from k_prime_merged
                filename = f'{directory_lb_test_k_prime_merged}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_k_prime_merged, times_k_prime_merged = data  # objs contains objs of a single instance of a lb test

                # test from baseline
                filename = f'{directory_lb_test_baseline}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs, times = data  # objs contains objs of a single instance of a lb test


                filename = f'{directory_lb_test_k_prime_merged_2}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_k_prime_merged_2, times_k_prime_merged_2 = data  # objs contains objs of a single instance of a lb test

                filename = f'{directory_lb_test_baseline_2}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_2, times_k_2 = data  # objs contains objs of a single instance of a lb test

                objs_reinforce = np.array(objs_reinforce).reshape(-1)
                times_reinforce = np.array(times_reinforce).reshape(-1)
                objs_regresison_reinforce = np.array(objs_regresison_reinforce).reshape(-1)
                times_regression_reinforce = np.array(times_regression_reinforce).reshape(-1)

                objs_reinforce_2 = np.array(objs_reinforce_2).reshape(-1)
                objs_regresison_reinforce_2 = np.array(objs_regresison_reinforce_2).reshape(-1)

                objs_reinforce_hybrid = np.array(objs_reinforce_hybrid).reshape(-1)
                times_reinforce_hybrid = np.array(times_reinforce_hybrid).reshape(-1)
                objs_regresison_reinforce_hybrid = np.array(objs_regresison_reinforce_hybrid).reshape(-1)
                times_regression_reinforce_hybrid = np.array(times_regression_reinforce_hybrid).reshape(-1)


                objs = np.array(objs).reshape(-1)
                times = np.array(times).reshape(-1)

                objs_2 = np.array(objs_2).reshape(-1)


                objs_k_prime_merged = np.array(objs_k_prime_merged).reshape(-1)
                times_k_prime_merged = np.array(times_k_prime_merged).reshape(-1)

                objs_k_prime_merged_2 = np.array(objs_k_prime_merged_2).reshape(-1)

                a = [objs_reinforce.min(), objs_regresison_reinforce.min(), objs_reinforce_2.min(), objs_regresison_reinforce_2.min(), objs.min(), objs_2.min(), objs_k_prime_merged.min(), objs_k_prime_merged_2.min(), objs_reinforce_hybrid.min(), objs_regresison_reinforce_hybrid.min()] # , objs_reinforce_hybrid_2.min(), objs_regresison_reinforce_hybrid_2.min(),
                obj_opt = np.amin(a)

                # lb-baseline:
                # compute primal gap for baseline localbranching run
                # if times[-1] < total_time_limit:
                primal_int_baseline, primal_gap_final_baseline, stepline_baseline = self.compute_primal_integral(times=times, objs=objs, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_baselines.append(primal_gap_final_baseline)
                steplines_baseline.append(stepline_baseline)
                primal_int_baselines.append(primal_int_baseline)


                # lb-regression-merged
                # if times_regression[-1] < total_time_limit:

                primal_int_regression_merged, primal_gap_final_regression_merged, stepline_regression_merged = self.compute_primal_integral(
                    times=times_k_prime_merged, objs=objs_k_prime_merged, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_regressions_merged.append(primal_gap_final_regression_merged)
                steplines_regression_merged.append(stepline_regression_merged)
                primal_int_regressions_merged.append(primal_int_regression_merged)


                # lb-regression-reinforce

                primal_int_regression_reinforce, primal_gap_final_regression_reinforce, stepline_regression_reinforce = self.compute_primal_integral(
                    times=times_regression_reinforce, objs=objs_regresison_reinforce, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_regression_reinforces.append(primal_gap_final_regression_reinforce)
                steplines_regression_reinforce.append(stepline_regression_reinforce)
                primal_int_regression_reinforces.append(primal_int_regression_reinforce)

                # lb-reinforce

                primal_int_reinforce, primal_gap_final_reinforce, stepline_reinforce = self.compute_primal_integral(
                    times=times_reinforce, objs=objs_reinforce, obj_opt=obj_opt,
                    total_time_limit=total_time_limit)
                primal_gap_final_reinforces.append(primal_gap_final_reinforce)
                steplines_reinforce.append(stepline_reinforce)
                primal_int_reinforces.append(primal_int_reinforce)

                # lb-regression-reinforce-talored
                primal_int_regression_reinforce_talored, primal_gap_final_regression_reinforce_talored, stepline_regression_reinforce_talored = self.compute_primal_integral(
                    times=times_regression_reinforce_hybrid, objs=objs_regresison_reinforce_hybrid, obj_opt=obj_opt,
                    total_time_limit=total_time_limit)
                primal_gap_final_regression_reinforces_talored.append(primal_gap_final_regression_reinforce_talored)
                steplines_regression_reinforce_talored.append(stepline_regression_reinforce_talored)
                primal_int_regression_reinforces_talored.append(primal_int_regression_reinforce_talored)

                # lb-reinforce

                primal_int_reinforce_talored, primal_gap_final_reinforce_talored, stepline_reinforce_talored = self.compute_primal_integral(
                    times=times_reinforce_hybrid, objs=objs_reinforce_hybrid, obj_opt=obj_opt,
                    total_time_limit=total_time_limit)
                primal_gap_final_reinforces_talored.append(primal_gap_final_reinforce_talored)
                steplines_reinforce_talored.append(stepline_reinforce_talored)
                primal_int_reinforces_talored.append(primal_int_reinforce_talored)


        primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_regressions = np.array(primal_int_regressions).reshape(-1)
        primal_int_regressions_merged = np.array(primal_int_regressions_merged).reshape(-1)
        primal_int_regression_reinforces = np.array(primal_int_regression_reinforces).reshape(-1)
        primal_int_reinforces = np.array(primal_int_reinforces).reshape(-1)

        primal_int_regression_reinforces_talored = np.array(primal_int_regression_reinforces_talored).reshape(-1)
        primal_int_reinforces_talored = np.array(primal_int_reinforces_talored).reshape(-1)

        primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_regressions = np.array(primal_gap_final_regressions).reshape(-1)
        primal_gap_final_regressions_merged = np.array(primal_gap_final_regressions_merged).reshape(-1)
        primal_gap_final_regression_reinforces = np.array(primal_gap_final_regression_reinforces).reshape(-1)
        primal_gap_final_reinforces = np.array(primal_gap_final_reinforces).reshape(-1)

        primal_gap_final_regression_reinforces_talored = np.array(primal_gap_final_regression_reinforces_talored).reshape(-1)
        primal_gap_final_reinforces_talored = np.array(primal_gap_final_reinforces_talored).reshape(-1)

        # average primal integral over test dataset
        primal_int_base_ave = primal_int_baselines.sum() / len(primal_int_baselines)
        primal_int_regression_ave = primal_int_regressions.sum() / len(primal_int_regressions)
        primal_int_regression_merged_ave = primal_int_regressions_merged.sum() / len(primal_int_regressions_merged)
        primal_int_regression_reinforce_ave = primal_int_regression_reinforces.sum() / len(primal_int_regression_reinforces)
        primal_int_reinforce_ave = primal_int_reinforces.sum() / len(
            primal_int_reinforces)

        primal_int_regression_reinforce_talored_ave = primal_int_regression_reinforces_talored.sum() / len(primal_int_regression_reinforces_talored)
        primal_int_reinforce_talored_ave = primal_int_reinforces_talored.sum() / len(
            primal_int_reinforces_talored)

        primal_gap_final_baseline_ave = primal_gap_final_baselines.sum() / len(primal_gap_final_baselines)
        primal_gap_final_regression_ave = primal_gap_final_regressions.sum() / len(primal_gap_final_regressions)
        primal_gap_final_regression_merged_ave = primal_gap_final_regressions_merged.sum() / len(primal_gap_final_regressions_merged)
        primal_gap_final_regression_reinforce_ave = primal_gap_final_regression_reinforces.sum() / len(primal_gap_final_regression_reinforces)
        primal_gap_final_reinforce_ave = primal_gap_final_reinforces.sum() / len(
            primal_gap_final_reinforces)

        primal_gap_final_regression_reinforce_talored_ave = primal_gap_final_regression_reinforces_talored.sum() / len(
            primal_gap_final_regression_reinforces_talored)
        primal_gap_final_reinforce_talored_ave = primal_gap_final_reinforces_talored.sum() / len(
            primal_gap_final_reinforces_talored)

        print(self.instance_type + test_instance_size)
        print(self.incumbent_mode + 'Solution')
        print('baseline primal integral: ', primal_int_base_ave)
        print('regression primal integral: ', primal_int_regression_ave)
        print('regression merged primal integral: ', primal_int_regression_merged_ave)
        print('rl primal integral: ', primal_int_reinforce_ave)
        print('regression-rl primal integral: ', primal_int_regression_reinforce_ave)

        print('rl-hybrid primal integral: ', primal_int_reinforce_talored_ave)
        print('regression-rl-hybrid primal integral: ', primal_int_regression_reinforce_talored_ave)

        print('\n')
        print('baseline primal gap: ', primal_gap_final_baseline_ave)
        print('regression primal gap: ', primal_gap_final_regression_ave)
        print('regression primal merged gap: ', primal_gap_final_regression_merged_ave)
        print('rl primal gap: ', primal_gap_final_reinforce_ave)
        print('regression-rl-hybrid primal gap: ', primal_gap_final_regression_reinforce_ave)

        print('rl-hybrid primal gap: ', primal_gap_final_reinforce_talored_ave)
        print('regression-rl-hybrid primal gap: ', primal_gap_final_regression_reinforce_talored_ave)

        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)

        primalgaps_baseline = None
        for n, stepline_baseline in enumerate(steplines_baseline):
            primal_gap = stepline_baseline(t)
            if n==0:
                primalgaps_baseline = primal_gap
            else:
                primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        primalgap_baseline_ave = np.average(primalgaps_baseline, axis=0)


        primalgaps_regression_merged = None
        for n, stepline_regression in enumerate(steplines_regression_merged):
            primal_gap = stepline_regression(t)
            if n == 0:
                primalgaps_regression_merged = primal_gap
            else:
                primalgaps_regression_merged = np.vstack((primalgaps_regression_merged, primal_gap))
        primalgap_regression_merged_ave = np.average(primalgaps_regression_merged, axis=0)

        primalgaps_regression_reinforce = None
        for n, stepline_regression_reinforce in enumerate(steplines_regression_reinforce):
            primal_gap = stepline_regression_reinforce(t)
            if n == 0:
                primalgaps_regression_reinforce = primal_gap
            else:
                primalgaps_regression_reinforce = np.vstack((primalgaps_regression_reinforce, primal_gap))
        primalgap_regression_reinforce_ave = np.average(primalgaps_regression_reinforce, axis=0)

        primalgaps_reinforce = None
        for n, stepline_reinforce in enumerate(steplines_reinforce):
            primal_gap = stepline_reinforce(t)
            if n == 0:
                primalgaps_reinforce = primal_gap
            else:
                primalgaps_reinforce = np.vstack((primalgaps_reinforce, primal_gap))
        primalgap_reinforce_ave = np.average(primalgaps_reinforce, axis=0)

        primalgaps_regression_reinforce_talored = None
        for n, stepline_regression_reinforce in enumerate(steplines_regression_reinforce_talored):
            primal_gap = stepline_regression_reinforce(t)
            if n == 0:
                primalgaps_regression_reinforce_talored = primal_gap
            else:
                primalgaps_regression_reinforce_talored = np.vstack((primalgaps_regression_reinforce_talored, primal_gap))
        primalgap_regression_reinforce_talored_ave = np.average(primalgaps_regression_reinforce_talored, axis=0)

        primalgaps_reinforce_talored = None
        for n, stepline_reinforce in enumerate(steplines_reinforce_talored):
            primal_gap = stepline_reinforce(t)
            if n == 0:
                primalgaps_reinforce_talored = primal_gap
            else:
                primalgaps_reinforce_talored = np.vstack((primalgaps_reinforce_talored, primal_gap))
        primalgap_reinforce_talored_ave = np.average(primalgaps_reinforce_talored, axis=0)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle(self.instance_type + '-' + self.incumbent_mode, fontsize=13)
        ax.plot(t, primalgap_baseline_ave, label='lb-base', color='tab:blue')
        ax.plot(t, primalgap_regression_merged_ave, label='lb-regression', color='tab:orange')
        ax.plot(t, primalgap_reinforce_ave, '--', label='lb-rl', color='tab:green')
        ax.plot(t, primalgap_regression_reinforce_ave,'--', label='lb-regression-rl', color='tab:red')
        #
        ax.plot(t, primalgap_reinforce_talored_ave, ':', label='lb-rl-adapt-t', color='tab:green')
        ax.plot(t, primalgap_regression_reinforce_talored_ave, ':', label='lb-regression-rl-adapt-t', color='tab:red')

        ax.set_xlabel('time /s', fontsize=12)
        ax.set_ylabel("scaled primal gap", fontsize=12)
        ax.legend()
        ax.grid()
        plt.savefig('./result/plots/' + self.instance_type + '_' + self.incumbent_mode + '_hybrid' +  '.png')
        plt.show()
        plt.clf()

    def primal_gap_integral_hybrid_03(self, test_instance_size, total_time_limit=60, node_time_limit=30, mean_option='arithmetic'):

        print(mean_option)

        direc = './data/generated_instances/' + self.instance_type + '/' + test_instance_size + '/'
        directory_transformedmodel = direc + 'transformedmodel' + '/test/'

        # set directory for the test result of RL-policy1
        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/reinforce/test/old_models/'
        directory_lb_test = directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/seed'+ str(self.seed) + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/' + 'rl/reinforce/test/old_models/'
            directory_lb_test_2 = directory_2 + 'evaluation-reinforce4lb-from-' +  'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/seed'+ str(self.seed) + '/'
        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/' + 'rl/reinforce/test/old_models/'
            directory_lb_test_2 = directory_2 + 'evaluation-reinforce4lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/seed'+ str(self.seed) + '/'

        # set directory for the test result of RL-policy1-t_node_baseline
        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/reinforce/test/old_models/'
        directory_lb_test_hybrid = directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive_t_node_baseline-rlpolicy-treward1/seed' + str(self.seed) + '/'
        # rlactive_t_node_baseline, -rlpolicy/seed' + str(self.seed) + '/',  '/rlactive_t_node_baseline/seed'+ str(self.seed) + '/', '/rlactive_t_node_baseline-rlpolicy/seed' + str(self.seed) + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/' + 'rl/reinforce/test/old_models/'
            directory_lb_test_hybrid_2 = directory_2 + 'evaluation-reinforce4lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(
                total_time_limit) + 's' + test_instance_size + '/rlactive_t_node_baseline-rlpolicy-treward1/seed' + str(self.seed) + '/' # -rlpolicy/seed' + str(self.seed) + '/',  /rlactive_t_node_baseline/seed'+ str(self.seed) + '/'   # 120
        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/' + 'rl/reinforce/test/old_models/'
            directory_lb_test_hybrid_2 = directory_2 + 'evaluation-reinforce4lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(
                total_time_limit) + 's' + test_instance_size + '/rlactive_t_node_baseline-rlpolicy-treward1/seed' + str(self.seed) + '/' # , /rlactive_t_node_baseline/seed'+ str(self.seed) + '/'  #  120


        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/seed'+ str(self.seed) + '/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/seed'+ str(self.seed) + '/'


        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/seed'+ str(self.seed) + '/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/seed'+ str(self.seed) + '/'

        # k_prime trained by data without merge
        directory_lb_test_k_prime = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        # k_prime trained by data with merge
        directory_lb_test_k_prime_merged = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/seed'+ str(self.seed) + '/'
        directory_lb_test_baseline = directory + 'k_prime/' + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/seed'+ str(self.seed) + '/'

        evaluation_directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.incumbent_mode + '/' + 'scip/' + 'heuristic_mode/'
        result_directory_scip = evaluation_directory + 'lb-from-' + self.incumbent_mode + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_scip_baseline/seed' + str(0) + '/'

        primal_int_baselines = []
        primal_int_regressions_merged = []
        primal_int_regressions = []
        primal_int_regression_reinforces = []
        primal_int_reinforces = []
        primal_int_regression_reinforces_talored = []
        primal_int_reinforces_talored = []

        primal_gap_final_baselines = []
        primal_gap_final_regressions = []
        primal_gap_final_regressions_merged = []
        primal_gap_final_regression_reinforces = []
        primal_gap_final_reinforces = []
        primal_gap_final_regression_reinforces_talored = []
        primal_gap_final_reinforces_talored = []

        steplines_baseline = []
        steplines_regression = []
        steplines_regression_merged = []
        steplines_regression_reinforce = []
        steplines_reinforce = []
        steplines_regression_reinforce_talored = []
        steplines_reinforce_talored = []


        pi_steplines_baseline = []
        pi_steplines_regression = []
        pi_steplines_regression_merged = []
        pi_steplines_regression_reinforce = []
        pi_steplines_reinforce = []

        pi_steplines_regression_reinforce_talored = []
        pi_steplines_reinforce_talored = []

        if self.instance_type == instancetypes[3]:
            index_mix = 80
            index_max = 115
        elif self.instance_type == instancetypes[4]:
            index_mix = 0
            index_max = 30

        for i in range(index_mix, index_max):

            if not (self.instance_type == instancetypes[4] and i == 18):

                instance_name = self.instance_type + '-' + str(i) + '_transformed' # instance 100-199

                mip_filename = f'{directory_transformedmodel}{instance_name}.cip'
                mip = Model()
                MIP_model = Model()
                MIP_model.readProblem(mip_filename)
                instance_name = MIP_model.getProbName()

                filename = f'{directory_lb_test}lb-test-{instance_name}.pkl'

                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_reinforce, times_reinforce, objs_regresison_reinforce, times_regression_reinforce = data  # objs contains objs of a single instance of a lb test

                filename = f'{directory_lb_test_2}lb-test-{instance_name}.pkl'

                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_reinforce_2, times_reinforce_2, objs_regresison_reinforce_2, times_regression_reinforce_2 = data  # objs contains objs of a single instance of a lb test

                # load data of hybrid algorithm adapting t_node
                filename = f'{directory_lb_test_hybrid}lb-test-{instance_name}.pkl'

                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_reinforce_hybrid, times_reinforce_hybrid, objs_regresison_reinforce_hybrid, times_regression_reinforce_hybrid = data  # objs contains objs of a single instance of a lb test


                # test from k_prime, SCIP_baseline
                filename = f'{result_directory_scip}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_k_prime, times_k_prime = data  # objs contains objs of a single instance of a lb test

                instance_name = self.instance_type + '-' + str(i) + '_transformed'  # instance 100-199
                # test from k_prime_merged
                filename = f'{directory_lb_test_k_prime_merged}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_k_prime_merged, times_k_prime_merged = data  # objs contains objs of a single instance of a lb test

                # test from baseline
                filename = f'{directory_lb_test_baseline}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs, times = data  # objs contains objs of a single instance of a lb test


                filename = f'{directory_lb_test_k_prime_merged_2}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_k_prime_merged_2, times_k_prime_merged_2 = data  # objs contains objs of a single instance of a lb test

                filename = f'{directory_lb_test_baseline_2}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_2, times_k_2 = data  # objs contains objs of a single instance of a lb test

                objs_reinforce = np.array(objs_reinforce).reshape(-1)
                times_reinforce = np.array(times_reinforce).reshape(-1)
                objs_regresison_reinforce = np.array(objs_regresison_reinforce).reshape(-1)
                times_regression_reinforce = np.array(times_regression_reinforce).reshape(-1)

                objs_reinforce_2 = np.array(objs_reinforce_2).reshape(-1)
                objs_regresison_reinforce_2 = np.array(objs_regresison_reinforce_2).reshape(-1)

                objs_reinforce_hybrid = np.array(objs_reinforce_hybrid).reshape(-1)
                times_reinforce_hybrid = np.array(times_reinforce_hybrid).reshape(-1)
                objs_regresison_reinforce_hybrid = np.array(objs_regresison_reinforce_hybrid).reshape(-1)
                times_regression_reinforce_hybrid = np.array(times_regression_reinforce_hybrid).reshape(-1)


                objs = np.array(objs).reshape(-1)
                times = np.array(times).reshape(-1)

                objs_2 = np.array(objs_2).reshape(-1)

                objs_k_prime = np.array(objs_k_prime).reshape(-1)
                times_k_prime = np.array(times_k_prime).reshape(-1)

                objs_k_prime_merged = np.array(objs_k_prime_merged).reshape(-1)
                times_k_prime_merged = np.array(times_k_prime_merged).reshape(-1)

                objs_k_prime_merged_2 = np.array(objs_k_prime_merged_2).reshape(-1)

                a = [objs_reinforce.min(), objs_regresison_reinforce.min(), objs_reinforce_2.min(), objs_regresison_reinforce_2.min(), objs.min(), objs_2.min(), objs_k_prime_merged.min(), objs_k_prime_merged_2.min(), objs_reinforce_hybrid.min(), objs_regresison_reinforce_hybrid.min(), objs_k_prime.min()] # , objs_reinforce_hybrid_2.min(), objs_regresison_reinforce_hybrid_2.min(),
                obj_opt = np.amin(a)

                # lb-baseline:
                # compute primal gap for baseline localbranching run
                # if times[-1] < total_time_limit:


                primal_int_baseline, primal_gap_final_baseline, stepline_baseline, pi_stepline_baseline = self.compute_primal_integral_2(
                    times=times, objs=objs, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_baselines.append(primal_gap_final_baseline)
                steplines_baseline.append(stepline_baseline)
                primal_int_baselines.append(primal_int_baseline)
                pi_steplines_baseline.append(pi_stepline_baseline)


                # # lb-regression
                # # if times_regression[-1] < total_time_limit:
                #
                primal_int_regression, primal_gap_final_regression, stepline_regression, pi_stepline_regression = self.compute_primal_integral_2(
                    times=times_k_prime, objs=objs_k_prime, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_regressions.append(primal_gap_final_regression)
                steplines_regression.append(stepline_regression)
                primal_int_regressions.append(primal_int_regression)
                pi_steplines_regression.append(pi_stepline_regression)

                # lb-regression-merged
                # if times_regression[-1] < total_time_limit:


                primal_int_regression_merged, primal_gap_final_regression_merged, stepline_regression_merged, pi_stepline_regression_merged = self.compute_primal_integral_2(
                    times=times_k_prime_merged, objs=objs_k_prime_merged, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_regressions_merged.append(primal_gap_final_regression_merged)
                steplines_regression_merged.append(stepline_regression_merged)
                primal_int_regressions_merged.append(primal_int_regression_merged)
                pi_steplines_regression_merged.append(pi_stepline_regression_merged)


                # lb-regression-reinforce


                primal_int_regression_reinforce, primal_gap_final_regression_reinforce, stepline_regression_reinforce, pi_stepline_regression_reinforce = self.compute_primal_integral_2(
                    times=times_regression_reinforce, objs=objs_regresison_reinforce, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_regression_reinforces.append(primal_gap_final_regression_reinforce)
                steplines_regression_reinforce.append(stepline_regression_reinforce)
                primal_int_regression_reinforces.append(primal_int_regression_reinforce)
                pi_steplines_regression_reinforce.append(pi_stepline_regression_reinforce)

                # lb-reinforce


                primal_int_reinforce, primal_gap_final_reinforce, stepline_reinforce, pi_stepline_reinforce = self.compute_primal_integral_2(
                    times=times_reinforce, objs=objs_reinforce, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_reinforces.append(primal_gap_final_reinforce)
                steplines_reinforce.append(stepline_reinforce)
                primal_int_reinforces.append(primal_int_reinforce)
                pi_steplines_reinforce.append(pi_stepline_reinforce)


                primal_int_regression_reinforce_talored, primal_gap_final_regression_reinforce_talored, stepline_regression_reinforce_talored, pi_stepline_regression_reinforce_talored = self.compute_primal_integral_2(
                    times=times_regression_reinforce_hybrid, objs=objs_regresison_reinforce_hybrid, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_regression_reinforces_talored.append(primal_gap_final_regression_reinforce_talored)
                steplines_regression_reinforce_talored.append(stepline_regression_reinforce_talored)
                primal_int_regression_reinforces_talored.append(primal_int_regression_reinforce_talored)
                pi_steplines_regression_reinforce_talored.append(pi_stepline_regression_reinforce_talored)

                # lb-reinforce-tailored


                primal_int_reinforce_talored, primal_gap_final_reinforce_talored, stepline_reinforce_talored, pi_stepline_reinforce_talored = self.compute_primal_integral_2(
                    times=times_reinforce_hybrid, objs=objs_reinforce_hybrid, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_reinforces_talored.append(primal_gap_final_reinforce_talored)
                steplines_reinforce_talored.append(stepline_reinforce_talored)
                primal_int_reinforces_talored.append(primal_int_reinforce_talored)
                pi_steplines_reinforce_talored.append(pi_stepline_reinforce_talored)


        primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_regressions = np.array(primal_int_regressions).reshape(-1)
        primal_int_regressions_merged = np.array(primal_int_regressions_merged).reshape(-1)
        primal_int_regression_reinforces = np.array(primal_int_regression_reinforces).reshape(-1)
        primal_int_reinforces = np.array(primal_int_reinforces).reshape(-1)

        primal_int_regression_reinforces_talored = np.array(primal_int_regression_reinforces_talored).reshape(-1)
        primal_int_reinforces_talored = np.array(primal_int_reinforces_talored).reshape(-1)

        primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_regressions = np.array(primal_gap_final_regressions).reshape(-1)
        primal_gap_final_regressions_merged = np.array(primal_gap_final_regressions_merged).reshape(-1)
        primal_gap_final_regression_reinforces = np.array(primal_gap_final_regression_reinforces).reshape(-1)
        primal_gap_final_reinforces = np.array(primal_gap_final_reinforces).reshape(-1)

        primal_gap_final_regression_reinforces_talored = np.array(primal_gap_final_regression_reinforces_talored).reshape(-1)
        primal_gap_final_reinforces_talored = np.array(primal_gap_final_reinforces_talored).reshape(-1)

        # average primal integral over test dataset
        primal_int_base_ave = mean_shift(primal_int_baselines,
                                         mean_option=mean_option)
        primal_int_regression_ave = mean_shift(primal_int_regressions,
                                         mean_option=mean_option)
        primal_int_regression_merged_ave = mean_shift(primal_int_regressions_merged,
                                         mean_option=mean_option)
        primal_int_regression_reinforce_ave = mean_shift(primal_int_regression_reinforces,
                                         mean_option=mean_option)
        primal_int_reinforce_ave = mean_shift(primal_int_reinforces,
                                         mean_option=mean_option)

        primal_int_regression_reinforce_talored_ave = mean_shift(primal_int_regression_reinforces_talored,
                                         mean_option=mean_option)
        primal_int_reinforce_talored_ave = mean_shift(primal_int_reinforces_talored,
                                         mean_option=mean_option)

        primal_gap_final_baseline_ave = mean_shift(primal_gap_final_baselines,
                                         mean_option=mean_option)
        primal_gap_final_regression_ave = mean_shift(primal_gap_final_regressions,
                                         mean_option=mean_option)
        primal_gap_final_regression_merged_ave = mean_shift(primal_gap_final_regressions_merged,
                                         mean_option=mean_option)
        primal_gap_final_regression_reinforce_ave = mean_shift(primal_gap_final_regression_reinforces,
                                         mean_option=mean_option)
        primal_gap_final_reinforce_ave = mean_shift(primal_gap_final_reinforces,
                                         mean_option=mean_option)

        primal_gap_final_regression_reinforce_talored_ave = mean_shift(primal_gap_final_regression_reinforces_talored,
                                         mean_option=mean_option)
        primal_gap_final_reinforce_talored_ave = mean_shift(primal_gap_final_reinforces_talored,
                                         mean_option=mean_option)

        print(self.instance_type + test_instance_size)
        print(self.incumbent_mode + 'Solution')
        print('baseline primal integral: ', primal_int_base_ave)
        print('scip primal integral: ', primal_int_regression_ave)
        print('regression merged primal integral: ', primal_int_regression_merged_ave)
        print('rl primal integral: ', primal_int_reinforce_ave)
        print('regression-rl primal integral: ', primal_int_regression_reinforce_ave)

        print('rl-hybrid primal integral: ', primal_int_reinforce_talored_ave)
        print('regression-rl-hybrid primal integral: ', primal_int_regression_reinforce_talored_ave)

        print('\n')
        print('baseline primal gap: ', primal_gap_final_baseline_ave)
        print('scip primal gap: ', primal_gap_final_regression_ave)
        print('regression primal merged gap: ', primal_gap_final_regression_merged_ave)
        print('rl primal gap: ', primal_gap_final_reinforce_ave)
        print('regression-rl-hybrid primal gap: ', primal_gap_final_regression_reinforce_ave)

        print('rl-hybrid primal gap: ', primal_gap_final_reinforce_talored_ave)
        print('regression-rl-hybrid primal gap: ', primal_gap_final_regression_reinforce_talored_ave)

        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)

        primalgaps_baseline = None
        for n, stepline_baseline in enumerate(steplines_baseline):
            primal_gap = stepline_baseline(t)
            if n==0:
                primalgaps_baseline = primal_gap
            else:
                primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        primalgap_baseline_ave = mean_shift(primalgaps_baseline, axis=0, mean_option=mean_option)

        primalgaps_regression = None
        for n, stepline_regression in enumerate(steplines_regression):
            primal_gap = stepline_regression(t)
            if n == 0:
                primalgaps_regression = primal_gap
            else:
                primalgaps_regression = np.vstack((primalgaps_regression, primal_gap))
        primalgap_regression_ave = mean_shift(primalgaps_regression, axis=0, mean_option=mean_option)

        primalgaps_regression_merged = None
        for n, stepline_regression in enumerate(steplines_regression_merged):
            primal_gap = stepline_regression(t)
            if n == 0:
                primalgaps_regression_merged = primal_gap
            else:
                primalgaps_regression_merged = np.vstack((primalgaps_regression_merged, primal_gap))
        primalgap_regression_merged_ave = mean_shift(primalgaps_regression_merged, axis=0, mean_option=mean_option)

        primalgaps_regression_reinforce = None
        for n, stepline_regression_reinforce in enumerate(steplines_regression_reinforce):
            primal_gap = stepline_regression_reinforce(t)
            if n == 0:
                primalgaps_regression_reinforce = primal_gap
            else:
                primalgaps_regression_reinforce = np.vstack((primalgaps_regression_reinforce, primal_gap))
        primalgap_regression_reinforce_ave = mean_shift(primalgaps_regression_reinforce, axis=0, mean_option=mean_option)

        primalgaps_reinforce = None
        for n, stepline_reinforce in enumerate(steplines_reinforce):
            primal_gap = stepline_reinforce(t)
            if n == 0:
                primalgaps_reinforce = primal_gap
            else:
                primalgaps_reinforce = np.vstack((primalgaps_reinforce, primal_gap))
        primalgap_reinforce_ave = mean_shift(primalgaps_reinforce, axis=0, mean_option=mean_option)

        primalgaps_regression_reinforce_talored = None
        for n, stepline_regression_reinforce in enumerate(steplines_regression_reinforce_talored):
            primal_gap = stepline_regression_reinforce(t)
            if n == 0:
                primalgaps_regression_reinforce_talored = primal_gap
            else:
                primalgaps_regression_reinforce_talored = np.vstack((primalgaps_regression_reinforce_talored, primal_gap))
        primalgap_regression_reinforce_talored_ave = mean_shift(primalgaps_regression_reinforce_talored, axis=0, mean_option=mean_option)

        primalgaps_reinforce_talored = None
        for n, stepline_reinforce in enumerate(steplines_reinforce_talored):
            primal_gap = stepline_reinforce(t)
            if n == 0:
                primalgaps_reinforce_talored = primal_gap
            else:
                primalgaps_reinforce_talored = np.vstack((primalgaps_reinforce_talored, primal_gap))
        primalgap_reinforce_talored_ave = mean_shift(primalgaps_regression_reinforce_talored, axis=0, mean_option=mean_option)


        pi_baseline = None
        for n, pi_stepline_baseline in enumerate(pi_steplines_baseline):
            pi = pi_stepline_baseline(t)
            if n == 0:
                pi_baseline = pi
            else:
                pi_baseline = np.vstack((pi_baseline, pi))
        pi_baseline_ave = mean_shift(pi_baseline, axis=0, mean_option=mean_option)

        pi_regression = None
        for n, pi_stepline_regression in enumerate(pi_steplines_regression):
            pi = pi_stepline_regression(t)
            if n == 0:
                pi_regression = pi
            else:
                pi_regression = np.vstack((pi_regression, pi))
        pi_regression_ave = mean_shift(pi_regression, axis=0,
                                              mean_option=mean_option)

        pi_regression_merged = None
        for n, pi_stepline_regression_merged in enumerate(pi_steplines_regression_merged):
            pi = pi_stepline_regression_merged(t)
            if n == 0:
                pi_regression_merged = pi
            else:
                pi_regression_merged = np.vstack((pi_regression_merged, pi))
        pi_regression_merged_ave = mean_shift(pi_regression_merged, axis=0, mean_option=mean_option)

        pi_regression_reinforce = None
        for n, pi_stepline_regression_reinforce in enumerate(pi_steplines_regression_reinforce):
            pi = pi_stepline_regression_reinforce(t)
            if n == 0:
                pi_regression_reinforce = pi
            else:
                pi_regression_reinforce = np.vstack((pi_regression_reinforce, pi))
        pi_regression_reinforce_ave = mean_shift(pi_regression_reinforce, axis=0, mean_option=mean_option)

        pi_reinforce = None
        for n, pi_stepline_reinforce in enumerate(pi_steplines_reinforce):
            pi = pi_stepline_reinforce(t)
            if n == 0:
                pi_reinforce = pi
            else:
                pi_reinforce = np.vstack((pi_reinforce, pi))
        pi_reinforce_ave = mean_shift(pi_reinforce, axis=0, mean_option=mean_option)

        pi_regression_reinforce_talored = None
        for n, pi_stepline_regression_reinforce_talored in enumerate(pi_steplines_regression_reinforce_talored):
            pi = pi_stepline_regression_reinforce_talored(t)
            if n == 0:
                pi_regression_reinforce_talored = pi
            else:
                pi_regression_reinforce_talored = np.vstack((pi_regression_reinforce_talored, pi))
        pi_regression_reinforce_talored_ave = mean_shift(pi_regression_reinforce_talored, axis=0, mean_option=mean_option)

        pi_reinforce_talored = None
        for n, pi_stepline_reinforce_talored in enumerate(pi_steplines_reinforce_talored):
            pi = pi_stepline_reinforce_talored(t)
            if n == 0:
                pi_reinforce_talored = pi
            else:
                pi_reinforce_talored = np.vstack((pi_reinforce_talored, pi))
        pi_reinforce_talored_ave = mean_shift(pi_reinforce_talored, axis=0, mean_option=mean_option)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle(self.instance_type + '-' + self.incumbent_mode, fontsize=13)
        ax.plot(t, primalgap_baseline_ave, label='lb-base', color='tab:blue')
        ax.plot(t, primalgap_regression_ave, label='scip', color ='tab:purple')
        ax.plot(t, primalgap_regression_merged_ave, label='lb-srm', color='tab:orange')
        ax.plot(t, primalgap_reinforce_ave, '--', label='lb-rl', color='tab:green')
        ax.plot(t, primalgap_regression_reinforce_ave,'--', label='lb-srm-rl', color='tab:red')
        #
        ax.plot(t, primalgap_reinforce_talored_ave, ':', label='lb-rl-adapt-t', color='tab:green')
        ax.plot(t, primalgap_regression_reinforce_talored_ave, ':', label='lb-srm-rl-adapt-t', color='tab:red')

        ax.set_xlabel('time /s', fontsize=12)
        ax.set_ylabel("scaled primal gap", fontsize=12)
        ax.legend()
        ax.grid()
        plt.savefig('./result/plots/'  + 'plot_primalgap_' + self.instance_type  + '_' + str(test_instance_size) + '_' + self.incumbent_mode + '_hybrid_rlpolicy-tk_enable-tbaseline_t1_'+ 'seed' + str(self.seed) + '_' + mean_option + '_20240418.png')
        plt.show()
        plt.clf()

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle(self.instance_type + '-' + self.incumbent_mode, fontsize=13)
        ax.plot(t, pi_baseline_ave, label='lb-base', color='tab:blue')
        ax.plot(t, pi_regression_ave, label='scip', color='tab:purple')
        ax.plot(t, pi_regression_merged_ave, label='lb-srm', color='tab:orange')
        ax.plot(t, pi_reinforce_ave, '--', label='lb-rl', color='tab:green')
        ax.plot(t, pi_regression_reinforce_ave, '--', label='lb-srm-rl', color='tab:red')
        #
        ax.plot(t, pi_reinforce_talored_ave, ':', label='lb-rl-adapt-t', color='tab:green')
        ax.plot(t, pi_regression_reinforce_talored_ave, ':', label='lb-srm-rl-adapt-t', color='tab:red')

        ax.set_xlabel('time /s', fontsize=12)
        ax.set_ylabel("primal integral", fontsize=12)
        ax.legend()
        ax.grid()
        plt.savefig('./result/plots/' + 'plot_primalintegral_' + self.instance_type + '_' + str(
            test_instance_size) + '_' + self.incumbent_mode + '_hybrid_rlpolicy-tk_enable-tbaseline_t1' + 'seed' + str(self.seed) + '_' + mean_option + '_20240418.png') # _rlpolicy-tk_enable-tbaseline' + ', _hybrid_t_node_baseline.png
        plt.show()
        plt.clf()


