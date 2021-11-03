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
from utility import lbconstraint_modes, instancetypes, incumbent_modes, instancesizes, generator_switcher, binary_support, copy_sol, mean_filter,mean_forward_filter, imitation_accuracy, haming_distance_solutions, haming_distance_solutions_asym
from localbranching import addLBConstraint, addLBConstraintAsymmetric
from ecole_extend.environment_extend import SimpleConfiguring, SimpleConfiguringEnablecuts, SimpleConfiguringEnableheuristics
from models import GraphDataset, GNNPolicy, BipartiteNodeData
import torch.nn.functional as F
import torch_geometric
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from scipy.interpolate import interp1d

from localbranching import LocalBranching

import gc
import sys
from memory_profiler import profile

from models_rl import SimplePolicy, ImitationLbDataset, AgentReinforce
from dataset import InstanceDataset, custom_collate

class MlLocalbranch:
    def __init__(self, instance_type, instance_size, lbconstraint_mode, incumbent_mode='firstsol', seed=100, enable_gpu=False):
        self.instance_type = instance_type
        self.instance_size = instance_size
        self.incumbent_mode = incumbent_mode
        self.lbconstraint_mode = lbconstraint_mode
        self.seed = seed
        self.directory = './result/generated_instances/' + self.instance_type + '/' + self.instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # self.generator = generator_switcher(self.instance_type + self.instance_size)

        self.initialize_ecole_env()
        self.env.seed(self.seed)  # environment (SCIP)
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
            #     self.env = SimpleConfiguringEnableheuristics(
            #
            #         # set up a few SCIP parameters
            #         scip_params={
            #             "presolving/maxrounds": 0,  # deactivate presolving
            #             "presolving/maxrestarts": 0,
            #         },
            #
            #         observation_function=ecole.observation.MilpBipartite(),
            #
            #         reward_function=None,
            #
            #         # collect additional metrics for information purposes
            #         information_function={
            #             'time': ecole.reward.SolvingTime().cumsum(),
            #         }
            #     )

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
        # MIP_model.setParam("limits/solutions", 1)
        MIP_model.setParam("display/verblevel", 0)
        MIP_model.setParam("lp/disablecutoff", 1)

        # MIP_model.setParam("limits/solutions", 1)
        MIP_model.optimize()
        #
        status = MIP_model.getStatus()
        lp_status = MIP_model.getLPSolstat()
        stage = MIP_model.getStage()
        n_sols = MIP_model.getNSols()
        # root_time = MIP_model.getSolvingTime()
        print("* Model status: %s" % status)
        print("* Solve stage: %s" % stage)
        print("* LP status: %s" % lp_status)
        print('* number of sol : ', n_sols)

        sol_lp = MIP_model.createLPSol()
        # sol_relax = MIP_model.createRelaxSol()

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
        # sol_filename = f'{incumbent_mode}-{self.instance_type}-*_transformed.sol'

        test_instances_directory = instances_directory
        instance_test_files = [str(path) for path in
                               sorted(pathlib.Path(test_instances_directory).glob(instance_filename),
                                      key=lambda path: int(
                                          path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        # test_sols_directory = sols_directory + 'test/'
        # sol_test_files = [str(path) for path in sorted(pathlib.Path(test_sols_directory).glob(sol_filename),
        #                                                key=lambda path: int(
        #                                                    path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        # test_dataset = InstanceDataset(mip_files=instance_test_files, sol_files=sol_test_files)

        return instance_test_files

    def compute_primal_integral(self, times, objs, obj_opt, total_time_limit=60):

        # obj_opt = objs.min()
        times = np.append(times, total_time_limit)
        objs = np.append(objs, objs[-1])

        gamma_baseline = np.zeros(len(objs))
        for j in range(len(objs)):
            if objs[j] == 0 and obj_opt == 0:
                gamma_baseline[j] = 0
            elif objs[j] * obj_opt < 0:
                gamma_baseline[j] = 1
            else:
                gamma_baseline[j] = np.abs(objs[j] - obj_opt) / np.maximum(np.abs(objs[j]), np.abs(obj_opt))  #

        # compute the primal gap of last objective
        primal_gap_final = np.abs(objs[-1] - obj_opt) / np.abs(obj_opt) * 100

        # create step line
        stepline = interp1d(times, gamma_baseline, 'previous')


        # compute primal integral
        primal_integral = 0
        for j in range(len(objs) - 1):
            primal_integral += gamma_baseline[j] * (times[j + 1] - times[j])

        return primal_integral, primal_gap_final, stepline

class RegressionInitialK:

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
        # self.generator = generator_switcher(self.instance_type + self.instance_size)

        self.initialize_ecole_env()

        self.env.seed(self.seed)  # environment (SCIP)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            #     self.env = SimpleConfiguringEnableheuristics(
            #
            #         # set up a few SCIP parameters
            #         scip_params={
            #             "presolving/maxrounds": 0,  # deactivate presolving
            #             "presolving/maxrestarts": 0,
            #         },
            #
            #         observation_function=ecole.observation.MilpBipartite(),
            #
            #         reward_function=None,
            #
            #         # collect additional metrics for information purposes
            #         information_function={
            #             'time': ecole.reward.SolvingTime().cumsum(),
            #         }
            #     )

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
            #     heuristics_off = False
            #     cuts_off = True

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

        # print("* Model status: %s" % status)
        # print("* LP status: %s" % lp_status)
        # print("* Solve stage: %s" % stage)
        # print("* Solving time: %s" % t)
        # print('* number of sol : ', n_sols)

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
        # MIP_model.freeTransform()
        status = MIP_model.getStatus()
        print("* Model status: %s" % status)
        MIP_model.resetParams()
        MIP_model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
        MIP_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
        MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
        MIP_model.setIntParam("lp/solvefreq", 0)
        MIP_model.setParam("limits/nodes", 1)
        # MIP_model.setParam("limits/solutions", 1)
        MIP_model.setParam("display/verblevel", 0)
        MIP_model.setParam("lp/disablecutoff", 1)

        MIP_model.setParam("lp/initalgorithm", lp_algo)
        MIP_model.setParam("lp/resolvealgorithm", lp_algo)

        # MIP_model.setParam("limits/solutions", 1)
        MIP_model.optimize()
        #
        status = MIP_model.getStatus()
        lp_status = MIP_model.getLPSolstat()
        stage = MIP_model.getStage()
        n_sols = MIP_model.getNSols()
        # root_time = MIP_model.getSolvingTime()
        print("* Model status: %s" % status)
        print("* Solve stage: %s" % stage)
        print("* LP status: %s" % lp_status)
        print('* number of sol : ', n_sols)

        return MIP_model, lp_status

    def compute_k_prime(self, MIP_model, incumbent):

        # solve the root node and get the LP solution
        # MIP_model.freeTransform()

        # solve the LP relaxation of root node
        MIP_model, lp_status = self.solve_lp(MIP_model)
        if lp_status == 1:
            sol_lp = MIP_model.createLPSol()
            # sol_relax = MIP_model.createRelaxSol()

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


        nsample = 11 # 101
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
            # print("the trivial solution of subMIP is feasible ")
            MIP_copy.addSol(sol_MIP_copy, False)
            # print("the feasible solution of subMIP_model is added to subMIP_model")
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


        # MIP_model.freeProb()
        # del MIP_model

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
                # print("the trivial solution of subMIP is feasible ")
                subMIP_model.addSol(sol_subMIP_model, False)
                # print("the feasible solution of subMIP_model is added to subMIP_model")
            else:
                print("Warning: the trivial solution of subMIP_model is not feasible!")

            # subMIP_model = MIP_copy
            # sol_subMIP_model =  sol_MIP_copy

            # add LB constraint to subMIP model
            alpha = 0.1 * (i)
            # if nsample == 41:
            #     if i<11:
            #         alpha = 0.01*i
            #     elif i<31:
            #         alpha = 0.02*(i-5)
            #     else:
            #         alpha = 0.05*(i-20)

            neigh_size = np.ceil(alpha * k_prime)
            if self.lbconstraint_mode == 'asymmetric':
                # neigh_size = np.ceil(alpha * n_supportbinvars)
                subMIP_model, constraint_lb = addLBConstraintAsymmetric(subMIP_model, sol_subMIP_model, neigh_size)
            else:
                # neigh_size = np.ceil(alpha * n_binvars)
                subMIP_model, constraint_lb = addLBConstraint(subMIP_model, sol_subMIP_model, neigh_size)

            print('Neigh size:', alpha)
            stage = subMIP_model.getStage()
            print("* Solve stage: %s" % stage)

            subMIP_model2 = subMIP_model
            # subMIP_model2, MIP_copy_vars, success = subMIP_model.createCopy(
            #     problemName='Baseline', origcopy=True)

            # subMIP_model2 = subMIP_model
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
            # subMIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.FAST)
            # subMIP_model.setPresolve(pyscipopt.SCIP_PARAMSETTING.FAST)
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

            # subMIP_model2, MIP_copy_vars, success = subMIP_model.createCopy(
            #     problemName='Baseline', origcopy=True)

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

        # t = mean_forward_filter(t,10)
        # objs = mean_forward_filter(objs, 10)

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
        ax[0].set_xlabel(r'$\ r $   ' + '(Neighborhood size: ' + r'$K = r \times N$)') #
        ax[0].set_ylabel("Objective")
        ax[1].plot(neigh_sizes, t)
        # ax[1].set_ylim([0,31])
        ax[1].set_ylabel("Solving time")
        ax[2].plot(neigh_sizes, perf_score)
        ax[2].set_ylabel("Performance score")
        ax[3].plot(neigh_sizes, relax_grips)
        ax[3].set_ylabel("Relaxation grip")
        plt.show()

        # f = self.k_samples_directory + instance_name
        # np.savez(f, neigh_sizes=neigh_sizes, objs=objs, t=t)

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

        # while index_instance < 86:
        #     instance = next(self.generator)
        #     MIP_model = instance.as_pyscipopt()
        #     MIP_model.setProbName(self.instance_type + '-' + str(index_instance))
        #     instance_name = MIP_model.getProbName()
        #     print(instance_name)
        #     index_instance += 1

        while index_instance < 100:
            index_instance = self.sample_k_per_instance(t_limit, index_instance)
            # instance = next(self.generator)
            # MIP_model = instance.as_pyscipopt()
            # MIP_model.setProbName(self.instance_type + '-' + str(index_instance))
            # instance_name = MIP_model.getProbName()
            # print(instance_name)
            #
            # n_vars = MIP_model.getNVars()
            # n_binvars = MIP_model.getNBinVars()
            # print("N of variables: {}".format(n_vars))
            # print("N of binary vars: {}".format(n_binvars))
            # print("N of constraints: {}".format(MIP_model.getNConss()))
            #
            # status, feasible, MIP_model, incumbent_solution = self.initialize_MIP(MIP_model)
            # if (not status == 'optimal') and feasible:
            #     initial_obj = MIP_model.getObjVal()
            #     print("Initial obj before LB: {}".format(initial_obj))
            #     print('Relative gap: ', MIP_model.getGap())
            #
            #     n_supportbinvars = binary_support(MIP_model, incumbent_solution)
            #     print('binary support: ', n_supportbinvars)
            #
            #
            #     MIP_model.resetParams()
            #
            #     neigh_sizes = []
            #     objs = []
            #     t = []
            #     n_supportbins = []
            #     statuss = []
            #     MIP_model.resetParams()
            #     nsample = 101
            #     for i in range(nsample):
            #
            #         # create a copy of the MIP to be 'locally branched'
            #         subMIP_model, subMIP_model_vars, success = MIP_model.createCopy(problemName='subMIPmodelCopy',
            #                                                                       origcopy=False)
            #         sol_subMIP_model = subMIP_model.createSol()
            #
            #         # create a primal solution for the copy MIP by copying the solution of original MIP
            #         n_vars = MIP_model.getNVars()
            #         subMIP_vars = MIP_model.getVars()
            #
            #         for j in range(n_vars):
            #             val = MIP_model.getSolVal(incumbent_solution, subMIP_vars[j])
            #             subMIP_model.setSolVal(sol_subMIP_model, subMIP_model_vars[j], val)
            #         feasible = subMIP_model.checkSol(solution=sol_subMIP_model)
            #
            #         if feasible:
            #             # print("the trivial solution of subMIP is feasible ")
            #             subMIP_model.addSol(sol_subMIP_model, False)
            #             # print("the feasible solution of subMIP_model is added to subMIP_model")
            #         else:
            #             print("Warn: the trivial solution of subMIP_model is not feasible!")
            #
            #         # add LB constraint to subMIP model
            #         alpha = 0.01 * i
            #         # if nsample == 41:
            #         #     if i<11:
            #         #         alpha = 0.01*i
            #         #     elif i<31:
            #         #         alpha = 0.02*(i-5)
            #         #     else:
            #         #         alpha = 0.05*(i-20)
            #
            #         if self.lbconstraint_mode == 'asymmetric':
            #             neigh_size = alpha * n_supportbinvars
            #             subMIP_model = addLBConstraintAsymmetric(subMIP_model, sol_subMIP_model, neigh_size)
            #         else:
            #             neigh_size = alpha * n_binvars
            #             subMIP_model = addLBConstraint(subMIP_model, sol_subMIP_model, neigh_size)
            #
            #         subMIP_model.setParam('limits/time', t_limit)
            #         subMIP_model.optimize()
            #
            #         status = subMIP_model.getStatus()
            #         best_obj = subMIP_model.getSolObjVal(subMIP_model.getBestSol())
            #         solving_time = subMIP_model.getSolvingTime()  # total time used for solving (including presolving) the current problem
            #
            #         best_sol = subMIP_model.getBestSol()
            #
            #         vars_subMIP = subMIP_model.getVars()
            #         n_binvars_subMIP = subMIP_model.getNBinVars()
            #         n_supportbins_subMIP = 0
            #         for i in range(n_binvars_subMIP):
            #             val = subMIP_model.getSolVal(best_sol, vars_subMIP[i])
            #             assert subMIP_model.isFeasIntegral(val), "Error: Value of a binary varialbe is not integral!"
            #             if subMIP_model.isFeasEQ(val, 1.0):
            #                 n_supportbins_subMIP += 1
            #
            #         neigh_sizes.append(alpha)
            #         objs.append(best_obj)
            #         t.append(solving_time)
            #         n_supportbins.append(n_supportbins_subMIP)
            #         statuss.append(status)
            #
            #     for i in range(len(t)):
            #         print('Neighsize: {:.4f}'.format(neigh_sizes[i]),
            #               'Best obj: {:.4f}'.format(objs[i]),
            #               'Binary supports:{}'.format(n_supportbins[i]),
            #               'Solving time: {:.4f}'.format(t[i]),
            #               'Status: {}'.format(statuss[i])
            #               )
            #
            #     neigh_sizes = np.array(neigh_sizes).reshape(-1).astype('float64')
            #     t = np.array(t).reshape(-1)
            #     objs = np.array(objs).reshape(-1)
            #     f = self.k_samples_directory + instance_name
            #     np.savez(f, neigh_sizes=neigh_sizes, objs=objs, t=t)
            #     index_instance += 1

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

            # t = mean_forward_filter(t,10)
            # objs = mean_forward_filter(objs, 10)

            # compute the performance score
            alpha = 1 / 2
            perf_score = alpha * t + (1 - alpha) * objs
            k_bests = k[np.where(perf_score == perf_score.min())]
            k_init = k_bests[0]

            # plt.clf()
            # fig, ax = plt.subplots(3, 1, figsize=(6.4, 6.4))
            # fig.suptitle("Evaluation of size of lb neighborhood")
            # fig.subplots_adjust(top=0.5)
            # ax[0].plot(k, objs)
            # ax[0].set_title(instance_name, loc='right')
            # ax[0].set_xlabel(r'$\ r $   ' + '(Neighborhood size: ' + r'$K = r \times N$)') #
            # ax[0].set_ylabel("Objective")
            # ax[1].plot(k, t)
            # # ax[1].set_ylim([0,31])
            # ax[1].set_ylabel("Solving time")
            # ax[2].plot(k, perf_score)
            # ax[2].set_ylabel("Performance score")
            # plt.show()

            # instance = ecole.scip.Model.from_pyscipopt(MIP_model)
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
        # list_phi_prime_invalid = []
        # list_phi_star_invalid = []
        while index_instance < 100:

            filename = f'{self.directory_transformedmodel}{self.instance_type}-{str(index_instance)}_transformed.cip'
            firstsol_filename = f'{self.directory_sol}{self.incumbent_mode}-{self.instance_type}-{str(index_instance)}_transformed.sol'

            MIP_model = Model()
            MIP_model.readProblem(filename)
            instance_name = MIP_model.getProbName()
            # print(instance_name)
            n_vars = MIP_model.getNVars()
            n_binvars = MIP_model.getNBinVars()
            # print("N of variables: {}".format(n_vars))
            print("N of binary vars: {}".format(n_binvars))
            # print("N of constraints: {}".format(MIP_model.getNConss()))

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

            # t = mean_forward_filter(t,10)
            # objs = mean_forward_filter(objs, 10)

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
            # MIP_model.setParam("limits/solutions", 1)
            MIP_model.setParam("display/verblevel", 0)
            MIP_model.setParam("lp/disablecutoff", 1)

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

            sol_lp = MIP_model.createLPSol()
            # sol_relax = MIP_model.createRelaxSol()

            k_prime = haming_distance_solutions(MIP_model, incumbent, sol_lp)
            # k_lp2relax = haming_distance_solutions(MIP_model, sol_relax, sol_lp)

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
            # phi_lp2relax = k_lp2relax / k_base
            # if phi_lp2relax > 0:
            #     count_phi_lp_relax_diff += 1

            phi_star = k_init
            list_phi_prime.append(phi_prime)
            list_phi_star.append(phi_star)
            # list_phi_lp2relax.append(phi_lp2relax)

            if phi_star <= phi_prime:
                count_phi_star_smaller += 1
            else:
                list_phi_prime_invalid = phi_prime
                list_phi_star_invalid = list_phi_star

            print('instance : ', MIP_model.getProbName())
            print('phi_prime = ', phi_prime)
            print('phi_star = ', phi_star)
            # print('phi_lp2relax = ', phi_lp2relax)
            print('valid count: ', count_phi_star_smaller)
            # print('lp relax diff count:', count_phi_lp_relax_diff)

            # plt.clf()
            # fig, ax = plt.subplots(3, 1, figsize=(6.4, 6.4))
            # fig.suptitle("Evaluation of size of lb neighborhood")
            # fig.subplots_adjust(top=0.5)
            # ax[0].plot(k, objs)
            # ax[0].set_title(instance_name, loc='right')
            # ax[0].set_xlabel(r'$\ r $   ' + '(Neighborhood size: ' + r'$K = r \times N$)') #
            # ax[0].set_ylabel("Objective")
            # ax[1].plot(k, t)
            # # ax[1].set_ylim([0,31])
            # ax[1].set_ylabel("Solving time")
            # ax[2].plot(k, perf_score)
            # ax[2].set_ylabel("Performance score")
            # plt.show()

            # instance = ecole.scip.Model.from_pyscipopt(MIP_model)
            # observation, _, _, done, _ = self.env.reset(instance)
            #
            # data_sample = [observation, k_init]
            # filename = f'{self.regression_samples_directory}regression-{instance_name}.pkl'
            # with gzip.open(filename, 'wb') as f:
            #     pickle.dump(data_sample, f)

            index_instance += 1

        arr_phi_prime = np.array(list_phi_prime).reshape(-1)
        arr_phi_star = np.array(list_phi_star).reshape(-1)
        # arr_phi_lp2relax = np.array(list_phi_lp2relax).reshape(-1)
        ave_phi_prime = arr_phi_prime.sum() / len(arr_phi_prime)
        ave_phi_star = arr_phi_star.sum() / len(arr_phi_star)
        # ave_phi_lp2relax = arr_phi_lp2relax.sum() / len(arr_phi_lp2relax)

        print(self.instance_type + self.instance_size)
        print(self.incumbent_mode + 'Solution')
        print('number of valid phi data points: ', count_phi_star_smaller)
        print('average phi_star :', ave_phi_star )
        print('average phi_prime: ', ave_phi_prime)
        # print('average phi_lp2relax: ', ave_phi_lp2relax)


    def load_dataset(self, dataset_directory=None):

        self.regression_samples_directory = dataset_directory
        filename = 'regression-' + self.instance_type + '-*.pkl'
        # print(filename)
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

        # large_dataset = self.instance_type + "-large"
        # large_directory = './result/generated_instances/' + self.instance_type + '/' + '-large' + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # test_regression_samples_directory = large_directory + 'regression_samples' + '/'
        # train_loader, valid_loader, test_loader = self.load_dataset(dataset_directory=test_regression_samples_directory)
        # train_loaders[large_dataset] = train_loader
        # val_loaders[large_dataset] = valid_loader
        # test_loaders[large_dataset] = test_loader

        model_gnn = GNNPolicy()
        train_dataset = small_dataset
        valid_dataset = small_dataset
        test_dataset = small_dataset
        # LEARNING_RATE = 0.0000001  # setcovering:0.0000005 cap-loc: 0.00000005 independentset: 0.0000001

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

            # torch.save(model_gnn.state_dict(), 'trained_params_' + train_dataset + '.pth')
            # model_gnn2.load_state_dict(torch.load('trained_params_' + train_dataset + '.pth'))

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

        # torch.save(model_gnn.state_dict(),
        #            saved_gnn_directory + 'trained_params_mean_' + train_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '.pth')

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

        # large_dataset = self.instance_type + "-large"
        # large_directory = './result/generated_instances/' + self.instance_type + '/' + '-large' + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # test_regression_samples_directory = large_directory + 'regression_samples' + '/'
        # train_loader, valid_loader, test_loader = self.load_dataset(dataset_directory=test_regression_samples_directory)
        # train_loaders[large_dataset] = train_loader
        # val_loaders[large_dataset] = valid_loader
        # test_loaders[large_dataset] = test_loader

        model_gnn = GNNPolicy()
        train_dataset = small_dataset
        valid_dataset = small_dataset
        test_dataset = small_dataset
        # LEARNING_RATE = 0.0000001  # setcovering:0.0000005 cap-loc: 0.00000005 independentset: 0.0000001

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

            # torch.save(model_gnn.state_dict(), 'trained_params_' + train_dataset + '.pth')
            # model_gnn2.load_state_dict(torch.load('trained_params_' + train_dataset + '.pth'))

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

    # def evaluate_lb_per_instance(self, node_time_limit, total_time_limit, index_instance, reset_k_at_2nditeration=False):
    #     """
    #     evaluate a single MIP instance by two algorithms: lb-baseline and lb-pred_k
    #     :param node_time_limit:
    #     :param total_time_limit:
    #     :param index_instance:
    #     :return:
    #     """
    #     instance = next(self.generator)
    #     MIP_model = instance.as_pyscipopt()
    #     MIP_model.setProbName(self.instance_type + '-' + str(index_instance))
    #     instance_name = MIP_model.getProbName()
    #     print('\n')
    #     print(instance_name)
    #
    #     n_vars = MIP_model.getNVars()
    #     n_binvars = MIP_model.getNBinVars()
    #     print("N of variables: {}".format(n_vars))
    #     print("N of binary vars: {}".format(n_binvars))
    #     print("N of constraints: {}".format(MIP_model.getNConss()))
    #
    #     valid, MIP_model, incumbent_solution = self.initialize_MIP(MIP_model)
    #     conti = -1
    #     # if self.incumbent_mode == 'rootsol' and self.instance_type == 'independentset':
    #     #     conti = 196
    #
    #     if valid:
    #         if index_instance > -1 and index_instance > conti:
    #             gc.collect()
    #             observation, _, _, done, _ = self.env.reset(instance)
    #             del observation
    #             # print(observation)
    #
    #             if self.incumbent_mode == 'firstsol':
    #                 action = {'limits/solutions': 1}
    #             elif self.incumbent_mode == 'rootsol':
    #                 action = {'limits/nodes': 1}  #
    #             sample_observation, _, _, done, _ = self.env.step(action)
    #
    #
    #             # print(sample_observation)
    #             graph = BipartiteNodeData(sample_observation.constraint_features,
    #                                       sample_observation.edge_features.indices,
    #                                       sample_observation.edge_features.values,
    #                                       sample_observation.variable_features)
    #
    #             # We must tell pytorch geometric how many nodes there are, for indexing purposes
    #             graph.num_nodes = sample_observation.constraint_features.shape[0] + \
    #                               sample_observation.variable_features.shape[
    #                                   0]
    #
    #             filename = f'{self.directory_transformedmodel}{self.instance_type}-{str(index_instance)}_transformed.cip'
    #             firstsol_filename = f'{self.directory_sol}{self.incumbent_mode}-{self.instance_type}-{str(index_instance)}_transformed.sol'
    #
    #             model = Model()
    #             model.readProblem(filename)
    #             sol = model.readSolFile(firstsol_filename)
    #
    #             feas = model.checkSol(sol)
    #             try:
    #                 model.addSol(sol, False)
    #             except:
    #                 print('Error: the root solution of ' + model.getProbName() + ' is not feasible!')
    #
    #             instance2 = ecole.scip.Model.from_pyscipopt(model)
    #             observation, _, _, done, _ = self.env.reset(instance2)
    #             graph2 = BipartiteNodeData(observation.constraint_features,
    #                                       observation.edge_features.indices,
    #                                       observation.edge_features.values,
    #                                       observation.variable_features)
    #
    #             # We must tell pytorch geometric how many nodes there are, for indexing purposes
    #             graph2.num_nodes = observation.constraint_features.shape[0] + \
    #                               observation.variable_features.shape[
    #                                   0]
    #
    #             # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
    #             # MIP_model = instance
    #
    #             # MIP_model.optimize()
    #             # print("Status:", MIP_model.getStatus())
    #             # print("best obj: ", MIP_model.getObjVal())
    #             # print("Solving time: ", MIP_model.getSolvingTime())
    #
    #             initial_obj = MIP_model.getSolObjVal(incumbent_solution)
    #             print("Initial obj before LB: {}".format(initial_obj))
    #
    #             binary_supports = binary_support(MIP_model, incumbent_solution)
    #             print('binary support: ', binary_supports)
    #
    #             model_gnn = GNNPolicy()
    #
    #             model_gnn.load_state_dict(torch.load(
    #                 self.saved_gnn_directory + 'trained_params_mean_' + self.train_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '.pth'))
    #
    #             # model_gnn.load_state_dict(torch.load(
    #             #      'trained_params_' + self.instance_type + '.pth'))
    #
    #             k_model = model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
    #                                 graph.variable_features)
    #
    #             k_pred = k_model.item() * n_binvars
    #             print('GNN prediction: ', k_model.item())
    #
    #             k_model2 = model_gnn(graph2.constraint_features, graph2.edge_index, graph2.edge_attr,
    #                                 graph2.variable_features)
    #
    #             print('GNN prediction of model2: ', k_model2.item())
    #
    #             if self.is_symmetric == False:
    #                 k_pred = k_model.item() * binary_supports
    #
    #             del k_model
    #             del graph
    #             del sample_observation
    #             del model_gnn
    #
    #             # create a copy of MIP
    #             MIP_model.resetParams()
    #             MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
    #                 problemName='Baseline', origcopy=False)
    #             MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
    #                 problemName='GNN',
    #                 origcopy=False)
    #             MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
    #                 problemName='GNN+reset',
    #                 origcopy=False)
    #
    #             print('MIP copies are created')
    #
    #             MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent_solution,
    #                                                     MIP_copy_vars)
    #             MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent_solution,
    #                                                       MIP_copy_vars2)
    #             MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent_solution,
    #                                                       MIP_copy_vars3)
    #
    #             print('incumbent solution is copied to MIP copies')
    #             MIP_model.freeProb()
    #             del MIP_model
    #             del incumbent_solution
    #
    #             # sol = MIP_model_copy.getBestSol()
    #             # initial_obj = MIP_model_copy.getSolObjVal(sol)
    #             # print("Initial obj before LB: {}".format(initial_obj))
    #
    #             # execute local branching baseline heuristic by Fischetti and Lodi
    #             lb_model = LocalBranching(MIP_model=MIP_model_copy, MIP_sol_bar=sol_MIP_copy, k=self.k_baseline,
    #                                       node_time_limit=node_time_limit,
    #                                       total_time_limit=total_time_limit)
    #             status, obj_best, elapsed_time, lb_bits, times, objs = lb_model.search_localbranch(is_symmetric=self.is_symmetric,
    #                                                                          reset_k_at_2nditeration=False)
    #             print("Instance:", MIP_model_copy.getProbName())
    #             print("Status of LB: ", status)
    #             print("Best obj of LB: ", obj_best)
    #             print("Solving time: ", elapsed_time)
    #             print('\n')
    #
    #             MIP_model_copy.freeProb()
    #             del sol_MIP_copy
    #             del MIP_model_copy
    #
    #             # sol = MIP_model_copy2.getBestSol()
    #             # initial_obj = MIP_model_copy2.getSolObjVal(sol)
    #             # print("Initial obj before LB: {}".format(initial_obj))
    #
    #             # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
    #             lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=k_pred,
    #                                        node_time_limit=node_time_limit,
    #                                        total_time_limit=total_time_limit)
    #             status, obj_best, elapsed_time, lb_bits_pred_reset, times_pred_rest, objs_pred_rest = lb_model3.search_localbranch(is_symmetric=self.is_symmetric,
    #                                                                           reset_k_at_2nditeration=reset_k_at_2nditeration)
    #
    #             print("Instance:", MIP_model_copy3.getProbName())
    #             print("Status of LB: ", status)
    #             print("Best obj of LB: ", obj_best)
    #             print("Solving time: ", elapsed_time)
    #             print('\n')
    #
    #             MIP_model_copy3.freeProb()
    #             del sol_MIP_copy3
    #             del MIP_model_copy3
    #
    #             # execute local branching with 1. first k predicted by GNN; 2. from 2nd iteration of lb, continue lb algorithm with no further injection
    #             lb_model2 = LocalBranching(MIP_model=MIP_model_copy2, MIP_sol_bar=sol_MIP_copy2, k=k_pred,
    #                                        node_time_limit=node_time_limit,
    #                                        total_time_limit=total_time_limit)
    #             status, obj_best, elapsed_time, lb_bits_pred, times_pred, objs_pred = lb_model2.search_localbranch(is_symmetric=self.is_symmetric,
    #                                                                           reset_k_at_2nditeration=False)
    #
    #             print("Instance:", MIP_model_copy2.getProbName())
    #             print("Status of LB: ", status)
    #             print("Best obj of LB: ", obj_best)
    #             print("Solving time: ", elapsed_time)
    #             print('\n')
    #
    #             MIP_model_copy2.freeProb()
    #             del sol_MIP_copy2
    #             del MIP_model_copy2
    #
    #             data = [objs, times, objs_pred, times_pred, objs_pred_rest, times_pred_rest]
    #             filename = f'{self.directory_lb_test}lb-test-{instance_name}.pkl'  # instance 100-199
    #             with gzip.open(filename, 'wb') as f:
    #                 pickle.dump(data, f)
    #             del data
    #             del objs
    #             del times
    #             del objs_pred
    #             del times_pred
    #             del objs_pred_rest
    #             del times_pred_rest
    #             del lb_model
    #             del lb_model2
    #             del lb_model3
    #
    #         index_instance += 1
    #     del instance
    #     return index_instance
    #
    # def evaluate_localbranching(self, test_instance_size='-small', train_instance_size='-small', total_time_limit=60, node_time_limit=30, reset_k_at_2nditeration=False):
    #
    #     self.train_dataset = self.instance_type + train_instance_size
    #     self.evaluation_dataset = self.instance_type + test_instance_size
    #
    #     self.generator = generator_switcher(self.evaluation_dataset)
    #     self.generator.seed(self.seed)
    #
    #     direc = './data/generated_instances/' + self.instance_type + '/' + test_instance_size + '/'
    #     self.directory_transformedmodel = direc + 'transformedmodel' + '/'
    #     self.directory_sol = direc + self.incumbent_mode + '/'
    #
    #     self.k_baseline = 20
    #
    #     self.is_symmetric = True
    #     if self.lbconstraint_mode == 'asymmetric':
    #         self.is_symmetric = False
    #         self.k_baseline = self.k_baseline / 2
    #     total_time_limit = total_time_limit
    #     node_time_limit = node_time_limit
    #
    #     self.saved_gnn_directory = './result/saved_models/'
    #
    #     directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
    #     self.directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
    #     pathlib.Path(self.directory_lb_test).mkdir(parents=True, exist_ok=True)
    #
    #     index_instance = 0
    #     while index_instance < 200:
    #         index_instance = self.evaluate_lb_per_instance(node_time_limit=node_time_limit, total_time_limit=total_time_limit, index_instance=index_instance, reset_k_at_2nditeration=reset_k_at_2nditeration)

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


        instance = ecole.scip.Model.from_pyscipopt(MIP_model)
        observation, _, _, done, _ = self.env.reset(instance)
        graph = BipartiteNodeData(observation.constraint_features,
                                   observation.edge_features.indices,
                                   observation.edge_features.values,
                                   observation.variable_features)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = observation.constraint_features.shape[0] + \
                           observation.variable_features.shape[
                               0]

        # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
        # MIP_model = instance

        # MIP_model.optimize()
        # print("Status:", MIP_model.getStatus())
        # print("best obj: ", MIP_model.getObjVal())
        # print("Solving time: ", MIP_model.getSolvingTime())

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

        # sol = MIP_model_copy.getBestSol()
        # initial_obj = MIP_model_copy.getSolObjVal(sol)
        # print("Initial obj before LB: {}".format(initial_obj))

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

        # sol = MIP_model_copy2.getBestSol()
        # initial_obj = MIP_model_copy2.getSolObjVal(sol)
        # print("Initial obj before LB: {}".format(initial_obj))

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
        graph = BipartiteNodeData(observation.constraint_features,
                                   observation.edge_features.indices,
                                   observation.edge_features.values,
                                   observation.variable_features)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = observation.constraint_features.shape[0] + \
                           observation.variable_features.shape[
                               0]

        # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
        # MIP_model = instance

        # MIP_model.optimize()
        # print("Status:", MIP_model.getStatus())
        # print("best obj: ", MIP_model.getObjVal())
        # print("Solving time: ", MIP_model.getSolvingTime())

        # create a copy of MIP
        MIP_model.resetParams()
        MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
            problemName='Baseline', origcopy=False)
        # MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
        #     problemName='GNN',
        #     origcopy=False)
        MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
            problemName='GNN+reset',
            origcopy=False)

        print('MIP copies are created')

        MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent,
                                                MIP_copy_vars)
        # MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent,
        #                                           MIP_copy_vars2)
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

        # sol = MIP_model_copy.getBestSol()
        # initial_obj = MIP_model_copy.getSolObjVal(sol)
        # print("Initial obj before LB: {}".format(initial_obj))

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

        # sol = MIP_model_copy2.getBestSol()
        # initial_obj = MIP_model_copy2.getSolObjVal(sol)
        # print("Initial obj before LB: {}".format(initial_obj))

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

        # # execute local branching with 1. first k predicted by GNN; 2. from 2nd iteration of lb, continue lb algorithm with no further injection
        #
        # lb_model2 = LocalBranching(MIP_model=MIP_model_copy2, MIP_sol_bar=sol_MIP_copy2, k=k_pred,
        #                            node_time_limit=node_time_limit,
        #                            total_time_limit=total_time_limit)
        # status, obj_best, elapsed_time, lb_bits_regression_noreset, times_regression_noreset, objs_regression_noreset, _, _ = lb_model2.mdp_localbranch(
        #     is_symmetric=self.is_symmetric,
        #     reset_k_at_2nditeration=False,
        #     policy=None,
        #     optimizer=None,
        #     device=device
        # )
        #
        # print("Instance:", MIP_model_copy2.getProbName())
        # print("Status of LB: ", status)
        # print("Best obj of LB: ", obj_best)
        # print("Solving time: ", elapsed_time)
        # print('\n')
        #
        # MIP_model_copy2.freeProb()
        # del sol_MIP_copy2
        # del MIP_model_copy2

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

            # filename_2 = f'{directory_lb_test_2}lb-test-{instance_name}.pkl'
            #
            # with gzip.open(filename_2, 'rb') as f:
            #     data = pickle.load(f)
            # objs_2, times_2, objs_pred_2, times_pred_2, objs_pred_reset_2, times_pred_reset_2 = data  # objs contains objs of a single instance of a lb test

            a = [objs.min(), objs_pred.min(), objs_pred_reset.min()] # objs_2.min(), objs_pred_2.min(), objs_pred_reset_2.min()
            # a = [objs.min(), objs_pred.min(), objs_pred_reset.min()]
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
                    gamma_baseline[j] = np.abs(objs[j] - obj_opt) / np.maximum(np.abs(objs[j]), np.abs(obj_opt)) #

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
                    gamma_pred[j] = np.abs(objs_pred[j] - obj_opt) / np.maximum(np.abs(objs_pred[j]), np.abs(obj_opt)) #

            primal_gap_final_pred = np.abs(objs_pred[-1] - obj_opt) / np.abs(obj_opt)
            primal_gap_final_preds.append(primal_gap_final_pred)

            stepline_pred = interp1d(times_pred, gamma_pred, 'previous')
            steplines_pred.append(stepline_pred)

            #
            # t = np.linspace(start=0.0, stop=total_time_limit, num=1001)
            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of primal gap")
            # fig.subplots_adjust(top=0.5)
            # # ax.set_title(instance_name, loc='right')
            # ax.plot(t, stepline_baseline(t), label='lb baseline')
            # ax.plot(t, stepline_pred(t), label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()

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
                    gamma_pred_reset[j] = np.abs(objs_pred_reset[j] - obj_opt) / np.maximum(np.abs(objs_pred_reset[j]), np.abs(obj_opt)) #

            primal_gap_final_pred_reset = np.abs(objs_pred_reset[-1] - obj_opt) / np.abs(obj_opt)
            primal_gap_final_preds_reset.append(primal_gap_final_pred_reset)

            stepline_pred_reset = interp1d(times_pred_reset, gamma_pred_reset, 'previous')
            steplines_pred_reset.append(stepline_pred_reset)

            # compute primal interal
            primal_int_pred_reset = 0
            for j in range(len(objs_pred_reset) - 1):
                primal_int_pred_reset += gamma_pred_reset[j] * (times_pred_reset[j + 1] - times_pred_reset[j])
            primal_int_preds_reset.append(primal_int_pred_reset)

            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of objective")
            # fig.subplots_adjust(top=0.5)
            # ax.set_title(instance_name, loc='right')
            # ax.plot(times, objs, label='lb baseline')
            # ax.plot(times_pred, objs_pred, label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()
            #
            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of primal gap")
            # fig.subplots_adjust(top=0.5)
            # ax.set_title(instance_name, loc='right')
            # ax.plot(times, gamma_baseline, label='lb baseline')
            # ax.plot(times_pred, gamma_pred, label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()


        primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_preds = np.array(primal_int_preds).reshape(-1)
        primal_int_preds_reset = np.array(primal_int_preds_reset).reshape(-1)

        primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_preds = np.array(primal_gap_final_preds).reshape(-1)
        primal_gap_final_preds_reset = np.array(primal_gap_final_preds_reset).reshape(-1)

        # avarage primal integral over test dataset
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
        # fig.subplots_adjust(top=0.5)
        ax.set_title(self.instance_type + '-' + self.incumbent_mode, loc='right')
        ax.plot(t, primalgap_baseline_ave, label='lb-baseline')
        ax.plot(t, primalgap_pred_ave, label='lb-gnn')
        ax.plot(t, primalgap_pred_ave_reset, '--', label='lb-gnn-reset')
        ax.set_xlabel('time /s')
        ax.set_ylabel("normalized primal gap")
        ax.legend()
        plt.show()

class RegressionInitialK_KPrime(MlLocalbranch):

    def __init__(self, instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed=100):
        super().__init__(instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed)

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
        # self.generator = generator_switcher(self.instance_type + self.instance_size)

    def sample_k_per_instance_k_prime(self, t_limit, index_instance):

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
        n_nodes = []
        firstlp_times = []
        n_lps = []
        presolve_times = []

        nsample = 101  # 101
        if self.instance_type == instancetypes[2]:
            nsample = 60
        if self.instance_type == instancetypes[1] and self.incumbent_mode == incumbent_modes[1]:
            nsample = 60

        # # create a copy of the MIP to be 'locally branched'
        # MIP_copy, subMIP_model_vars, success = MIP_model.createCopy(problemName='MIPCopy',
        #                                                            origcopy=False)
        # MIP_copy.resetParams()
        # sol_MIP_copy = MIP_copy.createSol()
        # 
        # # create a primal solution for the copy MIP by copying the solution of original MIP
        # n_vars = MIP_model.getNVars()
        # subMIP_vars = MIP_model.getVars()
        # 
        # for j in range(n_vars):
        #     val = MIP_model.getSolVal(incumbent, subMIP_vars[j])
        #     MIP_copy.setSolVal(sol_MIP_copy, subMIP_model_vars[j], val)
        # feasible = MIP_copy.checkSol(solution=sol_MIP_copy)
        # 
        # if feasible:
        #     # print("the trivial solution of subMIP is feasible ")
        #     MIP_copy.addSol(sol_MIP_copy, False)
        #     # print("the feasible solution of subMIP_model is added to subMIP_model")
        # else:
        #     print("Warn: the trivial solution of subMIP_model is not feasible!")
        # 
        # n_supportbinvars = binary_support(MIP_copy, sol_MIP_copy)
        # print('binary support: ', n_supportbinvars)
        # print('Number of solutions: ', MIP_copy.getNSols())

        k_base = n_binvars
        if self.is_symmetric == False:
            k_base = n_supportbinvars
        # solve the root node and get the LP solution
        k_prime = self.compute_k_prime(MIP_model, incumbent)
        phi_prime = k_prime / k_base
        print('phi_prime :', phi_prime)

        MIP_model.freeProb()
        # del MIP_model

        for i in range(nsample):

            # # create a copy of the MIP to be 'locally branched'
            # subMIP_model, subMIP_model_vars, success = MIP_copy.createCopy(problemName='MIPCopy',
            #                                                              origcopy=False)
            # subMIP_model.resetParams()
            # sol_subMIP_model = subMIP_model.createSol()
            #
            # # create a primal solution for the copy MIP by copying the solution of original MIP
            # n_vars = MIP_copy.getNVars()
            # MIP_copy_vars = MIP_copy.getVars()
            #
            # for j in range(n_vars):
            #     val = MIP_copy.getSolVal(sol_MIP_copy, MIP_copy_vars[j])
            #     subMIP_model.setSolVal(sol_subMIP_model, subMIP_model_vars[j], val)
            # feasible = subMIP_model.checkSol(solution=sol_subMIP_model)
            #
            # if feasible:
            #     # print("the trivial solution of subMIP is feasible ")
            #     subMIP_model.addSol(sol_subMIP_model, False)
            #     # print("the feasible solution of subMIP_model is added to subMIP_model")
            # else:
            #     print("Warn: the trivial solution of subMIP_model is not feasible!")

            subMIP_model = MIP_model
            # sol_subMIP_model = sol_MIP_copy


            subMIP_model.readProblem(filename)
            sol_subMIP_model = subMIP_model.readSolFile(sol_filename)
            feas = subMIP_model.checkSol(sol_subMIP_model)
            try:
                subMIP_model.addSol(sol_subMIP_model, False)
            except:
                print('Error: the root solution of ' + instance_name + ' is not feasible!')

            # add LB constraint to subMIP model
            alpha = 0.01 * (i)
            # if nsample == 41:
            #     if i<11:
            #         alpha = 0.01*i
            #     elif i<31:
            #         alpha = 0.02*(i-5)
            #     else:
            #         alpha = 0.05*(i-20)

            # neigh_size = np.ceil(alpha * k_base)
            if self.lbconstraint_mode == 'asymmetric':
                neigh_size = np.ceil(alpha * k_prime)
                subMIP_model, constraint_lb = addLBConstraintAsymmetric(subMIP_model, sol_subMIP_model, neigh_size)
            else:
                neigh_size = np.ceil(alpha * k_prime)
                subMIP_model, constraint_lb = addLBConstraint(subMIP_model, sol_subMIP_model, neigh_size)

            print('Neigh size:', alpha)
            # stage = subMIP_model.getStage()
            # print("* subMIP stage before solving: %s" % stage)

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

            # # subMIP_model2, MIP_copy_vars, success = subMIP_model.createCopy(
            # #     problemName='Baseline', origcopy=True)
            # subMIP_model.freeTransform()
            # subMIP_model2 = subMIP_model
            #
            # subMIP_model2.resetParams()
            # subMIP_model2.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
            # subMIP_model2.setParam('presolving/maxrounds', 0)
            # subMIP_model2.setParam('presolving/maxrestarts', 0)
            #
            # subMIP_model2.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
            # subMIP_model2.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
            # subMIP_model2.setIntParam("lp/solvefreq", 0)
            # subMIP_model2.setParam("limits/nodes", 1)
            # subMIP_model2.setParam('limits/time', 30)
            # # subMIP_model2.setParam("limits/solutions", 1)
            # subMIP_model2.setParam("display/verblevel", 0)
            #
            # subMIP_model2.setParam("lp/disablecutoff", 1)
            #
            # stage = subMIP_model2.getStage()
            # n_sols = subMIP_model2.getNSols()
            # print('* number of sol : ', n_sols)
            # print("* Solve stage: %s" % stage)
            #
            # # subMIP_model2.setParam("limits/solutions", 1)
            # subMIP_model2.optimize()
            #
            # status = subMIP_model2.getStatus()
            # lp_status = subMIP_model2.getLPSolstat()
            # stage = subMIP_model2.getStage()
            # n_sols = subMIP_model2.getNSols()
            # time = subMIP_model2.getSolvingTime()
            # print("* Model status: %s" % status)
            # print("* Solve stage: %s" % stage)
            # print("* LP status: %s" % lp_status)
            # print('* number of sol : ', n_sols)
            #
            # n_bins = subMIP_model2.getNBinVars()
            #
            # lpcands, lpcandssol, lpcadsfrac, nlpcands, npriolpcands, nfracimplvars = subMIP_model2.getLPBranchCands()
            # relax_grip = 1 - nlpcands / n_bins
            #
            #
            # # print('binvars :', n_bins)
            # # print('nbranchingcands :', nlpcands)
            # # print('nfracimplintvars :', nfracimplvars)
            # print('relaxation grip :', 1 - nlpcands / n_bins)

            neigh_sizes.append(alpha)
            objs.append(best_obj)
            t.append(solving_time)
            n_supportbins.append(n_supportbins_subMIP)
            statuss.append(status_subMIP)

            # relax_grips.append(relax_grip)
            # n_nodes.append(n_node)
            # firstlp_times.append(firstlp_time)
            # presolve_times.append(presolve_time)
            # n_lps.append(n_lp)


            subMIP_model.freeTransform()
            subMIP_model.resetParams()
            subMIP_model.delCons(constraint_lb)
            subMIP_model.releasePyCons(constraint_lb)
            del constraint_lb
            print('Number of solutions: ', subMIP_model.getNSols())
            subMIP_model.freeProb()
            # print('Number of solutions: ', subMIP_model.getNSols())
            # del subMIP_model


        for i in range(len(t)):
            print('Neighsize: {:.4f}'.format(neigh_sizes[i]),
                  'Best obj: {:.4f}'.format(objs[i]),
                  'Binary supports:{}'.format(n_supportbins[i]),
                  'Solving time: {:.4f}'.format(t[i]),
                  # 'Presolve_time: {:.4f}'.format(presolve_times[i]),
                  # 'FirstLP time: {:.4f}'.format(firstlp_times[i]),
                  # 'solved LPs: {:.4f}'.format(n_lps[i]),
                  # 'B&B nodes: {:.4f}'.format(n_nodes[i]),
                  # 'Relaxation grip: {:.4f}'.format(relax_grips[i]),
                  'Status: {}'.format(statuss[i])
                  )

        neigh_sizes = np.array(neigh_sizes).reshape(-1)
        t = np.array(t).reshape(-1)
        objs = np.array(objs).reshape(-1)
        # relax_grips = np.array(relax_grips).reshape(-1)

        saved_name = f'{self.instance_type}-{str(index_instance)}_transformed'
        f = self.k_samples_directory + saved_name
        np.savez(f, neigh_sizes=neigh_sizes, objs=objs, t=t)

        # normalize the objective and solving time
        t = t / t_limit
        objs_abs = objs
        objs = (objs_abs - np.min(objs_abs))
        objs = objs / np.max(objs)

        t = mean_filter(t, 5)
        objs = mean_filter(objs, 5)

        # t = mean_forward_filter(t,10)
        # objs = mean_forward_filter(objs, 10)

        # compute the performance score
        alpha = 1 / 2
        perf_score = alpha * t + (1 - alpha) * objs
        phi_bests = neigh_sizes[np.where(perf_score == perf_score.min())]
        phi_init = phi_bests[0]
        if phi_init > self.phi_max:
            self.phi_max = phi_init
        print('phi_0_star:', phi_init)
        print('phi_0_max:', self.phi_max)

        # plt.clf()
        # fig, ax = plt.subplots(3, 1, figsize=(6.4, 6.4))
        # fig.suptitle("Evaluation of size of lb neighborhood")
        # fig.subplots_adjust(top=0.5)
        # ax[0].plot(neigh_sizes, objs)
        # ax[0].set_title(instance_name, loc='right')
        # ax[0].set_xlabel(r'$\ r $   ' + '(Neighborhood size: ' + r'$K = r \times N$)') #
        # ax[0].set_ylabel("Objective")
        # ax[1].plot(neigh_sizes, t)
        # # ax[1].set_ylim([0,31])
        # ax[1].set_ylabel("Solving time")
        # ax[2].plot(neigh_sizes, perf_score)
        # ax[2].set_ylabel("Cost")
        # # ax[3].plot(neigh_sizes, relax_grips)
        # # ax[3].set_ylabel("Relaxation grip")
        # plt.show()

        index_instance += 1

        return index_instance

    def generate_k_samples_k_prime(self, t_limit, instance_size='-small'):
        """
        For each MIP instance, sample k from [0,1] * n_binary(symmetric) or [0,1] * n_binary_support(asymmetric),
        and evaluate the performance of 1st round of local-branching
        :param t_limit:
        :param k_samples_directory:
        :return:
        """

        self.k_samples_directory = self.directory + 'k_samples_k_prime' + '/'
        pathlib.Path(self.k_samples_directory).mkdir(parents=True, exist_ok=True)

        direc = './data/generated_instances/' + self.instance_type + '/' + instance_size + '/'
        self.directory_transformedmodel = direc + 'transformedmodel' + '/'
        self.directory_sol = direc + self.incumbent_mode + '/'

        index_instance = 100
        self.phi_max = 0

        # while index_instance < 86:
        #     instance = next(self.generator)
        #     MIP_model = instance.as_pyscipopt()
        #     MIP_model.setProbName(self.instance_type + '-' + str(index_instance))
        #     instance_name = MIP_model.getProbName()
        #     print(instance_name)
        #     index_instance += 1

        while index_instance < 200:
            index_instance = self.sample_k_per_instance_k_prime(t_limit, index_instance)
            # instance = next(self.generator)
            # MIP_model = instance.as_pyscipopt()
            # MIP_model.setProbName(self.instance_type + '-' + str(index_instance))
            # instance_name = MIP_model.getProbName()
            # print(instance_name)
            #
            # n_vars = MIP_model.getNVars()
            # n_binvars = MIP_model.getNBinVars()
            # print("N of variables: {}".format(n_vars))
            # print("N of binary vars: {}".format(n_binvars))
            # print("N of constraints: {}".format(MIP_model.getNConss()))
            #
            # status, feasible, MIP_model, incumbent_solution = self.initialize_MIP(MIP_model)
            # if (not status == 'optimal') and feasible:
            #     initial_obj = MIP_model.getObjVal()
            #     print("Initial obj before LB: {}".format(initial_obj))
            #     print('Relative gap: ', MIP_model.getGap())
            #
            #     n_supportbinvars = binary_support(MIP_model, incumbent_solution)
            #     print('binary support: ', n_supportbinvars)
            #
            #
            #     MIP_model.resetParams()
            #
            #     neigh_sizes = []
            #     objs = []
            #     t = []
            #     n_supportbins = []
            #     statuss = []
            #     MIP_model.resetParams()
            #     nsample = 101
            #     for i in range(nsample):
            #
            #         # create a copy of the MIP to be 'locally branched'
            #         subMIP_model, subMIP_model_vars, success = MIP_model.createCopy(problemName='subMIPmodelCopy',
            #                                                                       origcopy=False)
            #         sol_subMIP_model = subMIP_model.createSol()
            #
            #         # create a primal solution for the copy MIP by copying the solution of original MIP
            #         n_vars = MIP_model.getNVars()
            #         subMIP_vars = MIP_model.getVars()
            #
            #         for j in range(n_vars):
            #             val = MIP_model.getSolVal(incumbent_solution, subMIP_vars[j])
            #             subMIP_model.setSolVal(sol_subMIP_model, subMIP_model_vars[j], val)
            #         feasible = subMIP_model.checkSol(solution=sol_subMIP_model)
            #
            #         if feasible:
            #             # print("the trivial solution of subMIP is feasible ")
            #             subMIP_model.addSol(sol_subMIP_model, False)
            #             # print("the feasible solution of subMIP_model is added to subMIP_model")
            #         else:
            #             print("Warn: the trivial solution of subMIP_model is not feasible!")
            #
            #         # add LB constraint to subMIP model
            #         alpha = 0.01 * i
            #         # if nsample == 41:
            #         #     if i<11:
            #         #         alpha = 0.01*i
            #         #     elif i<31:
            #         #         alpha = 0.02*(i-5)
            #         #     else:
            #         #         alpha = 0.05*(i-20)
            #
            #         if self.lbconstraint_mode == 'asymmetric':
            #             neigh_size = alpha * n_supportbinvars
            #             subMIP_model = addLBConstraintAsymmetric(subMIP_model, sol_subMIP_model, neigh_size)
            #         else:
            #             neigh_size = alpha * n_binvars
            #             subMIP_model = addLBConstraint(subMIP_model, sol_subMIP_model, neigh_size)
            #
            #         subMIP_model.setParam('limits/time', t_limit)
            #         subMIP_model.optimize()
            #
            #         status = subMIP_model.getStatus()
            #         best_obj = subMIP_model.getSolObjVal(subMIP_model.getBestSol())
            #         solving_time = subMIP_model.getSolvingTime()  # total time used for solving (including presolving) the current problem
            #
            #         best_sol = subMIP_model.getBestSol()
            #
            #         vars_subMIP = subMIP_model.getVars()
            #         n_binvars_subMIP = subMIP_model.getNBinVars()
            #         n_supportbins_subMIP = 0
            #         for i in range(n_binvars_subMIP):
            #             val = subMIP_model.getSolVal(best_sol, vars_subMIP[i])
            #             assert subMIP_model.isFeasIntegral(val), "Error: Value of a binary varialbe is not integral!"
            #             if subMIP_model.isFeasEQ(val, 1.0):
            #                 n_supportbins_subMIP += 1
            #
            #         neigh_sizes.append(alpha)
            #         objs.append(best_obj)
            #         t.append(solving_time)
            #         n_supportbins.append(n_supportbins_subMIP)
            #         statuss.append(status)
            #
            #     for i in range(len(t)):
            #         print('Neighsize: {:.4f}'.format(neigh_sizes[i]),
            #               'Best obj: {:.4f}'.format(objs[i]),
            #               'Binary supports:{}'.format(n_supportbins[i]),
            #               'Solving time: {:.4f}'.format(t[i]),
            #               'Status: {}'.format(statuss[i])
            #               )
            #
            #     neigh_sizes = np.array(neigh_sizes).reshape(-1).astype('float64')
            #     t = np.array(t).reshape(-1)
            #     objs = np.array(objs).reshape(-1)
            #     f = self.k_samples_directory + instance_name
            #     np.savez(f, neigh_sizes=neigh_sizes, objs=objs, t=t)
            #     index_instance += 1

    def two_examples(self):

        plt.clf()
        fig, ax = plt.subplots(2, 1, figsize=(6.0, 4.0))
        # fig.suptitle("Evaluation of size of lb neighborhood")
        # fig.subplots_adjust(top=0.5)
        ax[0].set_xlabel(r'$\ r $   ' + '(neighborhood size: ' + r'$k = r \times N$)')  #
        ax[0].set_ylabel("Objective")
        ax[1].set_ylabel("Solving time")
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

            # t = mean_forward_filter(t,10)
            # objs = mean_forward_filter(objs, 10)

            # compute the performance score
            alpha = 1 / 2
            perf_score = alpha * t + (1 - alpha) * objs
            k_bests = k[np.where(perf_score == perf_score.min())]
            k_init = k_bests[0]
            print('phi_0_star :', k_init)

            ax[0].plot(k, objs, label=instance_name)
            ax[1].plot(k, t, label=instance_name)
        # ax[1].set_ylim([0,31])
        # ax[2].plot(k, perf_score)
        # ax[2].set_ylabel("Performance score")
        ax[0].legend()
        ax[1].legend()
        plt.show()

    def generate_regression_samples_k_prime(self, t_limit, instance_size='-small'):

        self.k_samples_directory = self.directory + 'k_samples_k_prime' + '/'
        self.regression_samples_directory_train = self.directory + 'regression_samples_k_prime' + '/train/'
        self.regression_samples_directory_test = self.directory + 'regression_samples_k_prime' + '/test/'
        pathlib.Path(self.regression_samples_directory_train).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.regression_samples_directory_test).mkdir(parents=True, exist_ok=True)

        direc = './data/generated_instances/' + self.instance_type + '/' + instance_size + '/'
        self.directory_transformedmodel = direc + 'transformedmodel' + '/train/'
        self.directory_sol = direc + self.incumbent_mode + '/train/'

        index_instance = 100
        list_phi = []
        while index_instance < 200:

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

            # instance_name = self.instance_type + '-' + str(index_instance)
            data = np.load(self.k_samples_directory + instance_name + '.npz')
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

            # t = mean_forward_filter(t,10)
            # objs = mean_forward_filter(objs, 10)

            # compute the performance score
            alpha = 1 / 2
            perf_score = alpha * t + (1 - alpha) * objs
            k_bests = k[np.where(perf_score == perf_score.min())]
            k_init = k_bests[0]
            print('phi_0_star :', k_init)
            list_phi.append(k_init)

            plt.clf()
            fig, ax = plt.subplots(2, 1, figsize=(6.4, 6.4))
            fig.suptitle("Evaluation of size of lb neighborhood")
            fig.subplots_adjust(top=0.5)
            ax[0].plot(k, objs)
            ax[0].set_title(instance_name, loc='right')
            ax[0].set_xlabel(r'$\ r $   ' + '(Neighborhood size: ' + r'$K = r \times N$)') #
            ax[0].set_ylabel("Objective")
            ax[1].plot(k, t)
            # ax[1].set_ylim([0,31])
            ax[1].set_ylabel("Solving time")
            # ax[2].plot(k, perf_score)
            # ax[2].set_ylabel("Performance score")
            plt.show()

            # instance = ecole.scip.Model.from_pyscipopt(MIP_model)

            # observation, _, _, done, _ = self.env.reset(instance)
            #
            # data_sample = [observation, k_init]
            # saved_name = f'{self.instance_type}-{str(index_instance)}_transformed'
            # if index_instance < 160:
            #     filename = f'{self.regression_samples_directory_train}regression-{saved_name}.pkl'
            # else:
            #     filename = f'{self.regression_samples_directory_test}regression-{saved_name}.pkl'
            # with gzip.open(filename, 'wb') as f:
            #     pickle.dump(data_sample, f)
            #
            # index_instance += 1

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

        index_instance = 0
        list_phi_prime = []
        list_phi_lp2relax = []
        list_phi_star = []
        count_phi_star_smaller = 0
        count_phi_lp_relax_diff = 0
        # list_phi_prime_invalid = []
        # list_phi_star_invalid = []
        while index_instance < 100:

            filename = f'{self.directory_transformedmodel}{self.instance_type}-{str(index_instance)}_transformed.cip'
            firstsol_filename = f'{self.directory_sol}{self.incumbent_mode}-{self.instance_type}-{str(index_instance)}_transformed.sol'

            MIP_model = Model()
            MIP_model.readProblem(filename)
            instance_name = MIP_model.getProbName()
            # print(instance_name)
            n_vars = MIP_model.getNVars()
            n_binvars = MIP_model.getNBinVars()
            # print("N of variables: {}".format(n_vars))
            print("N of binary vars: {}".format(n_binvars))
            # print("N of constraints: {}".format(MIP_model.getNConss()))

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

            # t = mean_forward_filter(t,10)
            # objs = mean_forward_filter(objs, 10)

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
            # MIP_model.setParam("limits/solutions", 1)
            MIP_model.setParam("display/verblevel", 0)
            MIP_model.setParam("lp/disablecutoff", 1)

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

            sol_lp = MIP_model.createLPSol()
            # sol_relax = MIP_model.createRelaxSol()

            k_prime = haming_distance_solutions(MIP_model, incumbent, sol_lp)
            # k_lp2relax = haming_distance_solutions(MIP_model, sol_relax, sol_lp)

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
            # phi_lp2relax = k_lp2relax / k_base
            # if phi_lp2relax > 0:
            #     count_phi_lp_relax_diff += 1

            phi_star = k_init
            list_phi_prime.append(phi_prime)
            list_phi_star.append(phi_star)
            # list_phi_lp2relax.append(phi_lp2relax)

            if phi_star <= phi_prime:
                count_phi_star_smaller += 1
            else:
                list_phi_prime_invalid = phi_prime
                list_phi_star_invalid = list_phi_star

            print('instance : ', MIP_model.getProbName())
            print('phi_prime = ', phi_prime)
            print('phi_star = ', phi_star)
            # print('phi_lp2relax = ', phi_lp2relax)
            print('valid count: ', count_phi_star_smaller)
            # print('lp relax diff count:', count_phi_lp_relax_diff)

            # plt.clf()
            # fig, ax = plt.subplots(3, 1, figsize=(6.4, 6.4))
            # fig.suptitle("Evaluation of size of lb neighborhood")
            # fig.subplots_adjust(top=0.5)
            # ax[0].plot(k, objs)
            # ax[0].set_title(instance_name, loc='right')
            # ax[0].set_xlabel(r'$\ r $   ' + '(Neighborhood size: ' + r'$K = r \times N$)') #
            # ax[0].set_ylabel("Objective")
            # ax[1].plot(k, t)
            # # ax[1].set_ylim([0,31])
            # ax[1].set_ylabel("Solving time")
            # ax[2].plot(k, perf_score)
            # ax[2].set_ylabel("Performance score")
            # plt.show()

            # instance = ecole.scip.Model.from_pyscipopt(MIP_model)
            # observation, _, _, done, _ = self.env.reset(instance)
            #
            # data_sample = [observation, k_init]
            # filename = f'{self.regression_samples_directory}regression-{instance_name}.pkl'
            # with gzip.open(filename, 'wb') as f:
            #     pickle.dump(data_sample, f)

            index_instance += 1

        arr_phi_prime = np.array(list_phi_prime).reshape(-1)
        arr_phi_star = np.array(list_phi_star).reshape(-1)
        # arr_phi_lp2relax = np.array(list_phi_lp2relax).reshape(-1)
        ave_phi_prime = arr_phi_prime.sum() / len(arr_phi_prime)
        ave_phi_star = arr_phi_star.sum() / len(arr_phi_star)
        # ave_phi_lp2relax = arr_phi_lp2relax.sum() / len(arr_phi_lp2relax)

        print(self.instance_type + self.instance_size)
        print(self.incumbent_mode + 'Solution')
        print('number of valid phi data points: ', count_phi_star_smaller)
        print('average phi_star :', ave_phi_star)
        print('average phi_prime: ', ave_phi_prime)
        # print('average phi_lp2relax: ', ave_phi_lp2relax)

    def generate_dataset(self, dataset_directory=None, filename=None):
        self.regression_samples_directory = dataset_directory
        train_directory = self.regression_samples_directory + 'train/'
        test_directory = self.regression_samples_directory + 'test/'
        # print(filename)
        sample_files = [str(path) for path in pathlib.Path(train_directory).glob(filename)]
        train_files = sample_files[:int(7/8 * len(sample_files))]
        # valid_files = sample_files[int(0.7 * len(sample_files)):int(0.8 * len(sample_files))]
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

        # large_dataset = self.instance_type + "-large"
        # large_directory = './result/generated_instances/' + self.instance_type + '/' + '-large' + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # test_regression_samples_directory = large_directory + 'regression_samples' + '/'
        # train_loader, valid_loader, test_loader = self.load_dataset(dataset_directory=test_regression_samples_directory)
        # train_loaders[large_dataset] = train_loader
        # val_loaders[large_dataset] = valid_loader
        # test_loaders[large_dataset] = test_loader

        model_gnn = GNNPolicy()
        train_dataset = small_dataset
        valid_dataset = small_dataset
        test_dataset = small_dataset
        # LEARNING_RATE = 0.0000001  # setcovering:0.0000005 cap-loc: 0.00000005 independentset: 0.0000001

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

            # torch.save(model_gnn.state_dict(), 'trained_params_' + train_dataset + '.pth')
            # model_gnn2.load_state_dict(torch.load('trained_params_' + train_dataset + '.pth'))

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
        # small_dataset_setcover = "setcovering" + "-small"
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

        # small_dataset_independentset = "independentset" + "-small"
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

        # small_dataset = "combinatorialauction-root-first"
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

        # # small_dataset
        # small_directory = './result/generated_instances/' + 'generalized_independentset' + '/' + '-small' + '/' + 'symmetric' + '/' + 'firstsol' + '/'
        # small_regression_samples_directory = small_directory + 'regression_samples_k_prime' + '/'
        # filename = 'regression-' + 'generalized_independentset' + '-*.pkl'
        # train_dataset_6, valid_dataset_6, test_dataset_6 = self.generate_dataset(
        #     dataset_directory=small_regression_samples_directory, filename=filename)
        #
        # small_directory = './result/generated_instances/' + 'generalized_independentset' + '/' + '-small' + '/' + 'symmetric' + '/' + 'rootsol' + '/'
        # small_regression_samples_directory = small_directory + 'regression_samples_k_prime' + '/'
        # filename = 'regression-' + 'generalized_independentset' + '-*.pkl'
        # train_dataset_7, valid_dataset_7, test_dataset_7 = self.generate_dataset(
        #     dataset_directory=small_regression_samples_directory, filename=filename)

        train_data = torch.utils.data.ConcatDataset([train_dataset_0, train_dataset_1, train_dataset_2, train_dataset_3, train_dataset_4, train_dataset_5]) #train_dataset_3, train_dataset_6, train_dataset_7
        valid_data = torch.utils.data.ConcatDataset([valid_dataset_0, valid_dataset_1, valid_dataset_2, valid_dataset_3, valid_dataset_4, valid_dataset_5]) #  valid_dataset_6, valid_dataset_7
        test_data = torch.utils.data.ConcatDataset([test_dataset_0, test_dataset_1,  test_dataset_2, test_dataset_3, test_dataset_4, test_dataset_5])# test_dataset_6, test_dataset_7]

        train_loader, valid_loader, test_loader = self.load_dataset(train_data, valid_data, test_data)
        train_loaders[small_dataset] = train_loader
        val_loaders[small_dataset] = valid_loader
        test_loaders[small_dataset] = test_loader

        # large_dataset = self.instance_type + "-large"
        # large_directory = './result/generated_instances/' + self.instance_type + '/' + '-large' + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # test_regression_samples_directory = large_directory + 'regression_samples' + '/'
        # train_loader, valid_loader, test_loader = self.load_dataset(dataset_directory=test_regression_samples_directory)
        # train_loaders[large_dataset] = train_loader
        # val_loaders[large_dataset] = valid_loader
        # test_loaders[large_dataset] = test_loader

        model_gnn = GNNPolicy()
        train_dataset = small_dataset
        valid_dataset = small_dataset
        test_dataset = small_dataset

        # model_gnn.load_state_dict(torch.load(
        #     saved_gnn_directory + 'trained_params_mean_' + train_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '_k_prime_lr0.0001_epoch300.pth'))

        # LEARNING_RATE = 0.0000001  # setcovering:0.0000005 cap-loc: 0.00000005 independentset: 0.0000001

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

            # torch.save(model_gnn.state_dict(), 'trained_params_' + train_dataset + '.pth')
            # model_gnn2.load_state_dict(torch.load('trained_params_' + train_dataset + '.pth'))

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
        graph = BipartiteNodeData(observation.constraint_features,
                                  observation.edge_features.indices,
                                  observation.edge_features.values,
                                  variable_features)

        # variable features: all the variable features
        # graph = BipartiteNodeData(observation.constraint_features,
        #                           observation.edge_features.indices,
        #                           observation.edge_features.values,
        #                           observation.variable_features)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = observation.constraint_features.shape[0] + \
                          observation.variable_features.shape[
                              0]

        # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
        # MIP_model = instance

        # MIP_model.optimize()
        # print("Status:", MIP_model.getStatus())
        # print("best obj: ", MIP_model.getObjVal())
        # print("Solving time: ", MIP_model.getSolvingTime())

        # create a copy of MIP
        MIP_model.resetParams()
        # MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
        #     problemName='Baseline', origcopy=False)
        # MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
        #     problemName='GNN',
        #     origcopy=False)
        MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
            problemName='GNN+reset',
            origcopy=False)

        print('MIP copies are created')

        # MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent,
        #                                         MIP_copy_vars)
        # MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent,
        #                                           MIP_copy_vars2)
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

        if k_pred < 10:
            k_pred = 10

        del k_model
        del graph
        del observation

        MIP_model.freeProb()
        del MIP_model
        del incumbent

        # sol = MIP_model_copy.getBestSol()
        # initial_obj = MIP_model_copy.getSolObjVal(sol)
        # print("Initial obj before LB: {}".format(initial_obj))
        #
        # # execute local branching baseline heuristic by Fischetti and Lodi
        # lb_model = LocalBranching(MIP_model=MIP_model_copy, MIP_sol_bar=sol_MIP_copy, k=self.k_baseline,
        #                           node_time_limit=node_time_limit,
        #                           total_time_limit=total_time_limit)
        # status, obj_best, elapsed_time, lb_bits, times, objs, _, _ = lb_model.mdp_localbranch(
        #     is_symmetric=self.is_symmetric,
        #     reset_k_at_2nditeration=False,
        #     policy=None,
        #     optimizer=None,
        #     device=device
        # )
        # print("Instance:", MIP_model_copy.getProbName())
        # print("Status of LB: ", status)
        # print("Best obj of LB: ", obj_best)
        # print("Solving time: ", elapsed_time)
        # print('\n')
        #
        # MIP_model_copy.freeProb()
        # del sol_MIP_copy
        # del MIP_model_copy
        #
        # sol = MIP_model_copy2.getBestSol()
        # initial_obj = MIP_model_copy2.getSolObjVal(sol)
        # print("Initial obj before LB: {}".format(initial_obj))

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

        # # execute local branching with 1. first k predicted by GNN; 2. from 2nd iteration of lb, continue lb algorithm with no further injection
        #
        # lb_model2 = LocalBranching(MIP_model=MIP_model_copy2, MIP_sol_bar=sol_MIP_copy2, k=k_pred,
        #                            node_time_limit=node_time_limit,
        #                            total_time_limit=total_time_limit)
        # status, obj_best, elapsed_time, lb_bits_regression_noreset, times_regression_noreset, objs_regression_noreset, _, _ = lb_model2.mdp_localbranch(
        #     is_symmetric=self.is_symmetric,
        #     reset_k_at_2nditeration=False,
        #     policy=None,
        #     optimizer=None,
        #     device=device
        # )
        #
        # print("Instance:", MIP_model_copy2.getProbName())
        # print("Status of LB: ", status)
        # print("Best obj of LB: ", obj_best)
        # print("Solving time: ", elapsed_time)
        # print('\n')
        #
        # MIP_model_copy2.freeProb()
        # del sol_MIP_copy2
        # del MIP_model_copy2

        data = [objs_regression_reset, times_regression_reset] # objs, times,
        saved_name = f'{self.instance_type}-{str(index_instance)}_transformed'
        filename = f'{self.directory_lb_test}lb-test-{saved_name}.pkl'  # instance 100-199
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)
        # del data
        # del objs
        # del times
        del objs_regression_reset
        del times_regression_reset
        # del lb_model
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
        try:
            MIP_model.addSol(incumbent, False)
        except:
            print('Error: the root solution of ' + instance_name + ' is not feasible!')

        instance = ecole.scip.Model.from_pyscipopt(MIP_model)
        observation, _, _, done, _ = self.env.reset(instance)

        # variable features: only incumbent solution
        variable_features = observation.variable_features[:, -1:]
        graph = BipartiteNodeData(observation.constraint_features,
                                  observation.edge_features.indices,
                                  observation.edge_features.values,
                                  variable_features)

        # variable features: all the variable features
        # graph = BipartiteNodeData(observation.constraint_features,
        #                           observation.edge_features.indices,
        #                           observation.edge_features.values,
        #                           observation.variable_features)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = observation.constraint_features.shape[0] + \
                          observation.variable_features.shape[
                              0]

        # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
        # MIP_model = instance

        # MIP_model.optimize()
        # print("Status:", MIP_model.getStatus())
        # print("best obj: ", MIP_model.getObjVal())
        # print("Solving time: ", MIP_model.getSolvingTime())

        # create a copy of MIP
        # MIP_model.resetParams()
        # MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
        #     problemName='Baseline', origcopy=False)
        # MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
        #     problemName='GNN',
        #     origcopy=False)
        MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
            problemName='GNN+reset',
            origcopy=False)

        print('MIP copies are created')

        # MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent,
        #                                         MIP_copy_vars)
        # MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent,
        #                                           MIP_copy_vars2)
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

        if k_pred < 10:
            k_pred = 10

        del k_model
        del graph
        del observation

        MIP_model.freeProb()
        del MIP_model
        del incumbent

        # sol = MIP_model_copy.getBestSol()
        # initial_obj = MIP_model_copy.getSolObjVal(sol)
        # print("Initial obj before LB: {}".format(initial_obj))
        #
        # # execute local branching baseline heuristic by Fischetti and Lodi
        # lb_model = LocalBranching(MIP_model=MIP_model_copy, MIP_sol_bar=sol_MIP_copy, k=self.k_baseline,
        #                           node_time_limit=node_time_limit,
        #                           total_time_limit=total_time_limit)
        # status, obj_best, elapsed_time, lb_bits, times, objs, _, _ = lb_model.mdp_localbranch(
        #     is_symmetric=self.is_symmetric,
        #     reset_k_at_2nditeration=False,
        #     policy=None,
        #     optimizer=None,
        #     device=device
        # )
        # print("Instance:", MIP_model_copy.getProbName())
        # print("Status of LB: ", status)
        # print("Best obj of LB: ", obj_best)
        # print("Solving time: ", elapsed_time)
        # print('\n')
        #
        # MIP_model_copy.freeProb()
        # del sol_MIP_copy
        # del MIP_model_copy
        #
        # sol = MIP_model_copy2.getBestSol()
        # initial_obj = MIP_model_copy2.getSolObjVal(sol)
        # print("Initial obj before LB: {}".format(initial_obj))

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

        # # execute local branching with 1. first k predicted by GNN; 2. from 2nd iteration of lb, continue lb algorithm with no further injection
        #
        # lb_model2 = LocalBranching(MIP_model=MIP_model_copy2, MIP_sol_bar=sol_MIP_copy2, k=k_pred,
        #                            node_time_limit=node_time_limit,
        #                            total_time_limit=total_time_limit)
        # status, obj_best, elapsed_time, lb_bits_regression_noreset, times_regression_noreset, objs_regression_noreset, _, _ = lb_model2.mdp_localbranch(
        #     is_symmetric=self.is_symmetric,
        #     reset_k_at_2nditeration=False,
        #     policy=None,
        #     optimizer=None,
        #     device=device
        # )
        #
        # print("Instance:", MIP_model_copy2.getProbName())
        # print("Status of LB: ", status)
        # print("Best obj of LB: ", obj_best)
        # print("Solving time: ", elapsed_time)
        # print('\n')
        #
        # MIP_model_copy2.freeProb()
        # del sol_MIP_copy2
        # del MIP_model_copy2

        data = [objs_regression_reset, times_regression_reset]
        saved_name = f'{self.instance_type}-{str(index_instance)}_transformed'
        filename = f'{self.directory_lb_test}lb-test-{saved_name}.pkl'  # instance 100-199
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)
        # del data
        # del objs
        # del times
        del objs_regression_reset
        del times_regression_reset
        # del lb_model
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

        # if index_instance == 18:
        #     index_instance = 19

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
        # observation, _, _, done, _ = self.env.reset(instance)
        #
        # # variable features: only incumbent solution
        # variable_features = observation.variable_features[:, -1:]
        # graph = BipartiteNodeData(observation.constraint_features,
        #                           observation.edge_features.indices,
        #                           observation.edge_features.values,
        #                           variable_features)

        # variable features: all the variable features
        # graph = BipartiteNodeData(observation.constraint_features,
        #                           observation.edge_features.indices,
        #                           observation.edge_features.values,
        #                           observation.variable_features)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        # graph.num_nodes = observation.constraint_features.shape[0] + \
        #                   observation.variable_features.shape[
        #                       0]

        # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
        # MIP_model = instance

        # MIP_model.optimize()
        # print("Status:", MIP_model.getStatus())
        # print("best obj: ", MIP_model.getObjVal())
        # print("Solving time: ", MIP_model.getSolvingTime())

        # create a copy of MIP
        MIP_model.resetParams()


        MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
            problemName='Baseline', origcopy=False)
        # MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
        #     problemName='GNN',
        #     origcopy=False)
        # MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
        #     problemName='GNN+reset',
        #     origcopy=False)

        print('MIP copies are created')

        MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent,
                                                MIP_copy_vars)
        # MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent,
        #                                           MIP_copy_vars2)
        # MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent,
        #                                           MIP_copy_vars3)

        print('incumbent solution is copied to MIP copies')

        # # solve the root node and get the LP solution, compute k_prime
        # k_prime = self.compute_k_prime(MIP_model, incumbent)

        initial_obj = MIP_model.getSolObjVal(incumbent)
        print("Initial obj before LB: {}".format(initial_obj))

        # binary_supports = binary_support(MIP_model, incumbent)
        # print('binary support: ', binary_supports)
        #
        # k_model = self.regression_model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
        #                                     graph.variable_features)
        #
        # k_pred = k_model.item() * k_prime
        # print('GNN prediction: ', k_model.item())
        #
        # if self.is_symmetric == False:
        #     k_pred = k_model.item() * k_prime

        # del k_model
        # del graph
        # del observation

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

        # sol = MIP_model_copy2.getBestSol()
        # initial_obj = MIP_model_copy2.getSolObjVal(sol)
        # print("Initial obj before LB: {}".format(initial_obj))

        # # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
        # lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=k_pred,
        #                            node_time_limit=node_time_limit,
        #                            total_time_limit=total_time_limit)
        # status, obj_best, elapsed_time, lb_bits_regression_reset, times_regression_reset, objs_regression_reset, _, _ = lb_model3.mdp_localbranch(
        #     is_symmetric=self.is_symmetric,
        #     reset_k_at_2nditeration=reset_k_at_2nditeration,
        #     policy=None,
        #     optimizer=None,
        #     device=device
        # )
        #
        # print("Instance:", MIP_model_copy3.getProbName())
        # print("Status of LB: ", status)
        # print("Best obj of LB: ", obj_best)
        # print("Solving time: ", elapsed_time)
        # print('\n')
        #
        # MIP_model_copy3.freeProb()
        # del sol_MIP_copy3
        # del MIP_model_copy3

        # # execute local branching with 1. first k predicted by GNN; 2. from 2nd iteration of lb, continue lb algorithm with no further injection
        #
        # lb_model2 = LocalBranching(MIP_model=MIP_model_copy2, MIP_sol_bar=sol_MIP_copy2, k=k_pred,
        #                            node_time_limit=node_time_limit,
        #                            total_time_limit=total_time_limit)
        # status, obj_best, elapsed_time, lb_bits_regression_noreset, times_regression_noreset, objs_regression_noreset, _, _ = lb_model2.mdp_localbranch(
        #     is_symmetric=self.is_symmetric,
        #     reset_k_at_2nditeration=False,
        #     policy=None,
        #     optimizer=None,
        #     device=device
        # )
        #
        # print("Instance:", MIP_model_copy2.getProbName())
        # print("Status of LB: ", status)
        # print("Best obj of LB: ", obj_best)
        # print("Solving time: ", elapsed_time)
        # print('\n')
        #
        # MIP_model_copy2.freeProb()
        # del sol_MIP_copy2
        # del MIP_model_copy2

        data = [objs, times]
        saved_name = f'{self.instance_type}-{str(index_instance)}_transformed'
        filename = f'{self.directory_lb_test}lb-test-{saved_name}.pkl'  # instance 100-199
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)
        # del data
        del objs
        del times
        # del objs_regression_reset
        # del times_regression_reset
        del lb_model
        # del lb_model3

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
                # self.regression_model_gnn.load_state_dict(torch.load(
                #     self.saved_gnn_directory + 'trained_params_mean_setcover-independentset-combinatorialauction-generalizedis_asymmetric_firstsol_k_prime_epoch183_used.pth'))
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

        index_instance = 160 # 160, 0
        index_max =200 # 200, 30

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
        # baseline: baseline
        # k_prime_merged: gnn_prime_merged

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
        #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/'
            # directory_lb_test_2 = directory_2 + 'lb-from-' + 'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(
            #     total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'


        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/'
            # directory_lb_test_2 = directory_2 + 'lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(
            #     total_time_limit) + 's' + test_instance_size + '/'
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

            # filename = f'{directory_lb_test}lb-test-{instance_name}.pkl'
            #
            # with gzip.open(filename, 'rb') as f:
            #     data = pickle.load(f)
            # objs, times, objs_pred, times_pred, objs_pred_reset, times_pred_reset = data  # objs contains objs of a single instance of a lb test
            #
            # filename_2 = f'{directory_lb_test_2}lb-test-{instance_name}.pkl'
            #
            # with gzip.open(filename_2, 'rb') as f:
            #     data = pickle.load(f)
            # objs_2, times_2, objs_pred_2, times_pred_2, objs_pred_reset_2, times_pred_reset_2 = data  # objs contains objs of a single instance of a lb test

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

            # print('baseline: ')
            # print('objs :', objs)
            # print('times :', times)
            #
            # print('homo: ')
            # print('objs :', objs_k_prime)
            # print('times :', times_k_prime)
            #
            # print('merged: ')
            # print('objs :', objs_k_prime_merged)
            # print('times :', times_k_prime_merged)



            objs = np.array(objs).reshape(-1)
            times = np.array(times).reshape(-1)

            objs_2 = np.array(objs_2).reshape(-1)

            objs_k_prime = np.array(objs_k_prime).reshape(-1)
            times_k_prime = np.array(times_k_prime).reshape(-1)

            objs_k_prime_2 = np.array(objs_k_prime_2).reshape(-1)

            objs_k_prime_merged = np.array(objs_k_prime_merged).reshape(-1)
            times_k_prime_merged = np.array(times_k_prime_merged).reshape(-1)

            objs_k_prime_merged_2 = np.array(objs_k_prime_merged_2).reshape(-1)

            a = [objs.min(), objs_2.min(), objs_k_prime.min(), objs_k_prime_2.min(), objs_k_prime_merged.min(), objs_k_prime_merged_2.min()]  # objs_2.min(), objs_pred_2.min(), objs_pred_reset_2.min(),
            # a = [objs.min(), objs_pred.min(), objs_pred_reset.min()]
            obj_opt = np.amin(a)

            # # compute primal gap for baseline localbranching run
            # # if times[-1] < total_time_limit:
            # times = np.append(times, total_time_limit)
            # objs = np.append(objs, objs[-1])
            #
            # gamma_baseline = np.zeros(len(objs))
            # for j in range(len(objs)):
            #     if objs[j] == 0 and obj_opt == 0:
            #         gamma_baseline[j] = 0
            #     elif objs[j] * obj_opt < 0:
            #         gamma_baseline[j] = 1
            #     else:
            #         gamma_baseline[j] = np.abs(objs[j] - obj_opt) / np.maximum(np.abs(objs[j]), np.abs(obj_opt))  #
            #
            # # compute the primal gap of last objective
            # primal_gap_final_baseline = np.abs(objs[-1] - obj_opt) / np.abs(obj_opt)
            # primal_gap_final_baselines.append(primal_gap_final_baseline)
            #
            # # create step line
            # stepline_baseline = interp1d(times, gamma_baseline, 'previous')
            # steplines_baseline.append(stepline_baseline)
            #
            # # compute primal integral
            # primal_int_baseline = 0
            # for j in range(len(objs) - 1):
            #     primal_int_baseline += gamma_baseline[j] * (times[j + 1] - times[j])
            # primal_int_baselines.append(primal_int_baseline)primal_int_baseline

            # # lb-gnn
            # # if times_pred[-1] < total_time_limit:
            # times_pred = np.append(times_pred, total_time_limit)
            # objs_pred = np.append(objs_pred, objs_pred[-1])
            #
            # gamma_pred = np.zeros(len(objs_pred))
            # for j in range(len(objs_pred)):
            #     if objs_pred[j] == 0 and obj_opt == 0:
            #         gamma_pred[j] = 0
            #     elif objs_pred[j] * obj_opt < 0:
            #         gamma_pred[j] = 1
            #     else:
            #         gamma_pred[j] = np.abs(objs_pred[j] - obj_opt) / np.maximum(np.abs(objs_pred[j]),
            #                                                                     np.abs(obj_opt))  #
            #
            # primal_gap_final_pred = np.abs(objs_pred[-1] - obj_opt) / np.abs(obj_opt)
            # primal_gap_final_preds.append(primal_gap_final_pred)
            #
            # stepline_pred = interp1d(times_pred, gamma_pred, 'previous')
            # steplines_pred.append(stepline_pred)
            #
            # # compute primal interal
            # primal_int_pred = 0
            # for j in range(len(objs_pred) - 1):
            #     primal_int_pred += gamma_pred[j] * (times_pred[j + 1] - times_pred[j])
            # primal_int_preds.append(primal_int_pred)
            #
            # # lb-gnn-reset
            # times_pred_reset = np.append(times_pred_reset, total_time_limit)
            # objs_pred_reset = np.append(objs_pred_reset, objs_pred_reset[-1])
            #
            # gamma_pred_reset = np.zeros(len(objs_pred_reset))
            # for j in range(len(objs_pred_reset)):
            #     if objs_pred_reset[j] == 0 and obj_opt == 0:
            #         gamma_pred_reset[j] = 0
            #     elif objs_pred_reset[j] * obj_opt < 0:
            #         gamma_pred_reset[j] = 1
            #     else:
            #         gamma_pred_reset[j] = np.abs(objs_pred_reset[j] - obj_opt) / np.maximum(np.abs(objs_pred_reset[j]),
            #                                                                                 np.abs(obj_opt))  #
            #
            # primal_gap_final_pred_reset = np.abs(objs_pred_reset[-1] - obj_opt) / np.abs(obj_opt)
            # primal_gap_final_preds_reset.append(primal_gap_final_pred_reset)
            #
            # stepline_pred_reset = interp1d(times_pred_reset, gamma_pred_reset, 'previous')
            # steplines_pred_reset.append(stepline_pred_reset)
            #
            # # compute primal interal
            # primal_int_pred_reset = 0
            # for j in range(len(objs_pred_reset) - 1):
            #     primal_int_pred_reset += gamma_pred_reset[j] * (times_pred_reset[j + 1] - times_pred_reset[j])
            # primal_int_preds_reset.append(primal_int_pred_reset)

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

            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of objective")
            # fig.subplots_adjust(top=0.5)
            # ax.set_title(instance_name, loc='right')
            # ax.plot(times, objs, label='lb baseline')
            # ax.plot(times_pred, objs_pred, label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()
            #
            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of primal gap")
            # fig.subplots_adjust(top=0.5)
            # ax.set_title(instance_name, loc='right')
            # ax.plot(times, gamma_baseline, label='lb baseline')
            # ax.plot(times_pred, gamma_pred, label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()

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

        # avarage primal integral over test dataset
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
        # print('k_pred primal integral: ', primal_int_pred_ave)
        # print('k_pred_reset primal integral: ', primal_int_pred_ave_reset)
        print('k_regre_prime primal integral: ', primal_int_regression_k_prime_ave)
        print('k_regre_prime_merged primal integral: ', primal_int_regression_k_prime_merged_ave)
        print('\n')
        print('baseline primal gap: ', primal_gap_final_baselines_ave)
        # print('k_pred primal gap: ', primal_gap_final_preds)
        # print('k_pred_reset primal gap: ', primal_gap_final_preds_reset)
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

        # primalgaps_pred = None
        # for n, stepline_pred in enumerate(steplines_pred):
        #     primal_gap = stepline_pred(t)
        #     if n == 0:
        #         primalgaps_pred = primal_gap
        #     else:
        #         primalgaps_pred = np.vstack((primalgaps_pred, primal_gap))
        # primalgap_pred_ave = np.average(primalgaps_pred, axis=0)
        #
        # primalgaps_pred_reset = None
        # for n, stepline_pred_reset in enumerate(steplines_pred_reset):
        #     primal_gap = stepline_pred_reset(t)
        #     if n == 0:
        #         primalgaps_pred_reset = primal_gap
        #     else:
        #         primalgaps_pred_reset = np.vstack((primalgaps_pred_reset, primal_gap))
        # primalgap_pred_ave_reset = np.average(primalgaps_pred_reset, axis=0)

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
        # fig.subplots_adjust(top=0.5)
        ax.set_title(self.instance_type + '-' + test_instance_size + '-' + self.incumbent_mode, loc='right')
        ax.plot(t, primalgap_baseline_ave, label='lb-baseline')
        # ax.plot(t, primalgap_pred_ave, label='lb-regression-noreset')
        # ax.plot(t, primalgap_pred_ave_reset, '--', label='lb-regression')
        ax.plot(t, primalgap_regression_k_prime_ave, label='lb-regression-k-prime-homo')
        ax.plot(t, primalgap_regression_k_prime_merged_ave, label='lb-regression-k-prime-merged')
        ax.set_xlabel('time /s')
        ax.set_ylabel("normalized primal gap")
        ax.legend()
        plt.show()

    def primal_integral_k_prime_3_sepa(self, test_instance_size, total_time_limit=60, node_time_limit=30):

        # input:
        # k_prime: gnn_prime without merged
        # baseline: baseline
        # k_prime_merged: gnn_prime_merged

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
        #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/'
            # directory_lb_test_2 = directory_2 + 'lb-from-' + 'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(
            #     total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'


        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/'
            # directory_lb_test_2 = directory_2 + 'lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(
            #     total_time_limit) + 's' + test_instance_size + '/'
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

            # filename = f'{directory_lb_test}lb-test-{instance_name}.pkl'
            #
            # with gzip.open(filename, 'rb') as f:
            #     data = pickle.load(f)
            # objs, times, objs_pred, times_pred, objs_pred_reset, times_pred_reset = data  # objs contains objs of a single instance of a lb test
            #
            # filename_2 = f'{directory_lb_test_2}lb-test-{instance_name}.pkl'
            #
            # with gzip.open(filename_2, 'rb') as f:
            #     data = pickle.load(f)
            # objs_2, times_2, objs_pred_2, times_pred_2, objs_pred_reset_2, times_pred_reset_2 = data  # objs contains objs of a single instance of a lb test

            # # test from k_prime
            # filename = f'{directory_lb_test_k_prime}lb-test-{instance_name}.pkl'
            # with gzip.open(filename, 'rb') as f:
            #     data = pickle.load(f)
            # objs_k_prime, times_k_prime = data  # objs contains objs of a single instance of a lb test

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

            # filename = f'{directory_lb_test_k_prime_2}lb-test-{instance_name}.pkl'
            # with gzip.open(filename, 'rb') as f:
            #     data = pickle.load(f)
            # objs_k_prime_2, times_k_prime_2 = data  # objs contains objs of a single instance of a lb test

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

            # objs_k_prime = np.array(objs_k_prime).reshape(-1)
            # times_k_prime = np.array(times_k_prime).reshape(-1)
            #
            # objs_k_prime_2 = np.array(objs_k_prime_2).reshape(-1)

            objs_k_prime_merged = np.array(objs_k_prime_merged).reshape(-1)
            times_k_prime_merged = np.array(times_k_prime_merged).reshape(-1)

            objs_k_prime_merged_2 = np.array(objs_k_prime_merged_2).reshape(-1)

            a = [objs.min(), objs_2.min(), objs_k_prime_merged.min(), objs_k_prime_merged_2.min()]  # objs_2.min(), objs_pred_2.min(), objs_pred_reset_2.min(), objs_k_prime.min(), objs_k_prime_2.min(),
            # a = [objs.min(), objs_pred.min(), objs_pred_reset.min()]
            obj_opt = np.amin(a)

            # # compute primal gap for baseline localbranching run
            # # if times[-1] < total_time_limit:
            # times = np.append(times, total_time_limit)
            # objs = np.append(objs, objs[-1])
            #
            # gamma_baseline = np.zeros(len(objs))
            # for j in range(len(objs)):
            #     if objs[j] == 0 and obj_opt == 0:
            #         gamma_baseline[j] = 0
            #     elif objs[j] * obj_opt < 0:
            #         gamma_baseline[j] = 1
            #     else:
            #         gamma_baseline[j] = np.abs(objs[j] - obj_opt) / np.maximum(np.abs(objs[j]), np.abs(obj_opt))  #
            #
            # # compute the primal gap of last objective
            # primal_gap_final_baseline = np.abs(objs[-1] - obj_opt) / np.abs(obj_opt)
            # primal_gap_final_baselines.append(primal_gap_final_baseline)
            #
            # # create step line
            # stepline_baseline = interp1d(times, gamma_baseline, 'previous')
            # steplines_baseline.append(stepline_baseline)
            #
            # # compute primal integral
            # primal_int_baseline = 0
            # for j in range(len(objs) - 1):
            #     primal_int_baseline += gamma_baseline[j] * (times[j + 1] - times[j])
            # primal_int_baselines.append(primal_int_baseline)primal_int_baseline

            # # lb-gnn
            # # if times_pred[-1] < total_time_limit:
            # times_pred = np.append(times_pred, total_time_limit)
            # objs_pred = np.append(objs_pred, objs_pred[-1])
            #
            # gamma_pred = np.zeros(len(objs_pred))
            # for j in range(len(objs_pred)):
            #     if objs_pred[j] == 0 and obj_opt == 0:
            #         gamma_pred[j] = 0
            #     elif objs_pred[j] * obj_opt < 0:
            #         gamma_pred[j] = 1
            #     else:
            #         gamma_pred[j] = np.abs(objs_pred[j] - obj_opt) / np.maximum(np.abs(objs_pred[j]),
            #                                                                     np.abs(obj_opt))  #
            #
            # primal_gap_final_pred = np.abs(objs_pred[-1] - obj_opt) / np.abs(obj_opt)
            # primal_gap_final_preds.append(primal_gap_final_pred)
            #
            # stepline_pred = interp1d(times_pred, gamma_pred, 'previous')
            # steplines_pred.append(stepline_pred)
            #
            # # compute primal interal
            # primal_int_pred = 0
            # for j in range(len(objs_pred) - 1):
            #     primal_int_pred += gamma_pred[j] * (times_pred[j + 1] - times_pred[j])
            # primal_int_preds.append(primal_int_pred)
            #
            # # lb-gnn-reset
            # times_pred_reset = np.append(times_pred_reset, total_time_limit)
            # objs_pred_reset = np.append(objs_pred_reset, objs_pred_reset[-1])
            #
            # gamma_pred_reset = np.zeros(len(objs_pred_reset))
            # for j in range(len(objs_pred_reset)):
            #     if objs_pred_reset[j] == 0 and obj_opt == 0:
            #         gamma_pred_reset[j] = 0
            #     elif objs_pred_reset[j] * obj_opt < 0:
            #         gamma_pred_reset[j] = 1
            #     else:
            #         gamma_pred_reset[j] = np.abs(objs_pred_reset[j] - obj_opt) / np.maximum(np.abs(objs_pred_reset[j]),
            #                                                                                 np.abs(obj_opt))  #
            #
            # primal_gap_final_pred_reset = np.abs(objs_pred_reset[-1] - obj_opt) / np.abs(obj_opt)
            # primal_gap_final_preds_reset.append(primal_gap_final_pred_reset)
            #
            # stepline_pred_reset = interp1d(times_pred_reset, gamma_pred_reset, 'previous')
            # steplines_pred_reset.append(stepline_pred_reset)
            #
            # # compute primal interal
            # primal_int_pred_reset = 0
            # for j in range(len(objs_pred_reset) - 1):
            #     primal_int_pred_reset += gamma_pred_reset[j] * (times_pred_reset[j + 1] - times_pred_reset[j])
            # primal_int_preds_reset.append(primal_int_pred_reset)

            # lb-regression-k-prime
            # if times_regression[-1] < total_time_limit:


            # baseline

            primal_int_baseline, primal_gap_final_baseline, stepline_baseline = self.compute_primal_integral(
                times=times, objs=objs, obj_opt=obj_opt, total_time_limit=total_time_limit)

            primal_gap_final_baselines.append(primal_gap_final_baseline)
            steplines_baseline.append(stepline_baseline)
            primal_int_baselines.append(primal_int_baseline)

            # # regression_k_prime
            # primal_int_regression_k_prime, primal_gap_final_regression_k_prime, stepline_regression_k_prime = self.compute_primal_integral(
            #     times=times_k_prime, objs=objs_k_prime, obj_opt=obj_opt, total_time_limit=total_time_limit)
            #
            # primal_gap_final_regression_k_primes.append(primal_gap_final_regression_k_prime)
            # steplines_regression_k_primes.append(stepline_regression_k_prime)
            # primal_int_regression_k_primes.append(primal_int_regression_k_prime)

            # regression_k_prime_merged
            primal_int_regression_k_prime_merged, primal_gap_final_regression_k_prime_merged, stepline_regression_k_prime_merged = self.compute_primal_integral(
                times=times_k_prime_merged, objs=objs_k_prime_merged, obj_opt=obj_opt, total_time_limit=total_time_limit)

            primal_gap_final_regression_k_primes_merged.append(primal_gap_final_regression_k_prime_merged)
            steplines_regression_k_primes_merged.append(stepline_regression_k_prime_merged)
            primal_int_regression_k_primes_merged.append(primal_int_regression_k_prime_merged)

            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of objective")
            # fig.subplots_adjust(top=0.5)
            # ax.set_title(instance_name, loc='right')
            # ax.plot(times, objs, label='lb baseline')
            # ax.plot(times_pred, objs_pred, label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()
            #
            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of primal gap")
            # fig.subplots_adjust(top=0.5)
            # ax.set_title(instance_name, loc='right')
            # ax.plot(times, gamma_baseline, label='lb baseline')
            # ax.plot(times_pred, gamma_pred, label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()

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

        # avarage primal integral over test dataset
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
        # print('k_pred primal integral: ', primal_int_pred_ave)
        # print('k_pred_reset primal integral: ', primal_int_pred_ave_reset)
        # print('k_regre_prime primal integral: ', primal_int_regression_k_prime_ave)
        print('k_regre_prime_merged primal integral: ', primal_int_regression_k_prime_merged_ave)
        print('\n')
        print('baseline primal gap: ', primal_gap_final_baselines_ave)
        # print('k_pred primal gap: ', primal_gap_final_preds)
        # print('k_pred_reset primal gap: ', primal_gap_final_preds_reset)
        # print('k_regre_prime primal gap: ', primal_gap_final_regression_k_primes_ave)
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

        # primalgaps_pred = None
        # for n, stepline_pred in enumerate(steplines_pred):
        #     primal_gap = stepline_pred(t)
        #     if n == 0:
        #         primalgaps_pred = primal_gap
        #     else:
        #         primalgaps_pred = np.vstack((primalgaps_pred, primal_gap))
        # primalgap_pred_ave = np.average(primalgaps_pred, axis=0)
        #
        # primalgaps_pred_reset = None
        # for n, stepline_pred_reset in enumerate(steplines_pred_reset):
        #     primal_gap = stepline_pred_reset(t)
        #     if n == 0:
        #         primalgaps_pred_reset = primal_gap
        #     else:
        #         primalgaps_pred_reset = np.vstack((primalgaps_pred_reset, primal_gap))
        # primalgap_pred_ave_reset = np.average(primalgaps_pred_reset, axis=0)

        # primalgaps_regression_k_prime = None
        # for n, stepline in enumerate(steplines_regression_k_primes):
        #     primal_gap = stepline(t)
        #     if n == 0:
        #         primalgaps_regression_k_prime = primal_gap
        #     else:
        #         primalgaps_regression_k_prime = np.vstack((primalgaps_regression_k_prime, primal_gap))
        # primalgap_regression_k_prime_ave = np.average(primalgaps_regression_k_prime, axis=0)

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
        # fig.subplots_adjust(top=0.5)
        ax.set_title(self.instance_type + '-' + test_instance_size + '-' + self.incumbent_mode, loc='right')
        ax.plot(t, primalgap_baseline_ave, label='lb-baseline')
        # ax.plot(t, primalgap_pred_ave, label='lb-regression-noreset')
        # ax.plot(t, primalgap_pred_ave_reset, '--', label='lb-regression')
        # ax.plot(t, primalgap_regression_k_prime_ave, label='lb-regression-k-prime-homo')
        ax.plot(t, primalgap_regression_k_prime_merged_ave, label='lb-regression-k-prime-merged')
        ax.set_xlabel('time /s')
        ax.set_ylabel("normalized primal gap")
        ax.legend()
        plt.show()

    def primal_integral_k_prime_miplib_bianry39(self, test_instance_size, total_time_limit=60, node_time_limit=30):

        # input:
        # baseline
        # k_prime_merged: gnn_merged

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
        #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/'
            # directory_lb_test_2 = directory_2 + 'lb-from-' + 'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(
            #     total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'


        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/'
            # directory_lb_test_2 = directory_2 + 'lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(
            #     total_time_limit) + 's' + test_instance_size + '/'
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

                # filename = f'{directory_lb_test}lb-test-{instance_name}.pkl'
                #
                # with gzip.open(filename, 'rb') as f:
                #     data = pickle.load(f)
                # objs, times, objs_pred, times_pred, objs_pred_reset, times_pred_reset = data  # objs contains objs of a single instance of a lb test
                #
                # filename_2 = f'{directory_lb_test_2}lb-test-{instance_name}.pkl'
                #
                # with gzip.open(filename_2, 'rb') as f:
                #     data = pickle.load(f)
                # objs_2, times_2, objs_pred_2, times_pred_2, objs_pred_reset_2, times_pred_reset_2 = data  # objs contains objs of a single instance of a lb test

                # # test from k_prime
                # filename = f'{directory_lb_test_k_prime}lb-test-{instance_name}.pkl'
                # with gzip.open(filename, 'rb') as f:
                #     data = pickle.load(f)
                # objs_k_prime, times_k_prime = data  # objs contains objs of a single instance of a lb test

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

                # filename = f'{directory_lb_test_k_prime_2}lb-test-{instance_name}.pkl'
                # with gzip.open(filename, 'rb') as f:
                #     data = pickle.load(f)
                # objs_k_prime_2, times_k_prime_2 = data  # objs contains objs of a single instance of a lb test

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

                # objs_k_prime = np.array(objs_k_prime).reshape(-1)
                # times_k_prime = np.array(times_k_prime).reshape(-1)
                #
                # objs_k_prime_2 = np.array(objs_k_prime_2).reshape(-1)

                objs_k_prime_merged = np.array(objs_k_prime_merged).reshape(-1)
                times_k_prime_merged = np.array(times_k_prime_merged).reshape(-1)

                objs_k_prime_merged_2 = np.array(objs_k_prime_merged_2).reshape(-1)

                a = [objs.min(), objs_2.min(), objs_k_prime_merged.min(), objs_k_prime_merged_2.min()]  # objs_2.min(), objs_pred_2.min(), objs_pred_reset_2.min(), objs_k_prime.min(),  objs_k_prime_2.min(),
                # a = [objs.min(), objs_pred.min(), objs_pred_reset.min()]
                obj_opt = np.amin(a)

                # # compute primal gap for baseline localbranching run
                # # if times[-1] < total_time_limit:
                # times = np.append(times, total_time_limit)
                # objs = np.append(objs, objs[-1])
                #
                # gamma_baseline = np.zeros(len(objs))
                # for j in range(len(objs)):
                #     if objs[j] == 0 and obj_opt == 0:
                #         gamma_baseline[j] = 0
                #     elif objs[j] * obj_opt < 0:
                #         gamma_baseline[j] = 1
                #     else:
                #         gamma_baseline[j] = np.abs(objs[j] - obj_opt) / np.maximum(np.abs(objs[j]), np.abs(obj_opt))  #
                #
                # # compute the primal gap of last objective
                # primal_gap_final_baseline = np.abs(objs[-1] - obj_opt) / np.abs(obj_opt)
                # primal_gap_final_baselines.append(primal_gap_final_baseline)
                #
                # # create step line
                # stepline_baseline = interp1d(times, gamma_baseline, 'previous')
                # steplines_baseline.append(stepline_baseline)
                #
                # # compute primal integral
                # primal_int_baseline = 0
                # for j in range(len(objs) - 1):
                #     primal_int_baseline += gamma_baseline[j] * (times[j + 1] - times[j])
                # primal_int_baselines.append(primal_int_baseline)primal_int_baseline

                # # lb-gnn
                # # if times_pred[-1] < total_time_limit:
                # times_pred = np.append(times_pred, total_time_limit)
                # objs_pred = np.append(objs_pred, objs_pred[-1])
                #
                # gamma_pred = np.zeros(len(objs_pred))
                # for j in range(len(objs_pred)):
                #     if objs_pred[j] == 0 and obj_opt == 0:
                #         gamma_pred[j] = 0
                #     elif objs_pred[j] * obj_opt < 0:
                #         gamma_pred[j] = 1
                #     else:
                #         gamma_pred[j] = np.abs(objs_pred[j] - obj_opt) / np.maximum(np.abs(objs_pred[j]),
                #                                                                     np.abs(obj_opt))  #
                #
                # primal_gap_final_pred = np.abs(objs_pred[-1] - obj_opt) / np.abs(obj_opt)
                # primal_gap_final_preds.append(primal_gap_final_pred)
                #
                # stepline_pred = interp1d(times_pred, gamma_pred, 'previous')
                # steplines_pred.append(stepline_pred)
                #
                # # compute primal interal
                # primal_int_pred = 0
                # for j in range(len(objs_pred) - 1):
                #     primal_int_pred += gamma_pred[j] * (times_pred[j + 1] - times_pred[j])
                # primal_int_preds.append(primal_int_pred)
                #
                # # lb-gnn-reset
                # times_pred_reset = np.append(times_pred_reset, total_time_limit)
                # objs_pred_reset = np.append(objs_pred_reset, objs_pred_reset[-1])
                #
                # gamma_pred_reset = np.zeros(len(objs_pred_reset))
                # for j in range(len(objs_pred_reset)):
                #     if objs_pred_reset[j] == 0 and obj_opt == 0:
                #         gamma_pred_reset[j] = 0
                #     elif objs_pred_reset[j] * obj_opt < 0:
                #         gamma_pred_reset[j] = 1
                #     else:
                #         gamma_pred_reset[j] = np.abs(objs_pred_reset[j] - obj_opt) / np.maximum(np.abs(objs_pred_reset[j]),
                #                                                                                 np.abs(obj_opt))  #
                #
                # primal_gap_final_pred_reset = np.abs(objs_pred_reset[-1] - obj_opt) / np.abs(obj_opt)
                # primal_gap_final_preds_reset.append(primal_gap_final_pred_reset)
                #
                # stepline_pred_reset = interp1d(times_pred_reset, gamma_pred_reset, 'previous')
                # steplines_pred_reset.append(stepline_pred_reset)
                #
                # # compute primal interal
                # primal_int_pred_reset = 0
                # for j in range(len(objs_pred_reset) - 1):
                #     primal_int_pred_reset += gamma_pred_reset[j] * (times_pred_reset[j + 1] - times_pred_reset[j])
                # primal_int_preds_reset.append(primal_int_pred_reset)

                # lb-regression-k-prime
                # if times_regression[-1] < total_time_limit:

                # baseline
                primal_int_baseline, primal_gap_final_baseline, stepline_baseline = self.compute_primal_integral(
                    times=times, objs=objs, obj_opt=obj_opt, total_time_limit=total_time_limit)

                primal_gap_final_baselines.append(primal_gap_final_baseline)
                steplines_baseline.append(stepline_baseline)
                primal_int_baselines.append(primal_int_baseline)

                # # regression_k_prime
                # primal_int_regression_k_prime, primal_gap_final_regression_k_prime, stepline_regression_k_prime = self.compute_primal_integral(
                #     times=times_k_prime, objs=objs_k_prime, obj_opt=obj_opt, total_time_limit=total_time_limit)
                #
                # primal_gap_final_regression_k_primes.append(primal_gap_final_regression_k_prime)
                # steplines_regression_k_primes.append(stepline_regression_k_prime)
                # primal_int_regression_k_primes.append(primal_int_regression_k_prime)

                # regression_k_prime_merged
                primal_int_regression_k_prime_merged, primal_gap_final_regression_k_prime_merged, stepline_regression_k_prime_merged = self.compute_primal_integral(
                    times=times_k_prime_merged, objs=objs_k_prime_merged, obj_opt=obj_opt, total_time_limit=total_time_limit)

                primal_gap_final_regression_k_primes_merged.append(primal_gap_final_regression_k_prime_merged)
                steplines_regression_k_primes_merged.append(stepline_regression_k_prime_merged)
                primal_int_regression_k_primes_merged.append(primal_int_regression_k_prime_merged)

                # plt.close('all')
                # plt.clf()
                # fig, ax = plt.subplots(figsize=(8, 6.4))
                # fig.suptitle("Test Result: comparison of objective")
                # fig.subplots_adjust(top=0.5)
                # ax.set_title(instance_name, loc='right')
                # ax.plot(times, objs, label='lb baseline')
                # ax.plot(times_pred, objs_pred, label='lb with k predicted')
                # ax.set_xlabel('time /s')
                # ax.set_ylabel("objective")
                # ax.legend()
                # plt.show()
                #
                # plt.close('all')
                # plt.clf()
                # fig, ax = plt.subplots(figsize=(8, 6.4))
                # fig.suptitle("Test Result: comparison of primal gap")
                # fig.subplots_adjust(top=0.5)
                # ax.set_title(instance_name, loc='right')
                # ax.plot(times, gamma_baseline, label='lb baseline')
                # ax.plot(times_pred, gamma_pred, label='lb with k predicted')
                # ax.set_xlabel('time /s')
                # ax.set_ylabel("objective")
                # ax.legend()
                # plt.show()

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

        # avarage primal integral over test dataset
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
        # print('k_pred primal integral: ', primal_int_pred_ave)
        # print('k_pred_reset primal integral: ', primal_int_pred_ave_reset)
        # print('k_regre_prime primal integral: ', primal_int_regression_k_prime_ave)
        print('k_regre_prime_merged primal integral: ', primal_int_regression_k_prime_merged_ave)
        print('\n')
        print('baseline primal gap: ', primal_gap_final_baselines_ave)
        # print('k_pred primal gap: ', primal_gap_final_preds)
        # print('k_pred_reset primal gap: ', primal_gap_final_preds_reset)
        # print('k_regre_prime primal gap: ', primal_gap_final_regression_k_primes_ave)
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

        # primalgaps_pred = None
        # for n, stepline_pred in enumerate(steplines_pred):
        #     primal_gap = stepline_pred(t)
        #     if n == 0:
        #         primalgaps_pred = primal_gap
        #     else:
        #         primalgaps_pred = np.vstack((primalgaps_pred, primal_gap))
        # primalgap_pred_ave = np.average(primalgaps_pred, axis=0)
        #
        # primalgaps_pred_reset = None
        # for n, stepline_pred_reset in enumerate(steplines_pred_reset):
        #     primal_gap = stepline_pred_reset(t)
        #     if n == 0:
        #         primalgaps_pred_reset = primal_gap
        #     else:
        #         primalgaps_pred_reset = np.vstack((primalgaps_pred_reset, primal_gap))
        # primalgap_pred_ave_reset = np.average(primalgaps_pred_reset, axis=0)

        # primalgaps_regression_k_prime = None
        # for n, stepline in enumerate(steplines_regression_k_primes):
        #     primal_gap = stepline(t)
        #     if n == 0:
        #         primalgaps_regression_k_prime = primal_gap
        #     else:
        #         primalgaps_regression_k_prime = np.vstack((primalgaps_regression_k_prime, primal_gap))
        # primalgap_regression_k_prime_ave = np.average(primalgaps_regression_k_prime, axis=0)

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
        # fig.subplots_adjust(top=0.5)
        ax.set_title(self.instance_type + '-' + test_instance_size + '-' + self.incumbent_mode, loc='right')
        ax.plot(t, primalgap_baseline_ave, label='lb-baseline')
        # ax.plot(t, primalgap_pred_ave, label='lb-regression-noreset')
        # ax.plot(t, primalgap_pred_ave_reset, '--', label='lb-regression')
        # ax.plot(t, primalgap_regression_k_prime_ave, label='lb-regression-k-prime-homo')
        ax.plot(t, primalgap_regression_k_prime_merged_ave, label='lb-regression-k-prime-merged')
        ax.set_xlabel('time /s')
        ax.set_ylabel("normalized primal gap")
        ax.legend()
        plt.show()

class ImitationLocalbranch(MlLocalbranch):
    def __init__(self, instance_type, instance_size, lbconstraint_mode, incumbent_mode,  seed=100):
        super().__init__(instance_type, instance_size, lbconstraint_mode, incumbent_mode,  seed)

    def rltrain_per_instance(self, node_time_limit, total_time_limit, index_instance,
                             reset_k_at_2nditeration=False, policy=None, optimizer=None,
                             criterion=None, device=None):
        """
        evaluate a single MIP instance by two algorithms: lb-baseline and lb-pred_k
        :param node_time_limit:
        :param total_time_limit:
        :param index_instance:
        :return:
        """
        instance = next(self.generator)
        MIP_model = instance.as_pyscipopt()
        MIP_model.setProbName(self.instance_type + '-' + str(index_instance))
        instance_name = MIP_model.getProbName()
        print('\n')
        print(instance_name)

        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

        valid, MIP_model, incumbent_solution = self.initialize_MIP(MIP_model)
        # conti = -1
        # if self.incumbent_mode == 'rootsol' and self.instance_type == 'independentset':
        #     conti = 196

        loss_instance = 0
        accu_instance = 0
        if valid:
            if index_instance > -1:
                gc.collect()
                observation, _, _, done, _ = self.env.reset(instance)
                del observation
                # print(observation)

                if self.incumbent_mode == 'firstsol':
                    action = {'limits/solutions': 1}
                elif self.incumbent_mode == 'rootsol':
                    action = {'limits/nodes': 1}  #
                sample_observation, _, _, done, _ = self.env.step(action)

                # print(sample_observation)
                graph = BipartiteNodeData(sample_observation.constraint_features,
                                          sample_observation.edge_features.indices,
                                          sample_observation.edge_features.values,
                                          sample_observation.variable_features)

                # We must tell pytorch geometric how many nodes there are, for indexing purposes
                graph.num_nodes = sample_observation.constraint_features.shape[0] + \
                                  sample_observation.variable_features.shape[
                                      0]

                # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
                # MIP_model = instance

                # MIP_model.optimize()
                # print("Status:", MIP_model.getStatus())
                # print("best obj: ", MIP_model.getObjVal())
                # print("Solving time: ", MIP_model.getSolvingTime())

                initial_obj = MIP_model.getSolObjVal(incumbent_solution)
                print("Initial obj before LB: {}".format(initial_obj))

                binary_supports = binary_support(MIP_model, incumbent_solution)
                print('binary support: ', binary_supports)

                model_gnn = GNNPolicy()

                model_gnn.load_state_dict(torch.load(
                    self.saved_gnn_directory + 'trained_params_' + self.train_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '.pth'))

                # model_gnn.load_state_dict(torch.load(
                #      'trained_params_' + self.instance_type + '.pth'))

                k_model = model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
                                    graph.variable_features)

                k_pred = k_model.item() * n_binvars
                print('GNN prediction: ', k_model.item())

                if self.is_symmetric == False:
                    k_pred = k_model.item() * binary_supports

                del k_model
                del graph
                del sample_observation
                del model_gnn

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
                lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=k_pred,
                                           node_time_limit=node_time_limit,
                                           total_time_limit=total_time_limit)
                status, obj_best, elapsed_time, lb_bits_pred_reset, times_pred_rest, objs_pred_rest, loss_instance, accu_instance = lb_model3.mdp_localbranch(
                    is_symmetric=self.is_symmetric,
                    reset_k_at_2nditeration=reset_k_at_2nditeration,
                    policy=policy,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    samples_dir=self.imitation_samples_directory)

                print("Instance:", MIP_model_copy3.getProbName())
                print("Status of LB: ", status)
                print("Best obj of LB: ", obj_best)
                print("Solving time: ", elapsed_time)
                print('\n')

                MIP_model_copy3.freeProb()
                del sol_MIP_copy3
                del MIP_model_copy3

                del objs_pred_rest
                del times_pred_rest

                del lb_model3

            index_instance += 1
        del instance
        return index_instance, loss_instance, accu_instance

    def execute_rl4localbranch(self, test_instance_size='-small', total_time_limit=60, node_time_limit=10,
                       reset_k_at_2nditeration=False, lr=0.001, n_epochs=20):

        self.train_dataset = self.instance_type + self.instance_size
        self.evaluation_dataset = self.instance_type + test_instance_size

        self.k_baseline = 20

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_baseline = self.k_baseline / 2
        total_time_limit = total_time_limit
        node_time_limit = node_time_limit

        self.saved_gnn_directory = './result/saved_models/'

        self.imitation_samples_directory = self.directory + 'imitation_samples' + '/'
        pathlib.Path(self.imitation_samples_directory).mkdir(parents=True, exist_ok=True)

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # self.directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
        #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        # pathlib.Path(self.directory_lb_test).mkdir(parents=True, exist_ok=True)

        rl_policy = SimplePolicy(7, 4)
        optimizer = torch.optim.Adam(rl_policy.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        device = self.device

        rl_policy = rl_policy.to(device)
        criterion = criterion.to(device)

        loss = []
        accu = []
        epochs = []

        for epoch in range(n_epochs):
            print(f"Epoch {epoch}")
            if epoch == 0:
                optimizer = None

            self.generator = generator_switcher(self.evaluation_dataset)
            self.generator.seed(self.seed)
            index_instance = 0
            loss_epoch = 0
            accu_epoch = 0
            size_trainset = 10
            while index_instance < size_trainset:
                # train_previous rl_policy
                index_instance, loss_instance, accu_instance = self.rltrain_per_instance(node_time_limit=node_time_limit,
                                                               total_time_limit=total_time_limit,
                                                               index_instance=index_instance,
                                                               reset_k_at_2nditeration=reset_k_at_2nditeration,
                                                               policy=None,
                                                               optimizer=optimizer,
                                                               criterion=criterion,
                                                               device=device
                                                           )
            #     loss_epoch += loss_instance
            #     accu_epoch += accu_instance
            #
            # loss_epoch /= size_trainset
            # accu_epoch /= size_trainset
            #
            # epochs.append(epoch)
            # loss.append(loss_epoch)
            # accu.append(accu_epoch)
            #
            # epochs_np = np.array(epochs).reshape(-1)
            # loss_np = np.array(loss).reshape(-1)
            # accu_np = np.array(accu).reshape(-1)
            #
            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(2, 1, figsize=(8, 6.4))
            # fig.suptitle("Train: loss and imitation accuracy")
            # fig.subplots_adjust(top=0.5)
            # ax[0].set_title('learning rate = ' + str(lr), loc='right')
            # ax[0].plot(epochs_np, loss_np, label='loss')
            # ax[0].set_xlabel('epoch')
            # ax[0].set_ylabel("loss")
            #
            # ax[1].plot(epochs_np, accu_np, label='accuracy')
            # ax[1].set_xlabel('epoch')
            # ax[1].set_ylabel("accuray")
            # ax[1].set_ylim([0, 1.1])
            # ax[1].legend()
            # plt.show()
            #
            # print(f"Train loss: {loss_epoch:0.6f}")
            # print(f"Train accu: {accu_epoch:0.6f}")

    def load_dataset(self, test_dataset_directory=None):

        if test_dataset_directory is not None:
            self.imitation_samples_directory = test_dataset_directory
        else:
            self.imitation_samples_directory = self.directory + 'imitation_samples' + '/'
            pathlib.Path(self.imitation_samples_directory).mkdir(parents=True, exist_ok=True)

        filename = 'imitation_*.pkl'
        # print(filename)
        sample_files = [str(path) for path in pathlib.Path(self.imitation_samples_directory).glob(filename)]
        train_files = sample_files[:int(0.7 * len(sample_files))]
        valid_files = sample_files[int(0.7 * len(sample_files)):int(0.8 * len(sample_files))]
        test_files =  sample_files[int(0.8 * len(sample_files)):]

        train_data = ImitationLbDataset(train_files)

        # state, lab = train_data.__getitem__(0)
        # print(state.shape)
        # print(lab.shape)

        train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
        valid_data = ImitationLbDataset(valid_files)
        valid_loader = DataLoader(valid_data, batch_size=4, shuffle=True)
        test_data = ImitationLbDataset(test_files)
        test_loader = DataLoader(test_data, batch_size=4, shuffle=True)

        return train_loader, valid_loader, test_loader

    def train(self, policy, data_loader, optimizer=None, criterion=None, device=None):
        """
        training function
        :param gnn_model:
        :param data_loader:
        :param optimizer:
        :return:
        """
        loss_epoch = 0
        accu_epoch = 0
        with torch.set_grad_enabled(optimizer is not None):
            for (state, label) in data_loader:

                state.to(device)
                label.to(device)
                label = label.view(-1)

                k_pred = policy(state)
                loss = criterion(k_pred, label)
                accu = imitation_accuracy(k_pred, label)

                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_epoch += loss.item()
                accu_epoch += accu.item()
        loss_mean = loss_epoch / len(data_loader)
        accu_mean = accu_epoch / len(data_loader)

        return loss_mean, accu_mean

    def test(self, policy, data_loader, criterion, device):

        loss_epoch = 0
        accu_epoch = 0
        for (state, label) in data_loader:
            state.to(device)
            label.to(device)
            label = label.view(-1)

            k_pred = policy(state)
            loss = criterion(k_pred, label)
            accu = imitation_accuracy(k_pred, label)

            loss_epoch += loss.item()
            accu_epoch += accu.item()

        loss_mean = loss_epoch / len(data_loader)
        accu_mean = accu_epoch / len(data_loader)

        return loss_mean, accu_mean

    def execute_imitation(self, lr=0.01, n_epochs=20):

        saved_gnn_directory = './result/saved_models/'
        pathlib.Path(saved_gnn_directory).mkdir(parents=True, exist_ok=True)

        train_loaders = {}
        val_loaders = {}
        test_loaders = {}

        # load the small dataset
        small_dataset = self.instance_type + self.instance_size
        self.imitation_samples_directory = self.directory + 'imitation_samples' + '/'

        train_loader, valid_loader, test_loader = self.load_dataset(test_dataset_directory=self.imitation_samples_directory)
        train_loaders[small_dataset] = train_loader
        val_loaders[small_dataset] = valid_loader
        test_loaders[small_dataset] = test_loader

        rl_policy = SimplePolicy(7, 4)
        optimizer = torch.optim.Adam(rl_policy.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        device = self.device

        rl_policy = rl_policy.to(device)
        criterion = criterion.to(device)

        loss = []
        accu = []
        epochs = []

        train_dataset = small_dataset
        valid_dataset = small_dataset
        test_dataset = small_dataset
        # LEARNING_RATE = 0.0000001  # setcovering:0.0000005 cap-loc: 0.00000005 independentset: 0.0000001

        for epoch in range(n_epochs):
            print(f"Epoch {epoch}")

            if epoch == 0:
                optim = None
            else:
                optim = optimizer

            train_loader = train_loaders[train_dataset]
            train_loss, train_accu = self.train(rl_policy, train_loader, optimizer=optim, criterion=criterion, device=device)
            print(f"Train loss: {train_loss:0.6f}")
            print(f"Train accu: {train_accu:0.6f}")

            # torch.save(model_gnn.state_dict(), 'trained_params_' + train_dataset + '.pth')
            # model_gnn2.load_state_dict(torch.load('trained_params_' + train_dataset + '.pth'))

            valid_loader = val_loaders[valid_dataset]
            valid_loss, valid_accu = self.train(rl_policy, valid_loader, optimizer=None, criterion=criterion, device=device)
            print(f"Valid loss: {valid_loss:0.6f}")
            print(f"Valid accu: {valid_accu:0.6f}")

            test_loader = test_loaders[test_dataset]
            test_loss, test_accu = self.test(policy=rl_policy, data_loader=test_loader, criterion=criterion, device=device)

            loss.append(test_loss)
            accu.append(test_accu)
            epochs.append(epoch)

        loss_np = np.array(loss).reshape(-1)
        accu_np = np.array(accu).reshape(-1)
        epochs_np = np.array(epochs).reshape(-1)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(2, 1, figsize=(8, 6.4))
        fig.suptitle("Test: loss and imitation accuracy")
        fig.subplots_adjust(top=0.5)
        ax[0].set_title('learning rate = ' + str(lr), loc='right')
        ax[0].plot(epochs_np, loss_np, label='loss')
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel("loss")

        ax[1].plot(epochs_np, accu_np, label='accuracy')
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel("accuray")
        ax[1].set_ylim([0, 1.1])
        ax[1].legend()
        plt.show()

        torch.save(rl_policy.state_dict(),
                   saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_imitation.pth')

    def evaluate_lb_per_instance(self, node_time_limit, total_time_limit, index_instance, reset_k_at_2nditeration=False, policy=None,
                             criterion=None, device=None):
        """
        evaluate a single MIP instance by two algorithms: lb-baseline and lb-pred_k
        :param node_time_limit:
        :param total_time_limit:
        :param index_instance:
        :return:
        """
        instance = next(self.generator)
        MIP_model = instance.as_pyscipopt()
        MIP_model.setProbName(self.instance_type + '-' + str(index_instance))
        instance_name = MIP_model.getProbName()
        print('\n')
        print(instance_name)

        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

        valid, MIP_model, incumbent_solution = self.initialize_MIP(MIP_model)
        conti = 99
        # if self.incumbent_mode == 'rootsol' and self.instance_type == 'independentset':
        #     conti = 196

        if valid:
            if index_instance > 99 and index_instance > conti:
                gc.collect()
                observation, _, _, done, _ = self.env.reset(instance)
                del observation
                # print(observation)

                if self.incumbent_mode == 'firstsol':
                    action = {'limits/solutions': 1}
                elif self.incumbent_mode == 'rootsol':
                    action = {'limits/nodes': 1}  #
                sample_observation, _, _, done, _ = self.env.step(action)

                # print(sample_observation)
                graph = BipartiteNodeData(sample_observation.constraint_features,
                                          sample_observation.edge_features.indices,
                                          sample_observation.edge_features.values,
                                          sample_observation.variable_features)

                # We must tell pytorch geometric how many nodes there are, for indexing purposes
                graph.num_nodes = sample_observation.constraint_features.shape[0] + \
                                  sample_observation.variable_features.shape[
                                      0]

                # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
                # MIP_model = instance

                # MIP_model.optimize()
                # print("Status:", MIP_model.getStatus())
                # print("best obj: ", MIP_model.getObjVal())
                # print("Solving time: ", MIP_model.getSolvingTime())

                initial_obj = MIP_model.getSolObjVal(incumbent_solution)
                print("Initial obj before LB: {}".format(initial_obj))

                binary_supports = binary_support(MIP_model, incumbent_solution)
                print('binary support: ', binary_supports)

                model_gnn = GNNPolicy()

                model_gnn.load_state_dict(torch.load(
                    self.saved_gnn_directory + 'trained_params_' + self.train_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '.pth'))

                # model_gnn.load_state_dict(torch.load(
                #      'trained_params_' + self.instance_type + '.pth'))

                k_model = model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
                                    graph.variable_features)

                k_pred = k_model.item() * n_binvars
                print('GNN prediction: ', k_model.item())

                if self.is_symmetric == False:
                    k_pred = k_model.item() * binary_supports

                k_pred = np.ceil(k_pred)

                del k_model
                del graph
                del sample_observation
                del model_gnn

                # create a copy of MIP
                MIP_model.resetParams()
                # MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
                #     problemName='Baseline', origcopy=False)
                MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
                    problemName='GNN',
                    origcopy=False)
                MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
                    problemName='GNN+reset',
                    origcopy=False)

                print('MIP copies are created')

                # MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent_solution,
                #                                         MIP_copy_vars)
                MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent_solution,
                                                          MIP_copy_vars2)
                MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent_solution,
                                                          MIP_copy_vars3)

                print('incumbent solution is copied to MIP copies')
                MIP_model.freeProb()
                del MIP_model
                del incumbent_solution

                # sol = MIP_model_copy.getBestSol()
                # initial_obj = MIP_model_copy.getSolObjVal(sol)
                # print("Initial obj before LB: {}".format(initial_obj))

                # # execute local branching baseline heuristic by Fischetti and Lodi
                # lb_model = LocalBranching(MIP_model=MIP_model_copy, MIP_sol_bar=sol_MIP_copy, k=self.k_baseline,
                #                           node_time_limit=node_time_limit,
                #                           total_time_limit=total_time_limit)
                # status, obj_best, elapsed_time, lb_bits, times, objs = lb_model.search_localbranch(is_symmeric=self.is_symmetric,
                #                                                              reset_k_at_2nditeration=False)
                # print("Instance:", MIP_model_copy.getProbName())
                # print("Status of LB: ", status)
                # print("Best obj of LB: ", obj_best)
                # print("Solving time: ", elapsed_time)
                # print('\n')
                #
                # MIP_model_copy.freeProb()
                # del sol_MIP_copy
                # del MIP_model_copy

                # sol = MIP_model_copy2.getBestSol()
                # initial_obj = MIP_model_copy2.getSolObjVal(sol)
                # print("Initial obj before LB: {}".format(initial_obj))

                # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
                lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=k_pred,
                                           node_time_limit=node_time_limit,
                                           total_time_limit=total_time_limit)
                status, obj_best, elapsed_time, lb_bits_pred_reset, times_reset_imitation, objs_reset_imitation, loss_instance, accu_instance = lb_model3.mdp_localbranch(
                    is_symmetric=self.is_symmetric,
                    reset_k_at_2nditeration=reset_k_at_2nditeration,
                    policy=policy,
                    optimizer=None,
                    criterion=criterion,
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
                lb_model2 = LocalBranching(MIP_model=MIP_model_copy2, MIP_sol_bar=sol_MIP_copy2, k=k_pred,
                                           node_time_limit=node_time_limit,
                                           total_time_limit=total_time_limit)
                status, obj_best, elapsed_time, lb_bits_pred, times_reset_vanilla, objs_reset_vanilla, _, _ = lb_model2.mdp_localbranch(
                    is_symmetric=self.is_symmetric,
                    reset_k_at_2nditeration=True,
                    policy=None,
                    optimizer=None,
                    criterion=None,
                    device=None
                )

                print("Instance:", MIP_model_copy2.getProbName())
                print("Status of LB: ", status)
                print("Best obj of LB: ", obj_best)
                print("Solving time: ", elapsed_time)
                print('\n')

                MIP_model_copy2.freeProb()
                del sol_MIP_copy2
                del MIP_model_copy2

                data = [objs_reset_vanilla, times_reset_vanilla, objs_reset_imitation, times_reset_imitation]
                filename = f'{self.directory_lb_test}lb-test-{instance_name}.pkl'  # instance 100-199
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)

                del data
                del lb_model2
                del lb_model3

            index_instance += 1
        del instance
        return index_instance

    def evaluate_localbranching(self, test_instance_size='-small', total_time_limit=60, node_time_limit=30, reset_k_at_2nditeration=False, greedy=True):

        self.train_dataset = self.instance_type + self.instance_size
        self.evaluation_dataset = self.instance_type + test_instance_size

        self.generator = generator_switcher(self.evaluation_dataset)
        self.generator.seed(self.seed)

        self.k_baseline = 20

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_baseline = self.k_baseline / 2
        total_time_limit = total_time_limit
        node_time_limit = node_time_limit

        self.saved_gnn_directory = './result/saved_models/'

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/'
        self.directory_lb_test = directory + 'imitation4lb-from-' + self.incumbent_mode + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        pathlib.Path(self.directory_lb_test).mkdir(parents=True, exist_ok=True)

        rl_policy = SimplePolicy(7, 4)

        rl_policy.load_state_dict(torch.load(
            self.saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_imitation.pth'))

        rl_policy.eval()
        criterion = nn.CrossEntropyLoss()
        device = self.device

        greedy = greedy
        rl_policy = rl_policy.to(device)
        agent = AgentReinforce(rl_policy, device, greedy, None, 0.0)

        index_instance = 0
        while index_instance < 200:
            index_instance = self.evaluate_lb_per_instance(node_time_limit=node_time_limit, total_time_limit=total_time_limit, index_instance=index_instance, reset_k_at_2nditeration=reset_k_at_2nditeration,
                                                           policy=agent, criterion=criterion, device=device
                                                           )

    def primal_integral(self, test_instance_size, total_time_limit=60, node_time_limit=30):

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/'
        directory_lb_test = directory + 'imitation4lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/' + 'rl/'
            directory_lb_test_2 = directory_2 + 'imitation4lb-from-' +  'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/' + 'rl/'
            directory_lb_test_2 = directory_2 + 'imitation4lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        # primal_int_baselines = []
        primal_int_reset_vanillas = []
        primal_in_reset_imitations = []
        # primal_gap_final_baselines = []
        primal_gap_final_reset_vanillas = []
        primal_gap_final_reset_imitations = []
        # steplines_baseline = []
        steplines_reset_vanillas = []
        steplines_reset_imitations = []

        for i in range(100,200):
            if not (i == 148 or i ==113 or i == 110 or i ==199 or i== 198 or i == 134 or i == 123 or i == 116):
                instance_name = self.instance_type + '-' + str(i)  # instance 100-199

                filename = f'{directory_lb_test}lb-test-{instance_name}.pkl'

                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_reset_vanilla, times_reset_vanilla, objs_reset_imitation, times_reset_imitation = data  # objs contains objs of a single instance of a lb test

                filename_2 = f'{directory_lb_test_2}lb-test-{instance_name}.pkl'

                with gzip.open(filename_2, 'rb') as f:
                    data = pickle.load(f)
                objs_reset_vanilla_2, times_reset_vanilla_2, objs_reset_imitation_2, times_reset_imitation_2 = data  # objs contains objs of a single instance of a lb test

                a = [objs_reset_vanilla.min(), objs_reset_imitation.min(), objs_reset_vanilla_2.min(), objs_reset_imitation_2.min()]
                # a = [objs.min(), objs_reset_vanilla.min(), objs_reset_imitation.min()]
                obj_opt = np.amin(a)

                # # compute primal gap for baseline localbranching run
                # # if times[-1] < total_time_limit:
                # times = np.append(times, total_time_limit)
                # objs = np.append(objs, objs[-1])
                #
                # gamma_baseline = np.zeros(len(objs))
                # for j in range(len(objs)):
                #     if objs[j] == 0 and obj_opt == 0:
                #         gamma_baseline[j] = 0
                #     elif objs[j] * obj_opt < 0:
                #         gamma_baseline[j] = 1
                #     else:
                #         gamma_baseline[j] = np.abs(objs[j] - obj_opt) / np.maximum(np.abs(objs[j]), np.abs(obj_opt)) #
                #
                # # compute the primal gap of last objective
                # primal_gap_final_baseline = np.abs(objs[-1] - obj_opt) / np.abs(obj_opt)
                # primal_gap_final_baselines.append(primal_gap_final_baseline)
                #
                # # create step line
                # stepline_baseline = interp1d(times, gamma_baseline, 'previous')
                # steplines_baseline.append(stepline_baseline)
                #
                # # compute primal integral
                # primal_int_baseline = 0
                # for j in range(len(objs) - 1):
                #     primal_int_baseline += gamma_baseline[j] * (times[j + 1] - times[j])
                # primal_int_baselines.append(primal_int_baseline)
                #


                # lb-gnn
                # if times_reset_vanilla[-1] < total_time_limit:
                times_reset_vanilla = np.append(times_reset_vanilla, total_time_limit)
                objs_reset_vanilla = np.append(objs_reset_vanilla, objs_reset_vanilla[-1])

                gamma_reset_vanilla = np.zeros(len(objs_reset_vanilla))
                for j in range(len(objs_reset_vanilla)):
                    if objs_reset_vanilla[j] == 0 and obj_opt == 0:
                        gamma_reset_vanilla[j] = 0
                    elif objs_reset_vanilla[j] * obj_opt < 0:
                        gamma_reset_vanilla[j] = 1
                    else:
                        gamma_reset_vanilla[j] = np.abs(objs_reset_vanilla[j] - obj_opt) / np.maximum(np.abs(objs_reset_vanilla[j]), np.abs(obj_opt)) #

                primal_gap_final_vanilla = np.abs(objs_reset_vanilla[-1] - obj_opt) / np.abs(obj_opt)
                primal_gap_final_reset_vanillas.append(primal_gap_final_vanilla)

                stepline_reset_vanilla = interp1d(times_reset_vanilla, gamma_reset_vanilla, 'previous')
                steplines_reset_vanillas.append(stepline_reset_vanilla)

                #
                # t = np.linspace(start=0.0, stop=total_time_limit, num=1001)
                # plt.close('all')
                # plt.clf()
                # fig, ax = plt.subplots(figsize=(8, 6.4))
                # fig.suptitle("Test Result: comparison of primal gap")
                # fig.subplots_adjust(top=0.5)
                # # ax.set_title(instance_name, loc='right')
                # ax.plot(t, stepline_baseline(t), label='lb baseline')
                # ax.plot(t, stepline_reset_vanilla(t), label='lb with k predicted')
                # ax.set_xlabel('time /s')
                # ax.set_ylabel("objective")
                # ax.legend()
                # plt.show()

                # compute primal interal
                primal_int_reset_vanilla = 0
                for j in range(len(objs_reset_vanilla) - 1):
                    primal_int_reset_vanilla += gamma_reset_vanilla[j] * (times_reset_vanilla[j + 1] - times_reset_vanilla[j])
                primal_int_reset_vanillas.append(primal_int_reset_vanilla)

                # lb-gnn-reset
                times_reset_imitation = np.append(times_reset_imitation, total_time_limit)
                objs_reset_imitation = np.append(objs_reset_imitation, objs_reset_imitation[-1])

                gamma_reset_imitation = np.zeros(len(objs_reset_imitation))
                for j in range(len(objs_reset_imitation)):
                    if objs_reset_imitation[j] == 0 and obj_opt == 0:
                        gamma_reset_imitation[j] = 0
                    elif objs_reset_imitation[j] * obj_opt < 0:
                        gamma_reset_imitation[j] = 1
                    else:
                        gamma_reset_imitation[j] = np.abs(objs_reset_imitation[j] - obj_opt) / np.maximum(np.abs(objs_reset_imitation[j]), np.abs(obj_opt)) #

                primal_gap_final_imitation = np.abs(objs_reset_imitation[-1] - obj_opt) / np.abs(obj_opt)
                primal_gap_final_reset_imitations.append(primal_gap_final_imitation)

                stepline_reset_imitation = interp1d(times_reset_imitation, gamma_reset_imitation, 'previous')
                steplines_reset_imitations.append(stepline_reset_imitation)

                # compute primal interal
                primal_int_reset_imitation = 0
                for j in range(len(objs_reset_imitation) - 1):
                    primal_int_reset_imitation += gamma_reset_imitation[j] * (times_reset_imitation[j + 1] - times_reset_imitation[j])
                primal_in_reset_imitations.append(primal_int_reset_imitation)

                # plt.close('all')
                # plt.clf()
                # fig, ax = plt.subplots(figsize=(8, 6.4))
                # fig.suptitle("Test Result: comparison of objective")
                # fig.subplots_adjust(top=0.5)
                # ax.set_title(instance_name, loc='right')
                # ax.plot(times, objs, label='lb baseline')
                # ax.plot(times_reset_vanilla, objs_reset_vanilla, label='lb with k predicted')
                # ax.set_xlabel('time /s')
                # ax.set_ylabel("objective")
                # ax.legend()
                # plt.show()
                #
                # plt.close('all')
                # plt.clf()
                # fig, ax = plt.subplots(figsize=(8, 6.4))
                # fig.suptitle("Test Result: comparison of primal gap")
                # fig.subplots_adjust(top=0.5)
                # ax.set_title(instance_name, loc='right')
                # ax.plot(times, gamma_baseline, label='lb baseline')
                # ax.plot(times_reset_vanilla, gamma_reset_vanilla, label='lb with k predicted')
                # ax.set_xlabel('time /s')
                # ax.set_ylabel("objective")
                # ax.legend()
                # plt.show()


        # primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_reset_vanilla = np.array(primal_int_reset_vanillas).reshape(-1)
        primal_in_reset_imitation = np.array(primal_in_reset_imitations).reshape(-1)

        # primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_reset_vanilla = np.array(primal_gap_final_reset_vanillas).reshape(-1)
        primal_gap_final_reset_imitation = np.array(primal_gap_final_reset_imitations).reshape(-1)

        # avarage primal integral over test dataset
        # primal_int_base_ave = primal_int_baselines.sum() / len(primal_int_baselines)
        primal_int_reset_vanilla_ave = primal_int_reset_vanilla.sum() / len(primal_int_reset_vanilla)
        primal_int_reset_imitation_ave = primal_in_reset_imitation.sum() / len(primal_in_reset_imitation)

        # primal_gap_final_baselines = primal_gap_final_baselines.sum() / len(primal_gap_final_baselines)
        primal_gap_final_reset_vanilla = primal_gap_final_reset_vanilla.sum() / len(primal_gap_final_reset_vanilla)
        primal_gap_final_reset_imitation = primal_gap_final_reset_imitation.sum() / len(primal_gap_final_reset_imitation)

        print(self.instance_type + self.instance_size)
        print(self.incumbent_mode + 'Solution')
        # print('baseline primal integral: ', primal_int_base_ave)
        print('baseline primal integral: ', primal_int_reset_vanilla_ave)
        print('imitation primal integral: ', primal_int_reset_imitation_ave)
        print('\n')
        # print('baseline primal gap: ',primal_gap_final_baselines)
        print('baseline primal gap: ', primal_gap_final_reset_vanilla)
        print('imitation primal gap: ', primal_gap_final_reset_imitation)

        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)

        # primalgaps_baseline = None
        # for n, stepline_baseline in enumerate(steplines_baseline):
        #     primal_gap = stepline_baseline(t)
        #     if n==0:
        #         primalgaps_baseline = primal_gap
        #     else:
        #         primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        # primalgap_baseline_ave = np.average(primalgaps_baseline, axis=0)

        primalgaps_reset_vanilla = None
        for n, stepline_reset_vanilla in enumerate(steplines_reset_vanillas):
            primal_gap = stepline_reset_vanilla(t)
            if n == 0:
                primalgaps_reset_vanilla = primal_gap
            else:
                primalgaps_reset_vanilla = np.vstack((primalgaps_reset_vanilla, primal_gap))
        primalgap_reset_vanilla_ave = np.average(primalgaps_reset_vanilla, axis=0)

        primalgaps_reset_imitation = None
        for n, stepline_reset_imitation in enumerate(steplines_reset_imitations):
            primal_gap = stepline_reset_imitation(t)
            if n == 0:
                primalgaps_reset_imitation = primal_gap
            else:
                primalgaps_reset_imitation = np.vstack((primalgaps_reset_imitation, primal_gap))
        primalgap_reset_imitation_ave = np.average(primalgaps_reset_imitation, axis=0)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle("Normalized primal gap")
        # fig.subplots_adjust(top=0.5)
        ax.set_title(self.instance_type + '-' + self.incumbent_mode, loc='right')
        # ax.plot(t, primalgap_baseline_ave, label='lb-baseline')
        ax.plot(t, primalgap_reset_vanilla_ave, label='lb-gnn-baseline')
        ax.plot(t, primalgap_reset_imitation_ave,'--', label='lb-gnn-imitation')
        ax.set_xlabel('time /s')
        ax.set_ylabel("normalized primal gap")
        ax.legend()
        plt.show()

class RlLocalbranch(MlLocalbranch):
    def __init__(self, instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed=100):
        super().__init__(instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed)
        self.alpha = 0.01
        self.gamma = 0.99
        self.eps = np.finfo(np.float32).eps.item()

    def load_mip_dataset(self, instances_directory=None, sols_directory=None, incumbent_mode=None):
        instance_filename = f'{self.instance_type}-*_transformed.cip'
        sol_filename = f'{incumbent_mode}-{self.instance_type}-*_transformed.sol'

        train_instances_directory = instances_directory + 'train/'
        instance_files = [str(path) for path in sorted(pathlib.Path(train_instances_directory).glob(instance_filename), key=lambda path: int(path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        instance_train_files = instance_files[:int(7/80 * len(instance_files))]
        instance_valid_files = instance_files[int(7/8 * len(instance_files)):]

        test_instances_directory = instances_directory + 'test/'
        instance_test_files = [str(path) for path in sorted(pathlib.Path(test_instances_directory).glob(instance_filename),
                                                       key=lambda path: int(
                                                           path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        train_sols_directory = sols_directory + 'train/'
        sol_files = [str(path) for path in sorted(pathlib.Path(train_sols_directory).glob(sol_filename), key=lambda path: int(path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        sol_train_files = sol_files[:int(7/80 * len(sol_files))]
        sol_valid_files = sol_files[int(7/8 * len(sol_files)):]

        test_sols_directory = sols_directory + 'test/'
        sol_test_files = [str(path) for path in sorted(pathlib.Path(test_sols_directory).glob(sol_filename),
                                                  key=lambda path: int(path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        train_dataset = InstanceDataset(mip_files=instance_train_files, sol_files=sol_train_files)
        valid_dataset = InstanceDataset(mip_files=instance_valid_files, sol_files=sol_valid_files)
        test_dataset = InstanceDataset(mip_files=instance_test_files, sol_files=sol_test_files)

        return train_dataset, valid_dataset, test_dataset

    def mdp_localbranch(self, localbranch=None, is_symmetric=True, reset_k_at_2nditeration=False, agent=None, optimizer=None, device=None, enable_adapt_t=False):

        # self.total_time_limit = total_time_limit
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
        state, reward, done, _ = localbranch.step_localbranch(k_action=k_action, t_action=t_action, lb_bits=lb_bits)
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

            state, reward, done, _ = localbranch.step_localbranch(k_action=k_action, t_action=t_action,
                                                                   lb_bits=lb_bits)
            localbranch.MIP_obj_init = localbranch.MIP_obj_best
            lb_bits_list.append(lb_bits)
            t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
            obj_list.append(localbranch.MIP_obj_best)
            k_list.append(localbranch.k)

        while not done:  # and localbranch.div < localbranch.div_max
            lb_bits += 1

            k_vanilla, t_action = localbranch.policy_vanilla(state)

            # data_sample = [state, k_vanilla]
            #
            # filename = f'{samples_dir}imitation_{localbranch.MIP_model.getProbName()}_{lb_bits}.pkl'
            #
            # with gzip.open(filename, 'wb') as f:
            #     pickle.dump(data_sample, f)

            k_action = k_vanilla
            if agent is not None:
                k_action = agent.select_action(state)

                # # for online learning, update policy
                # if optimizer is not None:
                #     optimizer.zero_grad()
                #     loss.backward()
                #     optimizer.step()

            # execute one iteration of LB, get the state and rewards

            state, reward, done, _ = localbranch.step_localbranch(k_action=k_action, t_action=t_action, lb_bits=lb_bits, enable_adapt_t=enable_adapt_t)

            if agent is not None:
                agent.rewards.append(reward)

            lb_bits_list.append(lb_bits)
            t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
            obj_list.append(localbranch.MIP_obj_best)
            k_list.append(localbranch.k)

        print(
            'K_final: {:.0f}'.format(localbranch.k),
            'div_final: {:.0f}'.format(localbranch.div)
        )

        localbranch.solve_rightbranch()
        t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
        obj_list.append(localbranch.MIP_obj_best)
        k_list.append(localbranch.k)

        status = localbranch.MIP_model.getStatus()
        # if status == "optimal" or status == "bestsollimit":
        #     localbranch.MIP_obj_best = localbranch.MIP_model.getObjVal()

        elapsed_time = localbranch.total_time_limit - localbranch.total_time_available

        lb_bits_list = np.array(lb_bits_list).reshape(-1)
        times_list = np.array(t_list).reshape(-1)
        objs_list = np.array(obj_list).reshape(-1)
        k_list = np.array(k_list).reshape(-1)

        plt.clf()
        fig, ax = plt.subplots(2, 1, figsize=(8, 6.4))
        fig.suptitle(self.instance_type + 'large' + '-' + self.incumbent_mode, fontsize=13)
        # ax.set_title(self.insancte_type + test_instance_size + '-' + self.incumbent_mode, fontsize=14)

        ax[0].plot(times_list, objs_list, label='lb-rl', color='tab:red')
        ax[0].set_xlabel('time /s', fontsize=12)
        ax[0].set_ylabel("objective", fontsize=12)
        ax[0].legend()
        ax[0].grid()

        ax[1].plot(times_list, k_list, label='lb-rl', color='tab:red')
        ax[1].set_xlabel('time /s', fontsize=12)
        ax[1].set_ylabel("k", fontsize=12)
        ax[1].legend()
        ax[1].grid()
        # fig.suptitle("Scaled primal gap", y=0.97, fontsize=13)
        # fig.tight_layout()
        # plt.savefig(
        #     './result/plots/' + self.instance_type + '_' + self.instance_size + '_' + self.incumbent_mode + '.png')
        plt.show()
        plt.clf()

        del localbranch.subMIP_sol_best
        del localbranch.MIP_sol_bar
        del localbranch.MIP_sol_best


        return status, localbranch.MIP_obj_best, elapsed_time, lb_bits_list, times_list, objs_list, agent

    def train_agent_per_instance(self, MIP_model, incumbent_solution, node_time_limit, total_time_limit, index_instance,
                                 reset_k_at_2nditeration=False, agent=None, optimizer=None,
                                 device=None):
        """
        evaluate a single MIP instance by two algorithms: lb-baseline and lb-pred_k
        :param node_time_limit:
        :param total_time_limit:
        :param index_instance:
        :return:
        """
        gc.collect()

        # filename = f'{self.directory_transformedmodel}{self.instance_type}-{str(index_instance)}_transformed.cip'
        # firstsol_filename = f'{self.directory_sol}{self.incumbent_mode}-{self.instance_type}-{str(index_instance)}_transformed.sol'
        #
        # MIP_model = Model()
        # MIP_model.readProblem(filename)
        # instance_name = MIP_model.getProbName()
        # print(instance_name)
        # n_vars = MIP_model.getNVars()
        # n_binvars = MIP_model.getNBinVars()
        # print("N of variables: {}".format(n_vars))
        # print("N of binary vars: {}".format(n_binvars))
        # print("N of constraints: {}".format(MIP_model.getNConss()))
        #
        # incumbent_solution = MIP_model.readSolFile(firstsol_filename)
        #
        # feas = MIP_model.checkSol(incumbent_solution)
        # try:
        #     MIP_model.addSol(incumbent_solution, False)
        # except:
        #     print('Error: the root solution of ' + instance_name + ' is not feasible!')

        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("Instance: ", MIP_model.getProbName())
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

        instance = ecole.scip.Model.from_pyscipopt(MIP_model)
        observation, _, _, done, _ = self.env.reset(instance)
        graph = BipartiteNodeData(observation.constraint_features,
                                  observation.edge_features.indices,
                                  observation.edge_features.values,
                                  observation.variable_features)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes

        graph.num_nodes = observation.constraint_features.shape[0] + \
                          observation.variable_features.shape[
                              0]

        # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
        # MIP_model = instance

        # MIP_model.optimize()
        # print("Status:", MIP_model.getStatus())
        # print("best obj: ", MIP_model.getObjVal())
        # print("Solving time: ", MIP_model.getSolvingTime())

        initial_obj = MIP_model.getSolObjVal(incumbent_solution)
        print("Initial obj before LB: {}".format(initial_obj))

        binary_supports = binary_support(MIP_model, incumbent_solution)
        print('binary support: ', binary_supports)

        # k_model = self.regression_model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
        #                     graph.variable_features)
        #
        # k_pred = k_model.item() * n_binvars
        # print('GNN prediction: ', k_model.item())
        #
        # if self.is_symmetric == False:
        #     k_pred = k_model.item() * binary_supports
        #
        # del k_model
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
        status, obj_best, elapsed_time, lb_bits_pred_reset, times_pred_reset_, objs_pred_reset_, agent = self.mdp_localbranch(
            localbranch=lb_model3,
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=reset_k_at_2nditeration,
            agent=agent,
            optimizer=optimizer,
            device=device)

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

        MIP_model_copy3.freeProb()
        del sol_MIP_copy3
        del MIP_model_copy3

        del objs_pred_reset
        del times_pred_reset

        del lb_model3
        del stepline

        index_instance += 1
        del instance
        return index_instance, agent, primal_integral, primal_gap_final

    def update_agent(self, agent, optimizer):

        R = 0
        policy_losses = []
        returns = []
        # calculate the return
        for r in agent.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0,R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        # calculate loss
        with torch.set_grad_enabled(optimizer is not None):
            for log_prob, Return in zip(agent.log_probs, returns):
                policy_losses.append(-log_prob * Return)

            # optimize policy network
            if optimizer is not None:
                optimizer.zero_grad()
                policy_losses = torch.cat(policy_losses).sum()
                policy_losses.backward()
                optimizer.step()

        del agent.rewards[:]
        del agent.log_probs[:]
        return agent, optimizer, R


    def train_agent(self, train_instance_size='-small', total_time_limit=60, node_time_limit=10,
                    reset_k_at_2nditeration=False, lr=0.001, n_epochs=20, epsilon=0, use_checkpoint=False):

        train_instance_type = self.instance_type
        direc = './data/generated_instances/' + train_instance_type + '/' + train_instance_size + '/'

        instances_directory = direc + 'transformedmodel' + '/'
        sols_directory = direc + 'firstsol' + '/'
        train_dataset_first, valid_dataset_first, test_dataset_first = self.load_mip_dataset(instances_directory=instances_directory, sols_directory=sols_directory, incumbent_mode='firstsol')
        sols_directory = direc + 'rootsol' + '/'
        train_dataset_root, valid_dataset_root, test_dataset_root = self.load_mip_dataset(instances_directory=instances_directory,
                                                                           sols_directory=sols_directory, incumbent_mode='rootsol')
        train_datasets = [train_dataset_first, train_dataset_root]
        train_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(train_dataset_first, shuffle=True, batch_size=1, collate_fn=custom_collate)
        size_trainset = len(train_loader.dataset)
        print(size_trainset)

        device = self.device
        self.regression_dataset = train_instance_type + '-small'
        # self.evaluation_dataset = self.instance_type + train_instance_size

        # direc = './data/generated_instances/' + self.instance_type + '/' + train_instance_size + '/'
        # self.directory_transformedmodel = direc + 'transformedmodel' + '/'
        # self.directory_sol = direc + self.incumbent_mode + '/'

        self.k_baseline = 20

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_baseline = self.k_baseline / 2
        total_time_limit = total_time_limit
        node_time_limit = node_time_limit

        self.saved_model_directory = './result/saved_models/'
        # self.regression_model_gnn = GNNPolicy()
        # self.regression_model_gnn.load_state_dict(torch.load(
        #     self.saved_model_directory + 'trained_params_' + self.regression_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '.pth'))
        # self.regression_model_gnn.to(device)

        self.saved_rlmodels_directory = self.saved_model_directory + 'rl/reinforce/' + train_instance_type + '/'
        pathlib.Path( self.saved_rlmodels_directory).mkdir(parents=True, exist_ok=True)

        train_directory = './result/generated_instances/' + self.instance_type + '/' + train_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        self.reinforce_train_directory = train_directory + 'rl/' + 'reinforce/train/data/'
        pathlib.Path(self.reinforce_train_directory).mkdir(parents=True, exist_ok=True)
        # self.directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
        #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        # pathlib.Path(self.directory_lb_test).mkdir(parents=True, exist_ok=True)

        rl_policy = SimplePolicy(7, 4)
        # rl_policy.load_state_dict(torch.load(
        #     # self.saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_reinforce_lr0.1_epsilon0.0_pre.pth'
        #     self.saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_imitation.pth'
        # ))
        rl_policy.train()


        optim = torch.optim.Adam(rl_policy.parameters(), lr=lr)

        greedy = False
        rl_policy = rl_policy.to(device)
        agent = AgentReinforce(rl_policy, device, greedy, optim, epsilon)

        returns = []
        epochs = []
        primal_integrals = []
        primal_gaps = []
        data = None
        epochs_np = None
        returns_np = None
        primal_integrals_np = None
        primal_gaps_np = None
        epoch_init = 0
        epoch_start = epoch_init  # 50
        epoch_end = epoch_start+n_epochs+1

        if use_checkpoint:
            checkpoint = torch.load(
                # self.saved_gnn_directory + 'checkpoint_simplepolicy_rl4lb_reinforce_lr' + str(lr) + '_epsilon' + str(epsilon) + '.pth'
                # self.saved_rlmodels_directory + 'good_models/' + 'checkpoint_noregression_noimitation_reward3_simplepolicy_rl4lb_reinforce_lr' + str(
                #     lr) + '_epsilon' + str(epsilon) + '_epoch210.pth'

                self.saved_rlmodels_directory + 'checkpoint_trained_reward3_simplepolicy_rl4lb_reinforce_trainset_' + train_instance_type + train_instance_size + '_0.1trainset_lr' + str(lr) + '_epochs' + str(45) + '.pth'
                # self.saved_rlmodels_directory + 'checkpoint_trained_reward3_simplepolicy_rl4lb_reinforce_trainset_' + train_instance_type + train_instance_size + '_lr' + str(lr) + '_epochs' + str(3) + '.pth'

            )
            rl_policy.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            data = checkpoint['loss_data']
            epochs, returns, primal_integrals, primal_gaps = data
            rl_policy.train()

            epoch_start = checkpoint['epoch'] + 1
            epoch_end = epoch_start + n_epochs
            optimizer = optim

        for epoch in range(epoch_start,epoch_end):
            del data
            print(f"Epoch {epoch}")
            if epoch == epoch_init:
                optimizer = None
            elif epoch == epoch_init + 1:
                optimizer = optim

            index_instance = 0
            # size_trainset = 5
            return_epoch = 0
            primal_integral_epoch = 0
            primal_gap_epoch = 0

            # while index_instance < size_trainset:
            for batch in (train_loader):
                MIP_model = batch['mip_model'][0]
                incumbent_solution = batch['incumbent_solution'][0]
                # train_previous rl_policy
                index_instance, agent, primal_integral, primal_gap_final = self.train_agent_per_instance(MIP_model=MIP_model,
                                                                                                    incumbent_solution=incumbent_solution,
                                                                                                    node_time_limit=node_time_limit,
                                                                                                    total_time_limit=total_time_limit,
                                                                                                    index_instance=index_instance,
                                                                                                    reset_k_at_2nditeration=reset_k_at_2nditeration,
                                                                                                    agent=agent,
                                                                                                    optimizer=optimizer,
                                                                                                    device=device
                                                                                                    )

                agent, optimizer, R = self.update_agent(agent, optimizer)
                return_epoch += R

                primal_integral_epoch += primal_integral
                primal_gap_epoch += primal_gap_final

            return_epoch = return_epoch/size_trainset
            primal_integral_epoch = primal_integral_epoch/size_trainset
            primal_gap_epoch = primal_gap_epoch/size_trainset

            returns.append(return_epoch)
            epochs.append(epoch)
            primal_integrals.append(primal_integral_epoch)
            primal_gaps.append(primal_gap_epoch)

            print(f"Return: {return_epoch:0.6f}")
            print(f"Primal ingtegral: {primal_integral_epoch:0.6f}")

            data = [epochs, returns, primal_integrals, primal_gaps]

            if epoch > 0:
                filename = f'{self.reinforce_train_directory}lb-rl-checkpoint-reward3-simplepolicy-0.1trainset-lr{str(lr)}-epochs{str(epoch)}.pkl'  # instance 10% of testset
                # filename = f'{self.reinforce_train_directory}lb-rl-checkpoint-reward3-simplepolicy-lr{str(lr)}-epochs{str(epoch)}.pkl'

                # filename = f'{self.reinforce_train_directory}lb-rl-noregression-noimitation-reward3-train-lr{str(lr)}-epsilon{str(epsilon)}_60s_talored.pkl'  # instance 100-199
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)

                # save checkpoint
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': rl_policy.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss_data':data,
                            },
                           # self.saved_rlmodels_directory + 'checkpoint_noregression_noimitation_reward3_simplepolicy_rl4lb_reinforce_lr' + str(lr) + '_epsilon' + str(epsilon) + '_60s_talored4independentset-small-firstsol.pth'

                            self.saved_rlmodels_directory + 'checkpoint_trained_reward3_simplepolicy_rl4lb_reinforce_trainset_' + train_instance_type + train_instance_size + '_0.1trainset_lr' + str(lr) + '_epochs' + str(epoch) + '.pth'
                            # self.saved_rlmodels_directory + 'checkpoint_trained_reward3_simplepolicy_rl4lb_reinforce_trainset_' + train_instance_type + train_instance_size + '_lr' + str(lr) + '_epochs' + str(epoch) + '.pth'

                           # self.saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_reinforce_trainset_' + train_instance_type + train_instance_size + '_lr' + str(lr) + '_epsilon' + str(epsilon) + '.pth'
                           )
                # torch.save(rl_policy.state_dict(),
                #            # self.saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_reinforce-checkpoint50_lr' + str(lr) + '_epsilon' + str(epsilon) + '.pth'
                #            self.saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_reinforce_lr' + str(lr) +'_epsilon' + str(epsilon) + '.pth'
                #            )

        epochs_np = np.array(epochs).reshape(-1)
        returns_np = np.array(returns).reshape(-1)
        primal_integrals_np = np.array(primal_integrals).reshape(-1)
        primal_gaps_np = np.array(primal_gaps).reshape(-1)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(3, 1, figsize=(8, 6.4))
        fig.suptitle(self.regression_dataset)
        fig.subplots_adjust(top=0.5)
        ax[0].set_title('lr= ' + str(lr) + ', epsilon=' + str(epsilon), loc='right')
        ax[0].plot(epochs_np, returns_np, label='loss')
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel("return")

        ax[1].plot(epochs_np, primal_integrals_np, label='primal ingegral')
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel("primal integral")
        # ax[1].set_ylim([0, 1.1])
        ax[1].legend()

        ax[2].plot(epochs_np, primal_gaps_np, label='primal gap')
        ax[2].set_xlabel('epoch')
        ax[2].set_ylabel("primal gap")
        ax[2].legend()
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
        graph = BipartiteNodeData(observation.constraint_features,
                                  observation.edge_features.indices,
                                  observation.edge_features.values,
                                  observation.variable_features)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = observation.constraint_features.shape[0] + \
                          observation.variable_features.shape[
                              0]
        # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
        # MIP_model = instance

        # MIP_model.optimize()
        # print("Status:", MIP_model.getStatus())
        # print("best obj: ", MIP_model.getObjVal())
        # print("Solving time: ", MIP_model.getSolvingTime())

        initial_obj = MIP_model.getSolObjVal(incumbent_solution)
        print("Initial obj before LB: {}".format(initial_obj))

        binary_supports = binary_support(MIP_model, incumbent_solution)
        print('binary support: ', binary_supports)

        # model_gnn.load_state_dict(torch.load(
        #      'trained_params_' + self.instance_type + '.pth'))

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
        # MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
        #     problemName='Baseline', origcopy=False)
        MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
            problemName='noregression-rl',
            origcopy=False)
        MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
            problemName='regression-rl',
            origcopy=False)

        print('MIP copies are created')

        # MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent_solution,
        #                                         MIP_copy_vars)
        MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent_solution,
                                                  MIP_copy_vars2)
        MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent_solution,
                                                  MIP_copy_vars3)

        print('incumbent solution is copied to MIP copies')
        MIP_model.freeProb()
        del MIP_model
        del incumbent_solution

        # sol = MIP_model_copy.getBestSol()
        # initial_obj = MIP_model_copy.getSolObjVal(sol)
        # print("Initial obj before LB: {}".format(initial_obj))

        # # execute local branching baseline heuristic by Fischetti and Lodi
        # lb_model = LocalBranching(MIP_model=MIP_model_copy, MIP_sol_bar=sol_MIP_copy, k=self.k_baseline,
        #                           node_time_limit=node_time_limit,
        #                           total_time_limit=total_time_limit)
        # status, obj_best, elapsed_time, lb_bits, times, objs = lb_model.search_localbranch(is_symmeric=self.is_symmetric,
        #                                                              reset_k_at_2nditeration=False)
        # print("Instance:", MIP_model_copy.getProbName())
        # print("Status of LB: ", status)
        # print("Best obj of LB: ", obj_best)
        # print("Solving time: ", elapsed_time)
        # print('\n')
        #
        # MIP_model_copy.freeProb()
        # del sol_MIP_copy
        # del MIP_model_copy

        # sol = MIP_model_copy2.getBestSol()
        # initial_obj = MIP_model_copy2.getSolObjVal(sol)
        # print("Initial obj before LB: {}".format(initial_obj))

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
        # self.evaluation_dataset = self.instance_type + evaluation_instance_size

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

        self.saved_rlmodels_directory = self.saved_gnn_directory + 'rl_noimitation/'
        # checkpoint = torch.load(
        #     self.saved_rlmodels_directory + 'checkpoint_noimitation_reward2_simplepolicy_rl4lb_reinforce_lr0.05_epsilon0.0.pth')
        # rl_policy.load_state_dict(checkpoint['model_state_dict'])
        checkpoint = torch.load(
            # self.saved_gnn_directory + 'checkpoint_simplepolicy_rl4lb_reinforce_lr' + str(lr) + '_epsilon' + str(epsilon) + '.pth'
            self.saved_rlmodels_directory + 'checkpoint_noregression_noimitation_reward3_simplepolicy_rl4lb_reinforce_lr0.01_epsilon0.0_60s_talored4independentset-small-firstsol.pth'
        )
        rl_policy.load_state_dict(checkpoint['model_state_dict'])

        # rl_policy.load_state_dict(torch.load(
        #     self.saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_reinforce_lr0.1_epsilon0.0_pre.pth'))

        rl_policy.eval()
        # criterion = nn.CrossEntropyLoss()

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

        # filename = f'{self.directory_transformedmodel}{self.instance_type}-{str(index_instance)}_transformed.cip'
        # firstsol_filename = f'{self.directory_sol}{self.incumbent_mode}-{self.instance_type}-{str(index_instance)}_transformed.sol'

        # MIP_model = Model()
        # MIP_model.readProblem(filename)
        # incumbent_solution = MIP_model.readSolFile(firstsol_filename)

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

        instance = ecole.scip.Model.from_pyscipopt(MIP_model)
        observation, _, _, done, _ = self.env.reset(instance)

        # variable features: only incumbent solution
        variable_features = observation.variable_features[:, -1:]
        graph = BipartiteNodeData(observation.constraint_features,
                                  observation.edge_features.indices,
                                  observation.edge_features.values,
                                  variable_features)

        # graph = BipartiteNodeData(observation.constraint_features,
        #                           observation.edge_features.indices,
        #                           observation.edge_features.values,
        #                           observation.variable_features)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = observation.constraint_features.shape[0] + \
                          observation.variable_features.shape[
                              0]
        # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
        # MIP_model = instance

        # MIP_model.optimize()
        # print("Status:", MIP_model.getStatus())
        # print("best obj: ", MIP_model.getObjVal())
        # print("Solving time: ", MIP_model.getSolvingTime())

        # solve the root node and get the LP solution, compute k_prime
        k_prime = self.compute_k_prime(MIP_model, incumbent)

        initial_obj = MIP_model.getSolObjVal(incumbent_solution)
        print("Initial obj before LB: {}".format(initial_obj))

        binary_supports = binary_support(MIP_model, incumbent_solution)
        print('binary support: ', binary_supports)

        # model_gnn.load_state_dict(torch.load(
        #      'trained_params_' + self.instance_type + '.pth'))

        k_model = self.regression_model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
                                            graph.variable_features)

        k_pred = k_model.item() * k_prime
        print('GNN prediction: ', k_model.item())

        if self.is_symmetric == False:
            k_pred = k_model.item() * k_prime

        k_pred = np.ceil(k_pred)

        del k_model
        del graph
        del observation

        # create a copy of MIP
        MIP_model.resetParams()
        # MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
        #     problemName='Baseline', origcopy=False)
        MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
            problemName='noregression-rl',
            origcopy=False)
        MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
            problemName='regression-rl',
            origcopy=False)

        print('MIP copies are created')

        # MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent_solution,
        #                                         MIP_copy_vars)
        MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent_solution,
                                                  MIP_copy_vars2)
        MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent_solution,
                                                  MIP_copy_vars3)

        print('incumbent solution is copied to MIP copies')
        MIP_model.freeProb()
        del MIP_model
        del incumbent_solution

        # sol = MIP_model_copy.getBestSol()
        # initial_obj = MIP_model_copy.getSolObjVal(sol)
        # print("Initial obj before LB: {}".format(initial_obj))

        # # execute local branching baseline heuristic by Fischetti and Lodi
        # lb_model = LocalBranching(MIP_model=MIP_model_copy, MIP_sol_bar=sol_MIP_copy, k=self.k_baseline,
        #                           node_time_limit=node_time_limit,
        #                           total_time_limit=total_time_limit)
        # status, obj_best, elapsed_time, lb_bits, times, objs = lb_model.search_localbranch(is_symmeric=self.is_symmetric,
        #                                                              reset_k_at_2nditeration=False)
        # print("Instance:", MIP_model_copy.getProbName())
        # print("Status of LB: ", status)
        # print("Best obj of LB: ", obj_best)
        # print("Solving time: ", elapsed_time)
        # print('\n')
        #
        # MIP_model_copy.freeProb()
        # del sol_MIP_copy
        # del MIP_model_copy

        # sol = MIP_model_copy2.getBestSol()
        # initial_obj = MIP_model_copy2.getSolObjVal(sol)
        # print("Initial obj before LB: {}".format(initial_obj))

        # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
        lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=k_pred,
                                   node_time_limit=node_time_limit,
                                   total_time_limit=total_time_limit)
        # status, obj_best, elapsed_time, lb_bits_pred_reset, times_regression_reinforce, objs_regression_reinforce, loss_instance, accu_instance = lb_model3.mdp_localbranch(
        #     is_symmetric=self.is_symmetric,
        #     reset_k_at_2nditeration=reset_k_at_2nditeration,
        #     policy=agent1,
        #     optimizer=None,
        #     device=device
        # )

        status, obj_best, elapsed_time, lb_bits_pred_reset, times_regression_reinforce_, objs_regression_reinforce_, agent1 = self.mdp_localbranch(
            localbranch=lb_model3,
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=reset_k_at_2nditeration,
            agent=agent1,
            optimizer=None,
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
        # status, obj_best, elapsed_time, lb_bits_pred, times_noregression_reinforce, objs_noregression_reinforce, _, _ = lb_model2.mdp_localbranch(
        #     is_symmetric=self.is_symmetric,
        #     reset_k_at_2nditeration=False,
        #     policy=agent2,
        #     optimizer=None,
        #     device=device
        # )

        status, obj_best, elapsed_time, lb_bits_pred, times_noregression_reinforce_, objs_noregression_reinforce_, agent2 = self.mdp_localbranch(
            localbranch=lb_model2,
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=reset_k_at_2nditeration,
            agent=agent2,
            optimizer=None,
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
        # saved_name = f'{self.instance_type}-{str(index_instance)}_transformed'
        filename = f'{self.directory_lb_test}lb-test-{instance_name}.pkl'  # instance 100-199
        # with gzip.open(filename, 'wb') as f:
        #     pickle.dump(data, f)

        del data
        del lb_model2
        del lb_model3

        # index_instance += 1
        return agent1, agent2

    def evaluate_localbranching_rlactive(self, evaluation_instance_size='-small', total_time_limit=60, node_time_limit=30,
                                reset_k_at_2nditeration=False, greedy=False, lr=None, regression_model_path='',
                    rl_model_path='', enable_adapt_t=False):

        self.regression_dataset = self.instance_type + '-small'
        # self.evaluation_dataset = self.instance_type + evaluation_instance_size

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
            total_time_limit) + 's' + evaluation_instance_size + '/rlactive_t_node_baseline/seed'+ str(self.seed) + '/'
        else:
            self.directory_lb_test = evaluation_directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(
                total_time_limit) + 's' + evaluation_instance_size + '/rlactive/seed' + str(
                self.seed) + '/'
        pathlib.Path(self.directory_lb_test).mkdir(parents=True, exist_ok=True)

        rl_policy1 = SimplePolicy(7, 4)
        rl_policy2 = SimplePolicy(7, 4)

        # self.saved_rlmodels_directory = self.saved_gnn_directory + 'rl_noimitation/good_models/'
        self.saved_rlmodels_directory = self.saved_model_directory + 'rl/reinforce/' + 'setcovering' + '/'

        # checkpoint = torch.load(
        #     self.saved_rlmodels_directory + 'checkpoint_noimitation_reward2_simplepolicy_rl4lb_reinforce_lr0.05_epsilon0.0.pth')
        # rl_policy.load_state_dict(checkpoint['model_state_dict'])
        checkpoint = torch.load(
            # self.saved_gnn_directory + 'checkpoint_simplepolicy_rl4lb_reinforce_lr' + str(lr) + '_epsilon' + str(epsilon) + '.pth'
            # self.saved_model_directory + '/rl_noimitation/good_models/checkpoint_noregression_noimitation_reward3_simplepolicy_rl4lb_reinforce_lr0.01_epsilon0.0_epoch210.pth'
            rl_model_path
        )
        rl_policy1.load_state_dict(checkpoint['model_state_dict'])
        rl_policy2.load_state_dict(checkpoint['model_state_dict'])

        # rl_policy.load_state_dict(torch.load(
        #     self.saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_reinforce_lr0.1_epsilon0.0_pre.pth'))

        rl_policy1.train()
        rl_policy2.train()
        # criterion = nn.CrossEntropyLoss()

        optim1 = torch.optim.Adam(rl_policy1.parameters(), lr=lr)
        optim2 = torch.optim.Adam(rl_policy2.parameters(), lr=lr)

        optim1.load_state_dict(checkpoint['optimizer_state_dict'])
        optim2.load_state_dict(checkpoint['optimizer_state_dict'])

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

    def primal_integral(self, test_instance_size, total_time_limit=60, node_time_limit=30):

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/reinforce/test/'
        directory_lb_test = directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/'

        # directory_rl_talored = directory_lb_test + 'rlactive/'
        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/' + 'rl/reinforce/test/'
            directory_lb_test_2 = directory_2 + 'evaluation-reinforce4lb-from-' +  'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/'
        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/' + 'rl/reinforce/test/'
            directory_lb_test_2 = directory_2 + 'evaluation-reinforce4lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/'

        # directory_3 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # directory_lb_test_3 = directory_3 + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
        #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'


        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
        #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/'
            # directory_lb_test_2 = directory_2 + 'lb-from-' + 'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(
            #     total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'


        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/'
            # directory_lb_test_2 = directory_2 + 'lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(
            #     total_time_limit) + 's' + test_instance_size + '/'
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

        # primal_int_regression_reinforces_talored = []
        # primal_int_reinforces_talored = []
        # primal_gap_final_regression_reinforces_talored = []
        # primal_gap_final_reinforces_talored = []
        # steplines_regression_reinforce_talored = []
        # steplines_reinforce_talored = []

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

            # filename_3 = f'{directory_lb_test_3}lb-test-{instance_name}.pkl'
            #
            # with gzip.open(filename_3, 'rb') as f:
            #     data = pickle.load(f)
            # objs, times, objs_pred_2, times_pred_2, objs_pred_reset_2, times_pred_reset_2 = data  # objs contains objs of a single instance of a lb test
            #
            # objs_regression = objs_pred_reset_2
            # times_regression = times_pred_reset_2

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

            # a = [objs_regression.min(), objs_regresison_reinforce.min(), objs_reset_vanilla_2.min(), objs_reset_imitation_2.min()]
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

            #
            # t = np.linspace(start=0.0, stop=total_time_limit, num=1001)
            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of primal gap")
            # fig.subplots_adjust(top=0.5)
            # # ax.set_title(instance_name, loc='right')
            # ax.plot(t, stepline_baseline(t), label='lb baseline')
            # ax.plot(t, stepline_reset_vanilla(t), label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()

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

            # # lb-regression-reinforce-talored
            # primal_int_regression_reinforce_talored, primal_gap_final_regression_reinforce_talored, stepline_regression_reinforce_talored = self.compute_primal_integral(
            #     times=times_regression_reinforce_talored, objs=objs_regresison_reinforce_talored, obj_opt=obj_opt,
            #     total_time_limit=total_time_limit)
            # primal_gap_final_regression_reinforces_talored.append(primal_gap_final_regression_reinforce_talored)
            # steplines_regression_reinforce_talored.append(stepline_regression_reinforce_talored)
            # primal_int_regression_reinforces_talored.append(primal_int_regression_reinforce_talored)
            #
            # # lb-reinforce
            #
            # primal_int_reinforce_talored, primal_gap_final_reinforce_talored, stepline_reinforce_talored = self.compute_primal_integral(
            #     times=times_reinforce_talored, objs=objs_reinforce_talored, obj_opt=obj_opt,
            #     total_time_limit=total_time_limit)
            # primal_gap_final_reinforces_talored.append(primal_gap_final_reinforce_talored)
            # steplines_reinforce_talored.append(stepline_reinforce_talored)
            # primal_int_reinforces_talored.append(primal_int_reinforce_talored)

            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of objective")
            # fig.subplots_adjust(top=0.5)
            # ax.set_title(instance_name, loc='right')
            # ax.plot(times, objs, label='lb baseline')
            # ax.plot(times_regression, objs_regression, label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()
            #
            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of primal gap")
            # fig.subplots_adjust(top=0.5)
            # ax.set_title(instance_name, loc='right')
            # ax.plot(times, gamma_baseline, label='lb baseline')
            # ax.plot(times_regression, gamma_reset_vanilla, label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()


        primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_regressions = np.array(primal_int_regressions).reshape(-1)
        primal_int_regressions_merged = np.array(primal_int_regressions_merged).reshape(-1)
        primal_int_regression_reinforces = np.array(primal_int_regression_reinforces).reshape(-1)
        primal_int_reinforces = np.array(primal_int_reinforces).reshape(-1)

        # primal_int_regression_reinforces_talored = np.array(primal_int_regression_reinforces_talored).reshape(-1)
        # primal_int_reinforces_talored = np.array(primal_int_reinforces_talored).reshape(-1)

        primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_regressions = np.array(primal_gap_final_regressions).reshape(-1)
        primal_gap_final_regressions_merged = np.array(primal_gap_final_regressions_merged).reshape(-1)
        primal_gap_final_regression_reinforces = np.array(primal_gap_final_regression_reinforces).reshape(-1)
        primal_gap_final_reinforces = np.array(primal_gap_final_reinforces).reshape(-1)

        # primal_gap_final_regression_reinforces_talored = np.array(primal_gap_final_regression_reinforces_talored).reshape(-1)
        # primal_gap_final_reinforces_talored = np.array(primal_gap_final_reinforces_talored).reshape(-1)

        # avarage primal integral over test dataset
        primal_int_base_ave = primal_int_baselines.sum() / len(primal_int_baselines)
        primal_int_regression_ave = primal_int_regressions.sum() / len(primal_int_regressions)
        primal_int_regression_merged_ave = primal_int_regressions_merged.sum() / len(primal_int_regressions_merged)
        primal_int_regression_reinforce_ave = primal_int_regression_reinforces.sum() / len(primal_int_regression_reinforces)
        primal_int_reinforce_ave = primal_int_reinforces.sum() / len(
            primal_int_reinforces)

        # primal_int_regression_reinforce_talored_ave = primal_int_regression_reinforces_talored.sum() / len(primal_int_regression_reinforces_talored)
        # primal_int_reinforce_talored_ave = primal_int_reinforces_talored.sum() / len(
        #     primal_int_reinforces_talored)

        primal_gap_final_baseline_ave = primal_gap_final_baselines.sum() / len(primal_gap_final_baselines)
        primal_gap_final_regression_ave = primal_gap_final_regressions.sum() / len(primal_gap_final_regressions)
        primal_gap_final_regression_merged_ave = primal_gap_final_regressions_merged.sum() / len(primal_gap_final_regressions_merged)
        primal_gap_final_regression_reinforce_ave = primal_gap_final_regression_reinforces.sum() / len(primal_gap_final_regression_reinforces)
        primal_gap_final_reinforce_ave = primal_gap_final_reinforces.sum() / len(
            primal_gap_final_reinforces)

        # primal_gap_final_regression_reinforce_talored_ave = primal_gap_final_regression_reinforces_talored.sum() / len(
        #     primal_gap_final_regression_reinforces_talored)
        # primal_gap_final_reinforce_talored_ave = primal_gap_final_reinforces_talored.sum() / len(
        #     primal_gap_final_reinforces_talored)

        print(self.instance_type + test_instance_size)
        print(self.incumbent_mode + 'Solution')
        print('baseline primal integral: ', primal_int_base_ave)
        print('regression primal integral: ', primal_int_regression_ave)
        print('regression merged primal integral: ', primal_int_regression_merged_ave)
        print('rl primal integral: ', primal_int_reinforce_ave)
        print('regression-rl primal integral: ', primal_int_regression_reinforce_ave)

        print('\n')
        print('baseline primal gap: ', primal_gap_final_baseline_ave)
        print('regression primal gap: ', primal_gap_final_regression_ave)
        print('regression primal merged gap: ', primal_gap_final_regression_merged_ave)
        print('rl primal gap: ', primal_gap_final_reinforce_ave)
        print('regression-rl primal gap: ', primal_gap_final_regression_reinforce_ave)

        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)

        primalgaps_baseline = None
        for n, stepline_baseline in enumerate(steplines_baseline):
            primal_gap = stepline_baseline(t)
            if n==0:
                primalgaps_baseline = primal_gap
            else:
                primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        primalgap_baseline_ave = np.average(primalgaps_baseline, axis=0)

        primalgaps_regression = None
        for n, stepline_regression in enumerate(steplines_regression):
            primal_gap = stepline_regression(t)
            if n == 0:
                primalgaps_regression = primal_gap
            else:
                primalgaps_regression = np.vstack((primalgaps_regression, primal_gap))
        primalgap_regression_ave = np.average(primalgaps_regression, axis=0)

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

        # primalgaps_regression_reinforce_talored = None
        # for n, stepline_regression_reinforce in enumerate(steplines_regression_reinforce_talored):
        #     primal_gap = stepline_regression_reinforce(t)
        #     if n == 0:
        #         primalgaps_regression_reinforce_talored = primal_gap
        #     else:
        #         primalgaps_regression_reinforce_talored = np.vstack((primalgaps_regression_reinforce_talored, primal_gap))
        # primalgap_regression_reinforce_talored_ave = np.average(primalgaps_regression_reinforce_talored, axis=0)
        #
        # primalgaps_reinforce_talored = None
        # for n, stepline_reinforce in enumerate(steplines_reinforce_talored):
        #     primal_gap = stepline_reinforce(t)
        #     if n == 0:
        #         primalgaps_reinforce_talored = primal_gap
        #     else:
        #         primalgaps_reinforce_talored = np.vstack((primalgaps_reinforce_talored, primal_gap))
        # primalgap_reinforce_talored_ave = np.average(primalgaps_reinforce_talored, axis=0)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle(self.instance_type + test_instance_size + '-' + self.incumbent_mode, fontsize=13)
        # ax.set_title(self.insancte_type + test_instance_size + '-' + self.incumbent_mode, fontsize=14)
        ax.plot(t, primalgap_baseline_ave, label='lb-base', color='tab:blue')
        ax.plot(t, primalgap_regression_ave, label='lb-sr', color ='tab:grey')
        ax.plot(t, primalgap_regression_merged_ave, label='lb-srm', color='tab:orange')
        ax.plot(t, primalgap_reinforce_ave, '--', label='lb-rl', color='tab:green')
        ax.plot(t, primalgap_regression_reinforce_ave,'--', label='lb-srmrl', color='tab:red')
        #
        # ax.plot(t, primalgap_reinforce_talored_ave, ':', label='lb-rl-active', color='tab:green')
        # ax.plot(t, primalgap_regression_reinforce_talored_ave, ':', label='lb-regression-rl-active', color='tab:red')

        ax.set_xlabel('time /s', fontsize=12)
        ax.set_ylabel("scaled primal gap", fontsize=12)
        ax.legend()
        ax.grid()
        # fig.suptitle("Scaled primal gap", y=0.97, fontsize=13)
        # fig.tight_layout()
        plt.savefig('./result/plots/' + self.instance_type + '_' + self.instance_size + '_' + self.incumbent_mode + '.png')
        plt.show()
        plt.clf()

    def primal_integral_03(self, test_instance_size, total_time_limit=60, node_time_limit=30):

        direc = './data/generated_instances/' + self.instance_type + '/' + test_instance_size + '/'
        directory_transformedmodel = direc + 'transformedmodel' + '/'

        # set directory for the test result of RL-policy1
        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/reinforce/test/old_models/'
        directory_lb_test = directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/'

        # directory_rl_talored = directory_lb_test + 'rlactive/'
        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/' + 'rl/reinforce/test/old_models/'
            directory_lb_test_2 = directory_2 + 'evaluation-reinforce4lb-from-' +  'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/'
        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/' + 'rl/reinforce/test/old_models/'
            directory_lb_test_2 = directory_2 + 'evaluation-reinforce4lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/'

        # directory_3 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # directory_lb_test_3 = directory_3 + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
        #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'


        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
        #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/'
            # directory_lb_test_2 = directory_2 + 'lb-from-' + 'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(
            #     total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'


        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/'
            # directory_lb_test_2 = directory_2 + 'lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(
            #     total_time_limit) + 's' + test_instance_size + '/'
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

        # primal_int_regression_reinforces_talored = []
        # primal_int_reinforces_talored = []
        # primal_gap_final_regression_reinforces_talored = []
        # primal_gap_final_reinforces_talored = []
        # steplines_regression_reinforce_talored = []
        # steplines_reinforce_talored = []

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

                # filename_3 = f'{directory_lb_test_3}lb-test-{instance_name}.pkl'
                #
                # with gzip.open(filename_3, 'rb') as f:
                #     data = pickle.load(f)
                # objs, times, objs_pred_2, times_pred_2, objs_pred_reset_2, times_pred_reset_2 = data  # objs contains objs of a single instance of a lb test
                #
                # objs_regression = objs_pred_reset_2
                # times_regression = times_pred_reset_2

                # # test from k_prime
                # filename = f'{directory_lb_test_k_prime}lb-test-{instance_name}.pkl'
                # with gzip.open(filename, 'rb') as f:
                #     data = pickle.load(f)
                # objs_k_prime, times_k_prime = data  # objs contains objs of a single instance of a lb test

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

                # filename = f'{directory_lb_test_k_prime_2}lb-test-{instance_name}.pkl'
                # with gzip.open(filename, 'rb') as f:
                #     data = pickle.load(f)
                # objs_k_prime_2, times_k_prime_2 = data  # objs contains objs of a single instance of a lb test

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

                # objs_k_prime = np.array(objs_k_prime).reshape(-1)
                # times_k_prime = np.array(times_k_prime).reshape(-1)
                #
                # objs_k_prime_2 = np.array(objs_k_prime_2).reshape(-1)

                objs_k_prime_merged = np.array(objs_k_prime_merged).reshape(-1)
                times_k_prime_merged = np.array(times_k_prime_merged).reshape(-1)

                objs_k_prime_merged_2 = np.array(objs_k_prime_merged_2).reshape(-1)

                # a = [objs_regression.min(), objs_regresison_reinforce.min(), objs_reset_vanilla_2.min(), objs_reset_imitation_2.min()]
                a = [objs_reinforce.min(), objs_regresison_reinforce.min(), objs_reinforce_2.min(), objs_regresison_reinforce_2.min(), objs.min(), objs_2.min(), objs_k_prime_merged.min(), objs_k_prime_merged_2.min()]
                obj_opt = np.amin(a)

                # lb-baseline:
                # compute primal gap for baseline localbranching run
                # if times[-1] < total_time_limit:
                primal_int_baseline, primal_gap_final_baseline, stepline_baseline = self.compute_primal_integral(times=times, objs=objs, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_baselines.append(primal_gap_final_baseline)
                steplines_baseline.append(stepline_baseline)
                primal_int_baselines.append(primal_int_baseline)

                # # lb-regression
                # # if times_regression[-1] < total_time_limit:
                #
                # primal_int_regression, primal_gap_final_regression, stepline_regression = self.compute_primal_integral(
                #     times=times_k_prime, objs=objs_k_prime, obj_opt=obj_opt, total_time_limit=total_time_limit)
                # primal_gap_final_regressions.append(primal_gap_final_regression)
                # steplines_regression.append(stepline_regression)
                # primal_int_regressions.append(primal_int_regression)

                # lb-regression-merged
                # if times_regression[-1] < total_time_limit:

                primal_int_regression_merged, primal_gap_final_regression_merged, stepline_regression_merged = self.compute_primal_integral(
                    times=times_k_prime_merged, objs=objs_k_prime_merged, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_regressions_merged.append(primal_gap_final_regression_merged)
                steplines_regression_merged.append(stepline_regression_merged)
                primal_int_regressions_merged.append(primal_int_regression_merged)

                #
                # t = np.linspace(start=0.0, stop=total_time_limit, num=1001)
                # plt.close('all')
                # plt.clf()
                # fig, ax = plt.subplots(figsize=(8, 6.4))
                # fig.suptitle("Test Result: comparison of primal gap")
                # fig.subplots_adjust(top=0.5)
                # # ax.set_title(instance_name, loc='right')
                # ax.plot(t, stepline_baseline(t), label='lb baseline')
                # ax.plot(t, stepline_reset_vanilla(t), label='lb with k predicted')
                # ax.set_xlabel('time /s')
                # ax.set_ylabel("objective")
                # ax.legend()
                # plt.show()

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

                # # lb-regression-reinforce-talored
                # primal_int_regression_reinforce_talored, primal_gap_final_regression_reinforce_talored, stepline_regression_reinforce_talored = self.compute_primal_integral(
                #     times=times_regression_reinforce_talored, objs=objs_regresison_reinforce_talored, obj_opt=obj_opt,
                #     total_time_limit=total_time_limit)
                # primal_gap_final_regression_reinforces_talored.append(primal_gap_final_regression_reinforce_talored)
                # steplines_regression_reinforce_talored.append(stepline_regression_reinforce_talored)
                # primal_int_regression_reinforces_talored.append(primal_int_regression_reinforce_talored)
                #
                # # lb-reinforce
                #
                # primal_int_reinforce_talored, primal_gap_final_reinforce_talored, stepline_reinforce_talored = self.compute_primal_integral(
                #     times=times_reinforce_talored, objs=objs_reinforce_talored, obj_opt=obj_opt,
                #     total_time_limit=total_time_limit)
                # primal_gap_final_reinforces_talored.append(primal_gap_final_reinforce_talored)
                # steplines_reinforce_talored.append(stepline_reinforce_talored)
                # primal_int_reinforces_talored.append(primal_int_reinforce_talored)

                # plt.close('all')
                # plt.clf()
                # fig, ax = plt.subplots(figsize=(8, 6.4))
                # fig.suptitle("Test Result: comparison of objective")
                # fig.subplots_adjust(top=0.5)
                # ax.set_title(instance_name, loc='right')
                # ax.plot(times, objs, label='lb baseline')
                # ax.plot(times_regression, objs_regression, label='lb with k predicted')
                # ax.set_xlabel('time /s')
                # ax.set_ylabel("objective")
                # ax.legend()
                # plt.show()
                #
                # plt.close('all')
                # plt.clf()
                # fig, ax = plt.subplots(figsize=(8, 6.4))
                # fig.suptitle("Test Result: comparison of primal gap")
                # fig.subplots_adjust(top=0.5)
                # ax.set_title(instance_name, loc='right')
                # ax.plot(times, gamma_baseline, label='lb baseline')
                # ax.plot(times_regression, gamma_reset_vanilla, label='lb with k predicted')
                # ax.set_xlabel('time /s')
                # ax.set_ylabel("objective")
                # ax.legend()
                # plt.show()


        primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_regressions = np.array(primal_int_regressions).reshape(-1)
        primal_int_regressions_merged = np.array(primal_int_regressions_merged).reshape(-1)
        primal_int_regression_reinforces = np.array(primal_int_regression_reinforces).reshape(-1)
        primal_int_reinforces = np.array(primal_int_reinforces).reshape(-1)

        # primal_int_regression_reinforces_talored = np.array(primal_int_regression_reinforces_talored).reshape(-1)
        # primal_int_reinforces_talored = np.array(primal_int_reinforces_talored).reshape(-1)

        primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_regressions = np.array(primal_gap_final_regressions).reshape(-1)
        primal_gap_final_regressions_merged = np.array(primal_gap_final_regressions_merged).reshape(-1)
        primal_gap_final_regression_reinforces = np.array(primal_gap_final_regression_reinforces).reshape(-1)
        primal_gap_final_reinforces = np.array(primal_gap_final_reinforces).reshape(-1)

        # primal_gap_final_regression_reinforces_talored = np.array(primal_gap_final_regression_reinforces_talored).reshape(-1)
        # primal_gap_final_reinforces_talored = np.array(primal_gap_final_reinforces_talored).reshape(-1)

        # avarage primal integral over test dataset
        primal_int_base_ave = primal_int_baselines.sum() / len(primal_int_baselines)
        primal_int_regression_ave = primal_int_regressions.sum() / len(primal_int_regressions)
        primal_int_regression_merged_ave = primal_int_regressions_merged.sum() / len(primal_int_regressions_merged)
        primal_int_regression_reinforce_ave = primal_int_regression_reinforces.sum() / len(primal_int_regression_reinforces)
        primal_int_reinforce_ave = primal_int_reinforces.sum() / len(
            primal_int_reinforces)

        # primal_int_regression_reinforce_talored_ave = primal_int_regression_reinforces_talored.sum() / len(primal_int_regression_reinforces_talored)
        # primal_int_reinforce_talored_ave = primal_int_reinforces_talored.sum() / len(
        #     primal_int_reinforces_talored)

        primal_gap_final_baseline_ave = primal_gap_final_baselines.sum() / len(primal_gap_final_baselines)
        primal_gap_final_regression_ave = primal_gap_final_regressions.sum() / len(primal_gap_final_regressions)
        primal_gap_final_regression_merged_ave = primal_gap_final_regressions_merged.sum() / len(primal_gap_final_regressions_merged)
        primal_gap_final_regression_reinforce_ave = primal_gap_final_regression_reinforces.sum() / len(primal_gap_final_regression_reinforces)
        primal_gap_final_reinforce_ave = primal_gap_final_reinforces.sum() / len(
            primal_gap_final_reinforces)

        # primal_gap_final_regression_reinforce_talored_ave = primal_gap_final_regression_reinforces_talored.sum() / len(
        #     primal_gap_final_regression_reinforces_talored)
        # primal_gap_final_reinforce_talored_ave = primal_gap_final_reinforces_talored.sum() / len(
        #     primal_gap_final_reinforces_talored)

        print(self.instance_type + test_instance_size)
        print(self.incumbent_mode + 'Solution')
        print('baseline primal integral: ', primal_int_base_ave)
        print('regression primal integral: ', primal_int_regression_ave)
        print('regression merged primal integral: ', primal_int_regression_merged_ave)
        print('rl primal integral: ', primal_int_reinforce_ave)
        print('regression-rl primal integral: ', primal_int_regression_reinforce_ave)

        print('\n')
        print('baseline primal gap: ', primal_gap_final_baseline_ave)
        print('regression primal gap: ', primal_gap_final_regression_ave)
        print('regression primal merged gap: ', primal_gap_final_regression_merged_ave)
        print('rl primal gap: ', primal_gap_final_reinforce_ave)
        print('regression-rl primal gap: ', primal_gap_final_regression_reinforce_ave)

        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)

        primalgaps_baseline = None
        for n, stepline_baseline in enumerate(steplines_baseline):
            primal_gap = stepline_baseline(t)
            if n==0:
                primalgaps_baseline = primal_gap
            else:
                primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        primalgap_baseline_ave = np.average(primalgaps_baseline, axis=0)

        # primalgaps_regression = None
        # for n, stepline_regression in enumerate(steplines_regression):
        #     primal_gap = stepline_regression(t)
        #     if n == 0:
        #         primalgaps_regression = primal_gap
        #     else:
        #         primalgaps_regression = np.vstack((primalgaps_regression, primal_gap))
        # primalgap_regression_ave = np.average(primalgaps_regression, axis=0)

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

        # primalgaps_regression_reinforce_talored = None
        # for n, stepline_regression_reinforce in enumerate(steplines_regression_reinforce_talored):
        #     primal_gap = stepline_regression_reinforce(t)
        #     if n == 0:
        #         primalgaps_regression_reinforce_talored = primal_gap
        #     else:
        #         primalgaps_regression_reinforce_talored = np.vstack((primalgaps_regression_reinforce_talored, primal_gap))
        # primalgap_regression_reinforce_talored_ave = np.average(primalgaps_regression_reinforce_talored, axis=0)
        #
        # primalgaps_reinforce_talored = None
        # for n, stepline_reinforce in enumerate(steplines_reinforce_talored):
        #     primal_gap = stepline_reinforce(t)
        #     if n == 0:
        #         primalgaps_reinforce_talored = primal_gap
        #     else:
        #         primalgaps_reinforce_talored = np.vstack((primalgaps_reinforce_talored, primal_gap))
        # primalgap_reinforce_talored_ave = np.average(primalgaps_reinforce_talored, axis=0)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle(self.instance_type + '-' + self.incumbent_mode, fontsize=13)
        # ax.set_title(self.insancte_type + test_instance_size + '-' + self.incumbent_mode, fontsize=14)
        ax.plot(t, primalgap_baseline_ave, label='lb-base', color='tab:blue')
        # ax.plot(t, primalgap_regression_ave, label='lb-sr', color ='tab:orange')
        ax.plot(t, primalgap_regression_merged_ave, label='lb-srm', color='tab:orange')
        ax.plot(t, primalgap_reinforce_ave, '--', label='lb-rl', color='tab:green')
        ax.plot(t, primalgap_regression_reinforce_ave,'--', label='lb-srmrl', color='tab:red')
        #
        # ax.plot(t, primalgap_reinforce_talored_ave, ':', label='lb-rl-active', color='tab:green')
        # ax.plot(t, primalgap_regression_reinforce_talored_ave, ':', label='lb-regression-rl-active', color='tab:red')

        ax.set_xlabel('time /s', fontsize=12)
        ax.set_ylabel("scaled primal gap", fontsize=12)
        ax.legend()
        ax.grid()
        # fig.suptitle("Scaled primal gap", y=0.97, fontsize=13)
        # fig.tight_layout()
        plt.savefig('./result/plots/' + self.instance_type + '_' + self.incumbent_mode + '.png')
        plt.show()
        plt.clf()

    def primal_integral_hybrid_03(self, test_instance_size, total_time_limit=60, node_time_limit=30):

        direc = './data/generated_instances/' + self.instance_type + '/' + test_instance_size + '/'
        directory_transformedmodel = direc + 'transformedmodel' + '/'

        # set directory for the test result of RL-policy1
        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/reinforce/test/old_models/'
        directory_lb_test = directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/'

        # directory_rl_talored = directory_lb_test + 'rlactive/'
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

        # directory_rl_talored = directory_lb_test + 'rlactive/'
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

        # directory_3 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # directory_lb_test_3 = directory_3 + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
        #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'


        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
        #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/'
            # directory_lb_test_2 = directory_2 + 'lb-from-' + 'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(
            #     total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
            directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/'
            directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
                node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/'


        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/'
            # directory_lb_test_2 = directory_2 + 'lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(
            #     total_time_limit) + 's' + test_instance_size + '/'
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

                # filename = f'{directory_lb_test_hybrid_2}lb-test-{instance_name}.pkl'
                #
                # with gzip.open(filename, 'rb') as f:
                #     data = pickle.load(f)
                # objs_reinforce_hybrid_2, times_reinforce_hybrid_2, objs_regresison_reinforce_hybrid_2, times_regression_reinforce_hybrid_2 = data  # objs contains objs of a single instance of a lb test



                # filename_3 = f'{directory_lb_test_3}lb-test-{instance_name}.pkl'
                #
                # with gzip.open(filename_3, 'rb') as f:
                #     data = pickle.load(f)
                # objs, times, objs_pred_2, times_pred_2, objs_pred_reset_2, times_pred_reset_2 = data  # objs contains objs of a single instance of a lb test
                #
                # objs_regression = objs_pred_reset_2
                # times_regression = times_pred_reset_2

                # # test from k_prime
                # filename = f'{directory_lb_test_k_prime}lb-test-{instance_name}.pkl'
                # with gzip.open(filename, 'rb') as f:
                #     data = pickle.load(f)
                # objs_k_prime, times_k_prime = data  # objs contains objs of a single instance of a lb test

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

                # filename = f'{directory_lb_test_k_prime_2}lb-test-{instance_name}.pkl'
                # with gzip.open(filename, 'rb') as f:
                #     data = pickle.load(f)
                # objs_k_prime_2, times_k_prime_2 = data  # objs contains objs of a single instance of a lb test

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

                # objs_reinforce_hybrid_2 = np.array(objs_reinforce_hybrid_2).reshape(-1)
                # objs_regresison_reinforce_hybird_2 = np.array(objs_regresison_reinforce_hybrid_2).reshape(-1)

                objs = np.array(objs).reshape(-1)
                times = np.array(times).reshape(-1)

                objs_2 = np.array(objs_2).reshape(-1)

                # objs_k_prime = np.array(objs_k_prime).reshape(-1)
                # times_k_prime = np.array(times_k_prime).reshape(-1)
                #
                # objs_k_prime_2 = np.array(objs_k_prime_2).reshape(-1)

                objs_k_prime_merged = np.array(objs_k_prime_merged).reshape(-1)
                times_k_prime_merged = np.array(times_k_prime_merged).reshape(-1)

                objs_k_prime_merged_2 = np.array(objs_k_prime_merged_2).reshape(-1)

                # a = [objs_regression.min(), objs_regresison_reinforce.min(), objs_reset_vanilla_2.min(), objs_reset_imitation_2.min()]
                a = [objs_reinforce.min(), objs_regresison_reinforce.min(), objs_reinforce_2.min(), objs_regresison_reinforce_2.min(), objs.min(), objs_2.min(), objs_k_prime_merged.min(), objs_k_prime_merged_2.min(), objs_reinforce_hybrid.min(), objs_regresison_reinforce_hybrid.min()] # , objs_reinforce_hybrid_2.min(), objs_regresison_reinforce_hybrid_2.min(),
                obj_opt = np.amin(a)

                # lb-baseline:
                # compute primal gap for baseline localbranching run
                # if times[-1] < total_time_limit:
                primal_int_baseline, primal_gap_final_baseline, stepline_baseline = self.compute_primal_integral(times=times, objs=objs, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_baselines.append(primal_gap_final_baseline)
                steplines_baseline.append(stepline_baseline)
                primal_int_baselines.append(primal_int_baseline)

                # # lb-regression
                # # if times_regression[-1] < total_time_limit:
                #
                # primal_int_regression, primal_gap_final_regression, stepline_regression = self.compute_primal_integral(
                #     times=times_k_prime, objs=objs_k_prime, obj_opt=obj_opt, total_time_limit=total_time_limit)
                # primal_gap_final_regressions.append(primal_gap_final_regression)
                # steplines_regression.append(stepline_regression)
                # primal_int_regressions.append(primal_int_regression)

                # lb-regression-merged
                # if times_regression[-1] < total_time_limit:

                primal_int_regression_merged, primal_gap_final_regression_merged, stepline_regression_merged = self.compute_primal_integral(
                    times=times_k_prime_merged, objs=objs_k_prime_merged, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_regressions_merged.append(primal_gap_final_regression_merged)
                steplines_regression_merged.append(stepline_regression_merged)
                primal_int_regressions_merged.append(primal_int_regression_merged)

                #
                # t = np.linspace(start=0.0, stop=total_time_limit, num=1001)
                # plt.close('all')
                # plt.clf()
                # fig, ax = plt.subplots(figsize=(8, 6.4))
                # fig.suptitle("Test Result: comparison of primal gap")
                # fig.subplots_adjust(top=0.5)
                # # ax.set_title(instance_name, loc='right')
                # ax.plot(t, stepline_baseline(t), label='lb baseline')
                # ax.plot(t, stepline_reset_vanilla(t), label='lb with k predicted')
                # ax.set_xlabel('time /s')
                # ax.set_ylabel("objective")
                # ax.legend()
                # plt.show()

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

                # plt.close('all')
                # plt.clf()
                # fig, ax = plt.subplots(figsize=(8, 6.4))
                # fig.suptitle("Test Result: comparison of objective")
                # fig.subplots_adjust(top=0.5)
                # ax.set_title(instance_name, loc='right')
                # ax.plot(times, objs, label='lb baseline')
                # ax.plot(times_regression, objs_regression, label='lb with k predicted')
                # ax.set_xlabel('time /s')
                # ax.set_ylabel("objective")
                # ax.legend()
                # plt.show()
                #
                # plt.close('all')
                # plt.clf()
                # fig, ax = plt.subplots(figsize=(8, 6.4))
                # fig.suptitle("Test Result: comparison of primal gap")
                # fig.subplots_adjust(top=0.5)
                # ax.set_title(instance_name, loc='right')
                # ax.plot(times, gamma_baseline, label='lb baseline')
                # ax.plot(times_regression, gamma_reset_vanilla, label='lb with k predicted')
                # ax.set_xlabel('time /s')
                # ax.set_ylabel("objective")
                # ax.legend()
                # plt.show()


        primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_regressions = np.array(primal_int_regressions).reshape(-1)
        primal_int_regressions_merged = np.array(primal_int_regressions_merged).reshape(-1)
        primal_int_regression_reinforces = np.array(primal_int_regression_reinforces).reshape(-1)
        primal_int_reinforces = np.array(primal_int_reinforces).reshape(-1)

        # primal_int_regression_reinforces_talored = np.array(primal_int_regression_reinforces_talored).reshape(-1)
        # primal_int_reinforces_talored = np.array(primal_int_reinforces_talored).reshape(-1)

        primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_regressions = np.array(primal_gap_final_regressions).reshape(-1)
        primal_gap_final_regressions_merged = np.array(primal_gap_final_regressions_merged).reshape(-1)
        primal_gap_final_regression_reinforces = np.array(primal_gap_final_regression_reinforces).reshape(-1)
        primal_gap_final_reinforces = np.array(primal_gap_final_reinforces).reshape(-1)

        # primal_gap_final_regression_reinforces_talored = np.array(primal_gap_final_regression_reinforces_talored).reshape(-1)
        # primal_gap_final_reinforces_talored = np.array(primal_gap_final_reinforces_talored).reshape(-1)

        # avarage primal integral over test dataset
        primal_int_base_ave = primal_int_baselines.sum() / len(primal_int_baselines)
        primal_int_regression_ave = primal_int_regressions.sum() / len(primal_int_regressions)
        primal_int_regression_merged_ave = primal_int_regressions_merged.sum() / len(primal_int_regressions_merged)
        primal_int_regression_reinforce_ave = primal_int_regression_reinforces.sum() / len(primal_int_regression_reinforces)
        primal_int_reinforce_ave = primal_int_reinforces.sum() / len(
            primal_int_reinforces)

        # primal_int_regression_reinforce_talored_ave = primal_int_regression_reinforces_talored.sum() / len(primal_int_regression_reinforces_talored)
        # primal_int_reinforce_talored_ave = primal_int_reinforces_talored.sum() / len(
        #     primal_int_reinforces_talored)

        primal_gap_final_baseline_ave = primal_gap_final_baselines.sum() / len(primal_gap_final_baselines)
        primal_gap_final_regression_ave = primal_gap_final_regressions.sum() / len(primal_gap_final_regressions)
        primal_gap_final_regression_merged_ave = primal_gap_final_regressions_merged.sum() / len(primal_gap_final_regressions_merged)
        primal_gap_final_regression_reinforce_ave = primal_gap_final_regression_reinforces.sum() / len(primal_gap_final_regression_reinforces)
        primal_gap_final_reinforce_ave = primal_gap_final_reinforces.sum() / len(
            primal_gap_final_reinforces)

        # primal_gap_final_regression_reinforce_talored_ave = primal_gap_final_regression_reinforces_talored.sum() / len(
        #     primal_gap_final_regression_reinforces_talored)
        # primal_gap_final_reinforce_talored_ave = primal_gap_final_reinforces_talored.sum() / len(
        #     primal_gap_final_reinforces_talored)

        print(self.instance_type + test_instance_size)
        print(self.incumbent_mode + 'Solution')
        print('baseline primal integral: ', primal_int_base_ave)
        print('regression primal integral: ', primal_int_regression_ave)
        print('regression merged primal integral: ', primal_int_regression_merged_ave)
        print('rl primal integral: ', primal_int_reinforce_ave)
        print('regression-rl primal integral: ', primal_int_regression_reinforce_ave)

        print('\n')
        print('baseline primal gap: ', primal_gap_final_baseline_ave)
        print('regression primal gap: ', primal_gap_final_regression_ave)
        print('regression primal merged gap: ', primal_gap_final_regression_merged_ave)
        print('rl primal gap: ', primal_gap_final_reinforce_ave)
        print('regression-rl primal gap: ', primal_gap_final_regression_reinforce_ave)

        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)

        primalgaps_baseline = None
        for n, stepline_baseline in enumerate(steplines_baseline):
            primal_gap = stepline_baseline(t)
            if n==0:
                primalgaps_baseline = primal_gap
            else:
                primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        primalgap_baseline_ave = np.average(primalgaps_baseline, axis=0)

        # primalgaps_regression = None
        # for n, stepline_regression in enumerate(steplines_regression):
        #     primal_gap = stepline_regression(t)
        #     if n == 0:
        #         primalgaps_regression = primal_gap
        #     else:
        #         primalgaps_regression = np.vstack((primalgaps_regression, primal_gap))
        # primalgap_regression_ave = np.average(primalgaps_regression, axis=0)

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
        # ax.set_title(self.insancte_type + test_instance_size + '-' + self.incumbent_mode, fontsize=14)
        ax.plot(t, primalgap_baseline_ave, label='lb-base', color='tab:blue')
        # ax.plot(t, primalgap_regression_ave, label='lb-sr', color ='tab:orange')
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
        # fig.suptitle("Scaled primal gap", y=0.97, fontsize=13)
        # fig.tight_layout()
        plt.savefig('./result/plots/' + self.instance_type + '_' + self.incumbent_mode + '.png')
        plt.show()
        plt.clf()














