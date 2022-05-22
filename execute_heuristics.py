import pyscipopt
from pyscipopt import Model, SCIP_HEURTIMING
import ecole
import numpy as np
import random
import pathlib
import gzip
import pickle
import json
import matplotlib.pyplot as plt
from geco.mips.loading.miplib import Loader
from utilities import lbconstraint_modes, instancetypes, incumbent_modes, instancesizes, generator_switcher, binary_support, copy_sol, mean_filter,mean_forward_filter, imitation_accuracy, haming_distance_solutions, haming_distance_solutions_asym, getBestFeasiSol

import torch.nn.functional as F
import torch_geometric
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from scipy.interpolate import interp1d

import gc
import sys
from memory_profiler import profile

from dataset import InstanceDataset, custom_collate, InstanceDataset_2
from event import PrimalBoundChangeEventHandler
from primal_heur_localbranch import HeurLocalbranch
from ecole_extend.environment_extend import SimpleConfiguring, SimpleConfiguringEnablecuts, SimpleConfiguringEnableheuristics
from models import GraphDataset, GNNPolicy, BipartiteNodeData

class ExecuteHeuristic:
    """
    Basic class for the execution of a MIP heuristic on a specific instance set. This basic class uses SCIP solver as the underlying heuristic method
    """

    def __init__(self, instance_directory, solution_directory, result_directory, no_improve_iteration_limit=20, seed=100, enable_gpu=False):

        self.instance_directory = instance_directory
        self.solution_directory = solution_directory
        self.result_directory = result_directory
        self.no_improve_iteration_limit = no_improve_iteration_limit

        self.seed = seed
        print('seed: {}'.format(str(seed)))
        # self.directory = './result/generated_instances/' + self.instance_type + '/' + self.instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # self.generator = generator_switcher(self.instance_type + self.instance_size)

        # self.initialize_ecole_env()
        # self.env.seed(self.seed)  # environment (SCIP)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # random.seed(seed)
        self.enable_gpu = enable_gpu
        if self.enable_gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        print(self.device)

    # def compute_k_prime(self, MIP_model, incumbent):
    #
    #     # solve the root node and get the LP solution
    #     MIP_model.freeTransform()
    #     status = MIP_model.getStatus()
    #     print("* Model status: %s" % status)
    #     MIP_model.resetParams()
    #     MIP_model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
    #     MIP_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    #     MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
    #     MIP_model.setIntParam("lp/solvefreq", 0)
    #     MIP_model.setParam("limits/nodes", 1)
    #     # MIP_model.setParam("limits/solutions", 1)
    #     MIP_model.setParam("display/verblevel", 0)
    #     MIP_model.setParam("lp/disablecutoff", 1)
    #
    #     # MIP_model.setParam("limits/solutions", 1)
    #     MIP_model.optimize()
    #     #
    #     status = MIP_model.getStatus()
    #     lp_status = MIP_model.getLPSolstat()
    #     stage = MIP_model.getStage()
    #     n_sols = MIP_model.getNSols()
    #     # root_time = MIP_model.getSolvingTime()
    #     print("* Model status: %s" % status)
    #     print("* Solve stage: %s" % stage)
    #     print("* LP status: %s" % lp_status)
    #     print('* number of sol : ', n_sols)
    #
    #     sol_lp = MIP_model.createLPSol()
    #     # sol_relax = MIP_model.createRelaxSol()
    #
    #     k_prime = haming_distance_solutions(MIP_model, incumbent, sol_lp)
    #     if not self.is_symmetric:
    #         k_prime = haming_distance_solutions_asym(MIP_model, incumbent, sol_lp)
    #     k_prime = np.ceil(k_prime)
    #
    #     return k_prime

    # def load_mip_dataset(self, instance_directory=None, sols_directory=None, incumbent_mode=None):
    #     instance_filename = f'{self.instance_type}-*_transformed.cip'
    #     sol_filename = f'{incumbent_mode}-{self.instance_type}-*_transformed.sol'
    #
    #     train_instances_directory = instance_directory + 'train/'
    #     instance_files = [str(path) for path in sorted(pathlib.Path(train_instances_directory).glob(instance_filename), key=lambda path: int(path.stem.replace('-', '_').rsplit("_", 2)[1]))]
    #
    #     instance_train_files = instance_files[:int(7/8 * len(instance_files))]
    #     instance_valid_files = instance_files[int(7/8 * len(instance_files)):]
    #
    #     test_instances_directory = instance_directory + 'test/'
    #     instance_test_files = [str(path) for path in sorted(pathlib.Path(test_instances_directory).glob(instance_filename),
    #                                                    key=lambda path: int(
    #                                                        path.stem.replace('-', '_').rsplit("_", 2)[1]))]
    #
    #     train_sols_directory = sols_directory + 'train/'
    #     sol_files = [str(path) for path in sorted(pathlib.Path(train_sols_directory).glob(sol_filename), key=lambda path: int(path.stem.replace('-', '_').rsplit("_", 2)[1]))]
    #
    #     sol_train_files = sol_files[:int(7/8 * len(sol_files))]
    #     sol_valid_files = sol_files[int(7/8 * len(sol_files)):]
    #
    #     test_sols_directory = sols_directory + 'test/'
    #     sol_test_files = [str(path) for path in sorted(pathlib.Path(test_sols_directory).glob(sol_filename),
    #                                               key=lambda path: int(path.stem.replace('-', '_').rsplit("_", 2)[1]))]
    #
    #     train_dataset = InstanceDataset(mip_files=instance_train_files, sol_files=sol_train_files)
    #     valid_dataset = InstanceDataset(mip_files=instance_valid_files, sol_files=sol_valid_files)
    #     test_dataset = InstanceDataset(mip_files=instance_test_files, sol_files=sol_test_files)
    #
    #     return train_dataset, valid_dataset, test_dataset

    def load_test_mip_dataset(self, instance_directory=None, sols_directory=None):
        instance_filename = f'*_transformed.cip'
        sol_filename = f'*_transformed.sol'

        test_instances_directory = instance_directory
        instance_test_files = [str(path) for path in sorted(pathlib.Path(test_instances_directory).glob(instance_filename),
                                                       key=lambda path: int(
                                                           path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        test_sols_directory = sols_directory
        sol_test_files = [str(path) for path in sorted(pathlib.Path(test_sols_directory).glob(sol_filename),
                                                  key=lambda path: int(path.stem.replace('-', '_').rsplit("_", 2)[1]))]

        test_dataset = InstanceDataset_2(mip_files=instance_test_files, sol_files=sol_test_files)

        return test_dataset

    def load_results_file_list(self, instance_directory=None):
        instance_filename = f'{self.instance_type}-*_transformed.cip'
        # sol_filename = f'{incumbent_mode}-{self.instance_type}-*_transformed.sol'

        test_instances_directory = instance_directory
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

    def execute_heuristic_per_instance(self, MIP_model, incumbent, node_time_limit, total_time_limit):
        """
        call the underlying heuristic method over an MIP instance, this is the basic method by directly running scip to solve the problem
        :param MIP_model:
        :param incumbent:
        :param node_time_limit:
        :param total_time_limit:
        :return:
        """

        objs = []
        times = []
        MIP_obj_best = MIP_model.getSolObjVal(incumbent)
        times.append(0.0)
        objs.append(MIP_obj_best)

        primalbound_handler = PrimalBoundChangeEventHandler()
        primalbound_handler.primal_times = []
        primalbound_handler.primal_bounds = []

        MIP_model.includeEventhdlr(primalbound_handler, 'primal_bound_update_handler',
                                   'store every new primal bound and its time stamp')

        MIP_model.setParam('limits/time', total_time_limit)
        MIP_model.setParam("display/verblevel", 0)
        MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.FAST)
        MIP_model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
        MIP_model.optimize()
        status = MIP_model.getStatus()
        n_sols_MIP = MIP_model.getNSols()
        # MIP_model.freeTransform()
        # feasible = MIP_model.checkSol(solution=MIP_model.getBestSol())
        elapsed_time = MIP_model.getSolvingTime()


        if n_sols_MIP > 0:
            feasible, MIP_sol_incumbent, MIP_obj_incumbent = getBestFeasiSol(MIP_model)
            feasible = MIP_model.checkSol(solution=MIP_sol_incumbent)
            assert feasible, "Error: the best solution from current SCIP solving is not feasible!"
            # MIP_obj_incumbent = MIP_model.getSolObjVal(MIP_sol_incumbent)

            if MIP_obj_incumbent < MIP_obj_best:
                primal_bounds = primalbound_handler.primal_bounds
                primal_times = primalbound_handler.primal_times
                MIP_obj_best = MIP_obj_incumbent

                # for i in range(len(primal_times)):
                #     primal_times[i] += self.total_time_expired

                objs.extend(primal_bounds)
                times.extend(primal_times)

        obj_best = MIP_obj_best
        objs = np.array(objs).reshape(-1)
        times = np.array(times).reshape(-1)

        print("Instance:", MIP_model.getProbName())
        print("Status of SCIP_baseline: ", status)
        print("Best obj of SCIP_baseline: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        data = [objs, times]

        return data

    def setup_heuristic_per_instance(self, MIP_model, incumbent, node_time_limit, total_time_limit):
        """
        wrapper method of seting up and evaluating the heuristic algorithm on a given MIP instance:
        :param node_time_limit:
        :param total_time_limit:
        :return:
        """
        device = self.device
        gc.collect()

        # if index_instance == 18:
        #     index_instance = 19

        instance_name = MIP_model.getProbName()
        print(instance_name)
        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

        feas = MIP_model.checkSol(incumbent)
        try:
            MIP_model.addSol(incumbent, False)
        except:
            print('Error: the root solution of ' + instance_name + ' is not feasible!')

        MIP_model.resetParams()

        initial_obj = MIP_model.getSolObjVal(incumbent)
        print("Initial obj of MIP: {}".format(initial_obj))

        # MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
        #     problemName='Baseline', origcopy=False)
        #
        # print('MIP copies are created')
        #
        # MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent,
        #                                         MIP_copy_vars)
        #
        # feasible =MIP_model_copy.checkSol(solution=sol_MIP_copy)
        # if feasible:
        #     print('The initial solution before SCIP running is feasible')
        # print('incumbent solution is copied to MIP copies')
        #
        # initial_obj = MIP_model.getSolObjVal(incumbent)
        # print("Initial obj of original MIP before LB: {}".format(initial_obj))
        #
        # MIP_model.freeProb()
        # del incumbent
        # del MIP_model
        # sol = MIP_model_copy.getBestSol()
        # initial_obj = MIP_model_copy.getSolObjVal(sol)
        # print("Initial obj of copied MIP: {}".format(initial_obj))

        # call and execute heuristic search for the given instance
        data = self.execute_heuristic_per_instance(MIP_model, incumbent, node_time_limit,
                                                          total_time_limit)
        print(data)

        # objs = np.array(objs).reshape(-1)
        # times = np.array(times).reshape(-1)

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
        #
        # objs = np.array(lb_model.primal_objs).reshape(-1)
        # times = np.array(lb_model.primal_times).reshape(-1)
        #
        # print("Instance:", MIP_model_copy.getProbName())
        # print("Status of LB: ", status)
        # print("Best obj of LB: ", obj_best)
        # print("Solving time: ", elapsed_time)
        # print('\n')

        # MIP_model_copy.freeProb()
        # del sol_MIP_copy
        # del MIP_model_copy

        filename = f'{self.result_directory}lb-test-{instance_name}.pkl'  # instance 100-199
        with gzip.open(filename, 'wb') as f:
            pickle.dump(data, f)
        MIP_model.freeProb()
        del MIP_model
        del data

    def execute_heuristic_baseline(self, total_time_limit=60, node_time_limit=10):
        """
        execution method for evaluating the underlying heuristic on a specified MIP instance set
        :param total_time_limit:
        :param node_time_limit:
        :return:
        """

        # self.regression_dataset = self.instance_type + '-small'
        # self.evaluation_dataset = self.instance_type + evaluation_instance_size

        instance_directory = self.instance_directory
        directory_sol = self.solution_directory

        test_dataset = self.load_test_mip_dataset(instance_directory, directory_sol)

        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn=custom_collate)

        i = 0
        for batch in (test_loader):
            if i >= 0:
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

                self.setup_heuristic_per_instance(MIP_model=MIP_model,
                                                  incumbent=incumbent_solution,
                                                  node_time_limit=node_time_limit,
                                                  total_time_limit=total_time_limit)
            i += 1

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
        primal_gap_final = np.abs(objs[-1] - obj_opt) / np.abs(obj_opt) * 100 # np.abs(obj_opt) * 100

        # create step line
        stepline = interp1d(times, gamma_baseline, 'previous')


        # compute primal integral
        primal_integral = 0
        for j in range(len(objs) - 1):
            primal_integral += gamma_baseline[j] * (times[j + 1] - times[j])

        return primal_integral, primal_gap_final, stepline# , gamma_baseline

        # gamma_baseline

    def primal_integral_03(self, seed_mcts=100, instance_type = 'miplib_39binary', instance_size = '-small', incumbent_mode = 'root', total_time_limit=60, node_time_limit=30, result_directory_1=None, result_directory_2=None, result_directory_3=None, result_directory_4=None, result_directory_5=None):

        instance_type = instance_type
        incumbent_mode = incumbent_mode
        instance_size = instance_size

        # direc = './data/generated_instances/' + self.instance_type + '/' + test_instance_size + '/'
        # directory_transformedmodel = direc + 'transformedmodel' + '/'
        #
        # # set directory for the test result of RL-policy1
        # directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/reinforce/test/old_models/'
        # directory_lb_test = directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(
        #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/seed'+ str(self.seed) + '/'
        #
        # # directory_rl_talored = directory_lb_test + 'rlactive/'
        # if self.incumbent_mode == 'firstsol':
        #     directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/' + 'rl/reinforce/test/old_models/'
        #     directory_lb_test_2 = directory_2 + 'evaluation-reinforce4lb-from-' +  'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/seed'+ str(self.seed) + '/'
        # elif self.incumbent_mode == 'rootsol':
        #     directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/' + 'rl/reinforce/test/old_models/'
        #     directory_lb_test_2 = directory_2 + 'evaluation-reinforce4lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/seed'+ str(self.seed) + '/'
        #
        # # directory_3 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # # directory_lb_test_3 = directory_3 + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
        # #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'


        # directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # # directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
        # #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        # if self.incumbent_mode == 'firstsol':
        #     directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/'
        #     # directory_lb_test_2 = directory_2 + 'lb-from-' + 'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(
        #     #     total_time_limit) + 's' + test_instance_size + '/'
        #     directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
        #         node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/seed'+ str(self.seed) + '/'
        #     directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
        #         node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/seed'+ str(self.seed) + '/'
        #     directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
        #         node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/seed'+ str(self.seed) + '/'
        #
        #
        # elif self.incumbent_mode == 'rootsol':
        #     directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/'
        #     # directory_lb_test_2 = directory_2 + 'lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(
        #     #     total_time_limit) + 's' + test_instance_size + '/'
        #     directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
        #         node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/seed'+ str(self.seed) + '/'
        #     directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
        #         node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/seed'+ str(self.seed) + '/'
        #     directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
        #         node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/seed'+ str(self.seed) + '/'

        # baseline algorithms
        directory_lns_random = result_directory_1
        directory_local_branch = result_directory_2
        directory_lns_lblp = result_directory_3
        directory_scip_baseline = result_directory_4
        directory_lns_lblpmcts = result_directory_5

        primal_int_baselines = []
        primal_int_lns_random_list = []
        primal_int_lns_lblp_list = []
        primal_int_scip_baselines_list = []
        primal_int_lns_lblpmcts_list = []
        primal_int_reinforces = []
        primal_gap_final_baselines = []
        primal_gap_final_lns_lblp_list = []
        primal_gap_final_lns_random_list = []
        primal_gap_final_scip_baselines_list = []
        primal_gap_final_lns_lblpmcts_list = []
        primal_gap_final_reinforces = []
        steplines_baseline = []
        steplines_lns_lblp_list = []
        steplines_lns_random_list = []
        steplines_scip_baseline_list = []
        steplines_lns_lblpmcts_list = []
        steplines_reinforce = []

        # primal_int_regression_reinforces_talored = []
        # primal_int_reinforces_talored = []
        # primal_gap_final_regression_reinforces_talored = []
        # primal_gap_final_reinforces_talored = []
        # steplines_regression_reinforce_talored = []
        # steplines_reinforce_talored = []

        if instance_type == instancetypes[3]:
            index_mix = 80
            index_max = 115
        elif instance_type == instancetypes[4]:
            index_mix = 0
            index_max = 30 # 30
        elif instance_type == instancetypes[2] and instance_size == '-large':
            index_mix = 0
            index_max = 40
        else:
            index_mix = 160
            index_max = 200


        for i in range(index_mix,index_max):

            if not (instance_type == instancetypes[4] and (i == 18)): # or i==4 # or i==5 or i==10 or i==21

                instance_name = instance_type + '-' + str(i) + '_transformed' # instance 100-199

                mip_filename = f'{self.instance_directory}{instance_name}.cip'
                mip = Model()
                MIP_model = Model()
                MIP_model.readProblem(mip_filename)
                instance_name = MIP_model.getProbName()

                # filename = f'{directory_lb_test}lb-test-{instance_name}.pkl'
                #
                # with gzip.open(filename, 'rb') as f:
                #     data = pickle.load(f)
                # objs_reinforce, times_reinforce, objs_regresison_reinforce, times_regression_reinforce = data  # objs contains objs of a single instance of a lb test
                #
                # filename = f'{directory_lb_test_2}lb-test-{instance_name}.pkl'
                #
                # with gzip.open(filename, 'rb') as f:
                #     data = pickle.load(f)
                # objs_reinforce_2, times_reinforce_2, objs_regresison_reinforce_2, times_regression_reinforce_2 = data  # objs contains objs of a single instance of a lb test
                #
                # # filename_3 = f'{directory_lb_test_3}lb-test-{instance_name}.pkl'
                # #
                # # with gzip.open(filename_3, 'rb') as f:
                # #     data = pickle.load(f)
                # # objs, times, objs_pred_2, times_pred_2, objs_pred_reset_2, times_pred_reset_2 = data  # objs contains objs of a single instance of a lb test
                # #
                # # objs_regression = objs_pred_reset_2
                # # times_regression = times_pred_reset_2
                #
                # test from lns-guided-by-localbranch-lp heuristic
                filename = f'{directory_lns_lblp}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_lns_lblp, times_lns_lblp = data  # objs contains objs of a single instance of a lb test

                # test from lns-random heuristic
                filename = f'{directory_lns_random}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_lns_random, times_lns_random = data  # objs contains objs of a single instance of a lb test

                # test from localbranch baseline
                filename = f'{directory_local_branch}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_lb, times_lb = data  # objs contains objs of a single instance of a lb test

                filename = f'{directory_scip_baseline}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_scip, times_scip = data  # objs contains objs of a single instance of a lb test

                # test from lns-guided-by-localbranch-lp-mcts heuristic
                filename = f'{directory_lns_lblpmcts}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_lns_lblpmcts, times_lns_lblpmcts = data  # objs contains objs of a single instance of a lb test

                # # filename = f'{directory_lb_test_k_prime_2}lb-test-{instance_name}.pkl'
                # # with gzip.open(filename, 'rb') as f:
                # #     data = pickle.load(f)
                # # objs_k_prime_2, times_k_prime_2 = data  # objs contains objs of a single instance of a lb test
                #
                # filename = f'{directory_lb_test_k_prime_merged_2}lb-test-{instance_name}.pkl'
                # with gzip.open(filename, 'rb') as f:
                #     data = pickle.load(f)
                # objs_k_prime_merged_2, times_k_prime_merged_2 = data  # objs contains objs of a single instance of a lb test
                #
                # filename = f'{directory_lb_test_baseline_2}lb-test-{instance_name}.pkl'
                # with gzip.open(filename, 'rb') as f:
                #     data = pickle.load(f)
                # objs_2, times_k_2 = data  # objs contains objs of a single instance of a lb test

                # objs_reinforce = np.array(objs_reinforce).reshape(-1)
                # times_reinforce = np.array(times_reinforce).reshape(-1)
                # objs_regresison_reinforce = np.array(objs_regresison_reinforce).reshape(-1)
                # times_regression_reinforce = np.array(times_regression_reinforce).reshape(-1)
                #
                # objs_reinforce_2 = np.array(objs_reinforce_2).reshape(-1)
                # objs_regresison_reinforce_2 = np.array(objs_regresison_reinforce_2).reshape(-1)

                # objs = np.array(objs).reshape(-1)
                # times = np.array(times).reshape(-1)

                # objs_2 = np.array(objs_2).reshape(-1)

                # objs_k_prime = np.array(objs_k_prime).reshape(-1)
                # times_lns_lblp = np.array(times_lns_lblp).reshape(-1)
                #
                # objs_k_prime_2 = np.array(objs_k_prime_2).reshape(-1)

                # objs_lns_random = np.array(objs_lns_random).reshape(-1)
                # times_lns_random = np.array(times_lns_random).reshape(-1)

                # objs_k_prime_merged_2 = np.array(objs_k_prime_merged_2).reshape(-1)

                # a = [objs_regression.min(), objs_regresison_reinforce.min(), objs_reset_vanilla_2.min(), objs_reset_imitation_2.min()]
                a = [objs_lb.min(), objs_lns_random.min(), objs_lns_lblp.min(), objs_scip.min(), objs_lns_lblpmcts.min()] #
                obj_opt = np.amin(a)

                # localbranch-baseline:
                # compute primal gap for baseline localbranching run
                # if times[-1] < total_time_limit:
                primal_int_baseline, primal_gap_final_baseline, stepline_baseline = self.compute_primal_integral(times=times_lb, objs=objs_lb, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_baselines.append(primal_gap_final_baseline)
                steplines_baseline.append(stepline_baseline)
                primal_int_baselines.append(primal_int_baseline)

                # lns-guided-by-localbranch-lp
                # if times_regression[-1] < total_time_limit:

                primal_int_lns_lblp, primal_gap_final_lns_lblp, stepline_lns_lblp = self.compute_primal_integral(
                    times=times_lns_lblp, objs=objs_lns_lblp, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_lns_lblp_list.append(primal_gap_final_lns_lblp)
                steplines_lns_lblp_list.append(stepline_lns_lblp)
                primal_int_lns_lblp_list.append(primal_int_lns_lblp)

                # lns-random heuristic
                # if times_regression[-1] < total_time_limit:

                primal_int_lns_random, primal_gap_final_lns_random, stepline_regression_lns_random = self.compute_primal_integral(
                    times=times_lns_random, objs=objs_lns_random, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_lns_random_list.append(primal_gap_final_lns_random)
                steplines_lns_random_list.append(stepline_regression_lns_random)
                primal_int_lns_random_list.append(primal_int_lns_random)

                # scip-baseline

                primal_int_scip, primal_gap_final_scip, stepline_scip = self.compute_primal_integral(
                    times=times_scip, objs=objs_scip, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_scip_baselines_list.append(primal_gap_final_scip)
                steplines_scip_baseline_list.append(stepline_scip)
                primal_int_scip_baselines_list.append(primal_int_scip)

                # lns-guided-by-localbranch-lp-mcts
                primal_int_lns_lblpmcts, primal_gap_final_lns_lblpmcts, stepline_lns_lblpmcts = self.compute_primal_integral(
                    times=times_lns_lblpmcts, objs=objs_lns_lblpmcts, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_lns_lblpmcts_list.append(primal_gap_final_lns_lblpmcts)
                steplines_lns_lblpmcts_list.append(stepline_lns_lblpmcts)
                primal_int_lns_lblpmcts_list.append(primal_int_lns_lblpmcts)

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

                #
                # # lb-reinforce
                #
                # primal_int_reinforce, primal_gap_final_reinforce, stepline_reinforce = self.compute_primal_integral(
                #     times=times_reinforce, objs=objs_reinforce, obj_opt=obj_opt,
                #     total_time_limit=total_time_limit)
                # primal_gap_final_reinforces.append(primal_gap_final_reinforce)
                # steplines_reinforce.append(stepline_reinforce)
                # primal_int_reinforces.append(primal_int_reinforce)

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
        primal_int_lns_lblp_list = np.array(primal_int_lns_lblp_list).reshape(-1)
        primal_int_lns_random_list = np.array(primal_int_lns_random_list).reshape(-1)
        primal_int_scip_baselines_list = np.array(primal_int_scip_baselines_list).reshape(-1)
        primal_int_lns_lblpmcts_list = np.array(primal_int_lns_lblpmcts_list).reshape(-1)
        primal_int_reinforces = np.array(primal_int_reinforces).reshape(-1)

        # primal_int_regression_reinforces_talored = np.array(primal_int_regression_reinforces_talored).reshape(-1)
        # primal_int_reinforces_talored = np.array(primal_int_reinforces_talored).reshape(-1)

        primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_lns_lblp_list = np.array(primal_gap_final_lns_lblp_list).reshape(-1)
        primal_gap_final_lns_random_list = np.array(primal_gap_final_lns_random_list).reshape(-1)
        primal_gap_final_scip_baselines_list = np.array(primal_gap_final_scip_baselines_list).reshape(-1)
        primal_gap_final_lns_lblpmcts_list = np.array(primal_gap_final_lns_lblpmcts_list).reshape(-1)
        primal_gap_final_reinforces = np.array(primal_gap_final_reinforces).reshape(-1)

        # primal_gap_final_regression_reinforces_talored = np.array(primal_gap_final_regression_reinforces_talored).reshape(-1)
        # primal_gap_final_reinforces_talored = np.array(primal_gap_final_reinforces_talored).reshape(-1)

        # avarage primal integral over test dataset
        primal_int_base_ave = primal_int_baselines.sum() / len(primal_int_baselines)
        primal_int_lns_lblp_ave = primal_int_lns_lblp_list.sum() / len(primal_int_lns_lblp_list)
        primal_int_lns_random_ave = primal_int_lns_random_list.sum() / len(primal_int_lns_random_list)
        primal_int_scip_baseline_ave = primal_int_scip_baselines_list.sum() / len(primal_int_scip_baselines_list)
        primal_int_lns_lblpmcts_ave = primal_int_lns_lblpmcts_list.sum() / len(primal_int_lns_lblpmcts_list)
        primal_int_reinforce_ave = primal_int_reinforces.sum() / len(
            primal_int_reinforces)

        # primal_int_regression_reinforce_talored_ave = primal_int_regression_reinforces_talored.sum() / len(primal_int_regression_reinforces_talored)
        # primal_int_reinforce_talored_ave = primal_int_reinforces_talored.sum() / len(
        #     primal_int_reinforces_talored)

        primal_gap_final_baseline_ave = primal_gap_final_baselines.sum() / len(primal_gap_final_baselines)
        primal_gap_final_lns_lblp_ave = primal_gap_final_lns_lblp_list.sum() / len(primal_gap_final_lns_lblp_list)
        primal_gap_final_lns_random_ave = primal_gap_final_lns_random_list.sum() / len(primal_gap_final_lns_random_list)
        primal_gap_final_scip_baseline_ave = primal_gap_final_scip_baselines_list.sum() / len(primal_gap_final_scip_baselines_list)
        primal_gap_final_lns_lblpmcts_ave = primal_gap_final_lns_lblpmcts_list.sum() / len(primal_gap_final_lns_lblpmcts_list)
        primal_gap_final_reinforce_ave = primal_gap_final_reinforces.sum() / len(
            primal_gap_final_reinforces)

        # primal_gap_final_regression_reinforce_talored_ave = primal_gap_final_regression_reinforces_talored.sum() / len(
        #     primal_gap_final_regression_reinforces_talored)
        # primal_gap_final_reinforce_talored_ave = primal_gap_final_reinforces_talored.sum() / len(
        #     primal_gap_final_reinforces_talored)

        print(instance_type)
        print(incumbent_mode + 'Solution')
        print('scip-baseline primal integral: ', primal_int_scip_baseline_ave)
        print('scip-lb-baseline primal integral: ', primal_int_base_ave)
        print('scip-lb-regression primal integral: ', primal_int_lns_random_ave)
        print('scip-lb-rl primal integral: ', primal_int_lns_lblp_ave)
        print('scip-lb-regression-rl primal integral: ', primal_int_lns_lblpmcts_ave)

        print('rl primal integral: ', primal_int_reinforce_ave)


        print('\n')
        print('scip-baseline primal gap: ', primal_gap_final_scip_baseline_ave)
        print('scip-lb-baseline primal gap: ', primal_gap_final_baseline_ave)
        print('scip-lb-regression primal gap: ', primal_gap_final_lns_random_ave)
        print('scip-lb-rl primal gap: ', primal_gap_final_lns_lblp_ave)
        print('scip-lb-regression-rl primal gap: ', primal_gap_final_lns_lblpmcts_ave)

        print('rl primal gap: ', primal_gap_final_reinforce_ave)


        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)

        primalgaps_baseline = None
        for n, stepline_baseline in enumerate(steplines_baseline):
            primal_gap = stepline_baseline(t)
            if n==0:
                primalgaps_baseline = primal_gap.reshape(1,-1)
            else:
                primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        primalgap_baseline_ave = np.average(primalgaps_baseline, axis=0)

        primalgaps_lns_lblp = None
        for n, stepline_lns_lblp in enumerate(steplines_lns_lblp_list):
            primal_gap = stepline_lns_lblp(t)
            if n == 0:
                primalgaps_lns_lblp = primal_gap.reshape(1,-1)
            else:
                primalgaps_lns_lblp = np.vstack((primalgaps_lns_lblp, primal_gap))
        primalgap_lns_lblp_ave = np.average(primalgaps_lns_lblp, axis=0)

        primalgaps_lns_random = None
        for n, stepline_lns_lblp in enumerate(steplines_lns_random_list):
            primal_gap = stepline_lns_lblp(t)
            if n == 0:
                primalgaps_lns_random = primal_gap.reshape(1,-1)
            else:
                primalgaps_lns_random = np.vstack((primalgaps_lns_random, primal_gap))
        primalgap_lns_random_ave = np.average(primalgaps_lns_random, axis=0)

        primalgaps_scip_baseline = None
        for n, stepline_scip in enumerate(steplines_scip_baseline_list):
            primal_gap = stepline_scip(t)
            if n == 0:
                primalgaps_scip_baseline = primal_gap.reshape(1,-1)
            else:
                primalgaps_scip_baseline = np.vstack((primalgaps_scip_baseline, primal_gap))
        primalgap_scip_baseline_ave = np.average(primalgaps_scip_baseline, axis=0)

        primalgaps_lns_lblpmcts = None
        for n, stepline_lns_lblpmcts in enumerate(steplines_lns_lblpmcts_list):
            primal_gap = stepline_lns_lblpmcts(t)
            if n == 0:
                primalgaps_lns_lblpmcts = primal_gap.reshape(1, -1)
            else:
                primalgaps_lns_lblpmcts = np.vstack((primalgaps_lns_lblpmcts, primal_gap))
        primalgap_lns_lblpmcts_ave = np.average(primalgaps_lns_lblpmcts, axis=0)
        #
        # primalgaps_reinforce = None
        # for n, stepline_reinforce in enumerate(steplines_reinforce):
        #     primal_gap = stepline_reinforce(t)
        #     if n == 0:
        #         primalgaps_reinforce = primal_gap
        #     else:
        #         primalgaps_reinforce = np.vstack((primalgaps_reinforce, primal_gap))
        # primalgap_reinforce_ave = np.average(primalgaps_reinforce, axis=0)

        # primalgaps_regression_reinforce_talored = None
        # for n, stepline_scip in enumerate(steplines_regression_reinforce_talored):
        #     primal_gap = stepline_scip(t)
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
        fig.suptitle(instance_name + '-' + 'primal gap' , fontsize=13) # instance_name
        ax.set_title(instance_type + instance_size + '-' + incumbent_mode, fontsize=14)
        ax.plot(t, primalgap_scip_baseline_ave, '--', label='scip', color='tab:grey')
        ax.plot(t, primalgap_baseline_ave, label='scip-lb', color='tab:blue')
        ax.plot(t, primalgap_lns_random_ave, label='scip-lb-regression', color='tab:orange')
        ax.plot(t, primalgap_lns_lblp_ave, label='scip-lb-rl', color='tab:red')
        # ax.plot(t, primalgap_lns_lblp_ave, label='lns_guided_by_lblp', color='tab:red')
        ax.plot(t, primalgap_lns_lblpmcts_ave, label='scip-lb-regression-rl', color='tab:green')
        # ax.plot(t, primalgap_reinforce_ave, '--', label='lb-rl', color='tab:green')
        #
        # ax.plot(t, primalgap_reinforce_talored_ave, ':', label='lb-rl-active', color='tab:green')
        # ax.plot(t, primalgap_regression_reinforce_talored_ave, ':', label='lb-regression-rl-active', color='tab:red')

        ax.set_xlabel('time /s', fontsize=12)
        ax.set_ylabel("scaled primal gap", fontsize=12)
        ax.legend()
        ax.grid()
        # fig.suptitle("Scaled primal gap", y=0.97, fontsize=13)
        # fig.tight_layout()
        plt.savefig('./result/plots/seed' + str(seed_mcts) + '_' + instance_type + '_' + incumbent_mode + '_scip' + '_ttotal' + str(total_time_limit)+ '_tnode' + str(node_time_limit) + '_disable_presolve_beforenode.png')
        plt.show()
        plt.clf()

        print("seed mcts: ", seed_mcts)

    def primal_gap(self, seed_mcts=100, instance_type = 'miplib_39binary', instance_size = '-small', incumbent_mode = 'root', total_time_limit=60, node_time_limit=30, result_directory_1=None, result_directory_2=None, result_directory_3=None, result_directory_4=None, result_directory_5=None):

        instance_type = instance_type
        incumbent_mode = incumbent_mode
        instance_size = instance_size

        # direc = './data/generated_instances/' + self.instance_type + '/' + test_instance_size + '/'
        # directory_transformedmodel = direc + 'transformedmodel' + '/'
        #
        # # set directory for the test result of RL-policy1
        # directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/reinforce/test/old_models/'
        # directory_lb_test = directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(
        #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/seed'+ str(self.seed) + '/'
        #
        # # directory_rl_talored = directory_lb_test + 'rlactive/'
        # if self.incumbent_mode == 'firstsol':
        #     directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/' + 'rl/reinforce/test/old_models/'
        #     directory_lb_test_2 = directory_2 + 'evaluation-reinforce4lb-from-' +  'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/seed'+ str(self.seed) + '/'
        # elif self.incumbent_mode == 'rootsol':
        #     directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/' + 'rl/reinforce/test/old_models/'
        #     directory_lb_test_2 = directory_2 + 'evaluation-reinforce4lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/rlactive/seed'+ str(self.seed) + '/'
        #
        # # directory_3 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # # directory_lb_test_3 = directory_3 + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
        # #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'


        # directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # # directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
        # #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        # if self.incumbent_mode == 'firstsol':
        #     directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/'
        #     # directory_lb_test_2 = directory_2 + 'lb-from-' + 'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(
        #     #     total_time_limit) + 's' + test_instance_size + '/'
        #     directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
        #         node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/seed'+ str(self.seed) + '/'
        #     directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
        #         node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/seed'+ str(self.seed) + '/'
        #     directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'rootsol' + '-t_node' + str(
        #         node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/seed'+ str(self.seed) + '/'
        #
        #
        # elif self.incumbent_mode == 'rootsol':
        #     directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/'
        #     # directory_lb_test_2 = directory_2 + 'lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(
        #     #     total_time_limit) + 's' + test_instance_size + '/'
        #     directory_lb_test_k_prime_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
        #         node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/seed'+ str(self.seed) + '/'
        #     directory_lb_test_k_prime_merged_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
        #         node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_merged/seed'+ str(self.seed) + '/'
        #     directory_lb_test_baseline_2 = directory_2 + 'k_prime/' + 'lb-from-' + 'firstsol' + '-t_node' + str(
        #         node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '_baseline/seed'+ str(self.seed) + '/'

        # baseline algorithms
        directory_lns_random = result_directory_1
        directory_local_branch = result_directory_2
        directory_lns_lblp = result_directory_3
        directory_scip_baseline = result_directory_4
        directory_lns_lblpmcts = result_directory_5

        primal_int_baselines = []
        primal_int_lns_random_list = []
        primal_int_lns_lblp_list = []
        primal_int_scip_baselines_list = []
        primal_int_lns_lblpmcts_list = []
        primal_int_reinforces = []
        primal_gap_final_baselines = []
        primal_gap_final_lns_lblp_list = []
        primal_gap_final_lns_random_list = []
        primal_gap_final_scip_baselines_list = []
        primal_gap_final_lns_lblpmcts_list = []
        primal_gap_final_reinforces = []
        steplines_baseline = []
        steplines_lns_lblp_list = []
        steplines_lns_random_list = []
        steplines_scip_baseline_list = []
        steplines_lns_lblpmcts_list = []
        steplines_reinforce = []

        # primal_int_regression_reinforces_talored = []
        # primal_int_reinforces_talored = []
        # primal_gap_final_regression_reinforces_talored = []
        # primal_gap_final_reinforces_talored = []
        # steplines_regression_reinforce_talored = []
        # steplines_reinforce_talored = []

        if instance_type == instancetypes[3]:
            index_mix = 80
            index_max = 115
        elif instance_type == instancetypes[4]:
            index_mix = 0
            index_max = 20 # 30

        elif instance_type == instancetypes[5]:
            index_mix = 0
            index_max = 3 # 30

        for i in range(index_mix,index_max):

            if not (instance_type == instancetypes[4] and (i == 18)): # or i==4

                instance_name = instance_type + '-' + str(i) + '_transformed' # instance 100-199

                mip_filename = f'{self.instance_directory}{instance_name}.cip'
                mip = Model()
                MIP_model = Model()
                MIP_model.readProblem(mip_filename)
                instance_name = MIP_model.getProbName()

                # filename = f'{directory_lb_test}lb-test-{instance_name}.pkl'
                #
                # with gzip.open(filename, 'rb') as f:
                #     data = pickle.load(f)
                # objs_reinforce, times_reinforce, objs_regresison_reinforce, times_regression_reinforce = data  # objs contains objs of a single instance of a lb test
                #
                # filename = f'{directory_lb_test_2}lb-test-{instance_name}.pkl'
                #
                # with gzip.open(filename, 'rb') as f:
                #     data = pickle.load(f)
                # objs_reinforce_2, times_reinforce_2, objs_regresison_reinforce_2, times_regression_reinforce_2 = data  # objs contains objs of a single instance of a lb test
                #
                # # filename_3 = f'{directory_lb_test_3}lb-test-{instance_name}.pkl'
                # #
                # # with gzip.open(filename_3, 'rb') as f:
                # #     data = pickle.load(f)
                # # objs, times, objs_pred_2, times_pred_2, objs_pred_reset_2, times_pred_reset_2 = data  # objs contains objs of a single instance of a lb test
                # #
                # # objs_regression = objs_pred_reset_2
                # # times_regression = times_pred_reset_2
                #

                # test from lns-guided-by-localbranch-lp heuristic
                filename = f'{directory_lns_lblp}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_lns_lblp, times_lns_lblp = data  # objs contains objs of a single instance of a lb test

                # test from lns-random heuristic
                filename = f'{directory_lns_random}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_lns_random, times_lns_random, bits_lns_random, bits_objs_lns_random, bits_times_lns_random = data  # objs contains objs of a single instance of a lb test

                # test from localbranch baseline
                filename = f'{directory_local_branch}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_lb, times_lb, bits_lb, bits_objs_lb, bits_times_lb = data  # objs contains objs of a single instance of a lb test

                filename = f'{directory_scip_baseline}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_scip, times_scip = data  # objs contains objs of a single instance of a lb test
                #
                # test from lns-guided-by-localbranch-lp-mcts heuristic
                filename = f'{directory_lns_lblpmcts}lb-test-{instance_name}.pkl'
                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_lns_lblpmcts, times_lns_lblpmcts = data  # objs contains objs of a single instance of a lb test

                # # filename = f'{directory_lb_test_k_prime_2}lb-test-{instance_name}.pkl'
                # # with gzip.open(filename, 'rb') as f:
                # #     data = pickle.load(f)
                # # objs_k_prime_2, times_k_prime_2 = data  # objs contains objs of a single instance of a lb test
                #
                # filename = f'{directory_lb_test_k_prime_merged_2}lb-test-{instance_name}.pkl'
                # with gzip.open(filename, 'rb') as f:
                #     data = pickle.load(f)
                # objs_k_prime_merged_2, times_k_prime_merged_2 = data  # objs contains objs of a single instance of a lb test
                #
                # filename = f'{directory_lb_test_baseline_2}lb-test-{instance_name}.pkl'
                # with gzip.open(filename, 'rb') as f:
                #     data = pickle.load(f)
                # objs_2, times_k_2 = data  # objs contains objs of a single instance of a lb test

                # objs_reinforce = np.array(objs_reinforce).reshape(-1)
                # times_reinforce = np.array(times_reinforce).reshape(-1)
                # objs_regresison_reinforce = np.array(objs_regresison_reinforce).reshape(-1)
                # times_regression_reinforce = np.array(times_regression_reinforce).reshape(-1)
                #
                # objs_reinforce_2 = np.array(objs_reinforce_2).reshape(-1)
                # objs_regresison_reinforce_2 = np.array(objs_regresison_reinforce_2).reshape(-1)

                # objs = np.array(objs).reshape(-1)
                # times = np.array(times).reshape(-1)

                # objs_2 = np.array(objs_2).reshape(-1)

                # objs_k_prime = np.array(objs_k_prime).reshape(-1)
                # times_lns_lblp = np.array(times_lns_lblp).reshape(-1)
                #
                # objs_k_prime_2 = np.array(objs_k_prime_2).reshape(-1)

                # objs_lns_random = np.array(objs_lns_random).reshape(-1)
                # times_lns_random = np.array(times_lns_random).reshape(-1)

                # objs_k_prime_merged_2 = np.array(objs_k_prime_merged_2).reshape(-1)

                # a = [objs_regression.min(), objs_regresison_reinforce.min(), objs_reset_vanilla_2.min(), objs_reset_imitation_2.min()]
                a = [objs_lb.min(),objs_lns_random.min(), objs_lns_lblp.min()] # objs_lns_lblp.min(), objs_scip.min(), objs_lns_lblpmcts.min()
                obj_opt = np.amin(a)

                # localbranch-baseline:
                # compute primal gap for baseline localbranching run
                # if times[-1] < total_time_limit:
                primal_int_baseline, primal_gap_final_baseline, stepline_baseline, gamma_baseline_lb = self.compute_primal_integral(times=bits_times_lb, objs=bits_objs_lb, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_baselines.append(primal_gap_final_baseline)
                steplines_baseline.append(stepline_baseline)
                primal_int_baselines.append(primal_int_baseline)

                # # lns-guided-by-localbranch-lp
                # # if times_regression[-1] < total_time_limit:
                #
                # primal_int_lns_lblp, primal_gap_final_lns_lblp, stepline_lns_lblp, = self.compute_primal_integral(
                #     times=times_lns_lblp, objs=objs_lns_lblp, obj_opt=obj_opt, total_time_limit=total_time_limit)
                # primal_gap_final_lns_lblp_list.append(primal_gap_final_lns_lblp)
                # steplines_lns_lblp_list.append(stepline_lns_lblp)
                # primal_int_lns_lblp_list.append(primal_int_lns_lblp)

                # lns-random heuristic
                # if times_regression[-1] < total_time_limit:

                primal_int_lns_random, primal_gap_final_lns_random, stepline_regression_lns_random, gamma_lns_random = self.compute_primal_integral(
                    times=bits_times_lns_random, objs=bits_objs_lns_random, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_lns_random_list.append(primal_gap_final_lns_random)
                steplines_lns_random_list.append(stepline_regression_lns_random)
                primal_int_lns_random_list.append(primal_int_lns_random)

                # scip-baseline

                primal_int_scip, primal_gap_final_scip, stepline_scip, _ = self.compute_primal_integral(
                    times=times_scip, objs=objs_scip, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_scip_baselines_list.append(primal_gap_final_scip)
                steplines_scip_baseline_list.append(stepline_scip)
                primal_int_scip_baselines_list.append(primal_int_scip)

                # lns-guided-by-localbranch-lp-mcts
                primal_int_lns_lblpmcts, primal_gap_final_lns_lblpmcts, stepline_lns_lblpmcts, _ = self.compute_primal_integral(
                    times=times_lns_lblpmcts, objs=objs_lns_lblpmcts, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_lns_lblpmcts_list.append(primal_gap_final_lns_lblpmcts)
                steplines_lns_lblpmcts_list.append(stepline_lns_lblpmcts)
                primal_int_lns_lblpmcts_list.append(primal_int_lns_lblpmcts)

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

                #
                # # lb-reinforce
                #
                # primal_int_reinforce, primal_gap_final_reinforce, stepline_reinforce = self.compute_primal_integral(
                #     times=times_reinforce, objs=objs_reinforce, obj_opt=obj_opt,
                #     total_time_limit=total_time_limit)
                # primal_gap_final_reinforces.append(primal_gap_final_reinforce)
                # steplines_reinforce.append(stepline_reinforce)
                # primal_int_reinforces.append(primal_int_reinforce)

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

                # bits_lb = np.append(bits_lb, bits_lb[-1])
                # bits_lns_random = np.append(bits_lns_random, bits_lns_random[-1])
                # length = np.minimum(len(bits_lb), len(bits_lns_random))
                #
                #
                # plt.close('all')
                # plt.clf()
                # fig, ax = plt.subplots(figsize=(6.4, 4.8))
                # fig.suptitle(instance_name + '-' + 'primal gap', fontsize=13)  # instance_name
                # ax.set_title(instance_type + instance_size + '-' + incumbent_mode, fontsize=14)
                # # ax.plot(t, primalgap_scip_baseline_ave, '--', label='scip', color='tab:grey')
                # ax.plot(bits_lb[0:length], bits_objs_lb[0:length], label='expert policy', color='tab:blue')
                # ax.plot(bits_lns_random[0:length], bits_objs_lns_random[0:length], label='LNS-mutation', color='tab:orange')
                # # ax.plot(t, primalgap_lns_lblp_ave, label='lns_guided_by_lblp', color='tab:red')
                # # ax.plot(t, primalgap_lns_lblpmcts_ave, label='lns_guided_by_lblpmcts', color='tab:green')
                # # ax.plot(t, primalgap_reinforce_ave, '--', label='lb-rl', color='tab:green')
                # #
                # # ax.plot(t, primalgap_reinforce_talored_ave, ':', label='lb-rl-active', color='tab:green')
                # # ax.plot(t, primalgap_regression_reinforce_talored_ave, ':', label='lb-regression-rl-active', color='tab:red')
                #
                # ax.set_xlabel('iteration', fontsize=12)
                # ax.set_ylabel("objective", fontsize=12)
                # ax.legend()
                # ax.grid()
                # # fig.suptitle("Scaled primal gap", y=0.97, fontsize=13)
                # # fig.tight_layout()
                # plt.savefig('./plots/seed' + str(seed_mcts) + '_' + instance_type + '_' + incumbent_mode + '.png')
                # plt.show()
                # plt.clf()
                #
                # plt.close('all')
                # plt.clf()
                # fig, ax = plt.subplots(figsize=(6.4, 4.8))
                # fig.suptitle(instance_name + '-' + 'primal gap', fontsize=13)  # instance_name
                # ax.set_title(instance_type + instance_size + '-' + incumbent_mode, fontsize=14)
                # # ax.plot(t, primalgap_scip_baseline_ave, '--', label='scip', color='tab:grey')
                # ax.plot(bits_times_lb, bits_objs_lb, label='expert policy', color='tab:blue')
                # ax.plot(bits_times_lns_random, bits_objs_lns_random, label='LNS-mutation',
                #         color='tab:orange')
                # # ax.plot(t, primalgap_lns_lblp_ave, label='lns_guided_by_lblp', color='tab:red')
                # # ax.plot(t, primalgap_lns_lblpmcts_ave, label='lns_guided_by_lblpmcts', color='tab:green')
                # # ax.plot(t, primalgap_reinforce_ave, '--', label='lb-rl', color='tab:green')
                # #
                # # ax.plot(t, primalgap_reinforce_talored_ave, ':', label='lb-rl-active', color='tab:green')
                # # ax.plot(t, primalgap_regression_reinforce_talored_ave, ':', label='lb-regression-rl-active', color='tab:red')
                #
                # ax.set_xlabel('time', fontsize=12)
                # ax.set_ylabel("objective", fontsize=12)
                # ax.legend()
                # ax.grid()
                # # fig.suptitle("Scaled primal gap", y=0.97, fontsize=13)
                # # fig.tight_layout()
                # plt.savefig('./plots/seed' + str(seed_mcts) + '_' + instance_type + '_' + incumbent_mode + '.png')
                # plt.show()
                # plt.clf()

                # lns-guided-by-localbranch-lp
                # if times_regression[-1] < total_time_limit:

                primal_int_lns_lblp, primal_gap_final_lns_lblp, stepline_lns_lblp, _ = self.compute_primal_integral(
                    times=times_lns_lblp, objs=objs_lns_lblp, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_lns_lblp_list.append(primal_gap_final_lns_lblp)
                steplines_lns_lblp_list.append(stepline_lns_lblp)
                primal_int_lns_lblp_list.append(primal_int_lns_lblp)

                primal_int_baseline, primal_gap_final_baseline, stepline_baseline, gamma_baseline_lb = self.compute_primal_integral(
                    times=times_lb, objs=objs_lb, obj_opt=obj_opt, total_time_limit=total_time_limit)
                primal_gap_final_baselines.append(primal_gap_final_baseline)
                steplines_baseline.append(stepline_baseline)
                primal_int_baselines.append(primal_int_baseline)

                primal_int_lns_random, primal_gap_final_lns_random, stepline_regression_lns_random, gamma_lns_random = self.compute_primal_integral(
                    times=times_lns_random, objs=objs_lns_random, obj_opt=obj_opt,
                    total_time_limit=total_time_limit)
                primal_gap_final_lns_random_list.append(primal_gap_final_lns_random)
                steplines_lns_random_list.append(stepline_regression_lns_random)
                primal_int_lns_random_list.append(primal_int_lns_random)

        primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_lns_lblp_list = np.array(primal_int_lns_lblp_list).reshape(-1)
        primal_int_lns_random_list = np.array(primal_int_lns_random_list).reshape(-1)
        primal_int_scip_baselines_list = np.array(primal_int_scip_baselines_list).reshape(-1)
        primal_int_lns_lblpmcts_list = np.array(primal_int_lns_lblpmcts_list).reshape(-1)
        primal_int_reinforces = np.array(primal_int_reinforces).reshape(-1)

        # primal_int_regression_reinforces_talored = np.array(primal_int_regression_reinforces_talored).reshape(-1)
        # primal_int_reinforces_talored = np.array(primal_int_reinforces_talored).reshape(-1)

        primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_lns_lblp_list = np.array(primal_gap_final_lns_lblp_list).reshape(-1)
        primal_gap_final_lns_random_list = np.array(primal_gap_final_lns_random_list).reshape(-1)
        primal_gap_final_scip_baselines_list = np.array(primal_gap_final_scip_baselines_list).reshape(-1)
        primal_gap_final_lns_lblpmcts_list = np.array(primal_gap_final_lns_lblpmcts_list).reshape(-1)
        primal_gap_final_reinforces = np.array(primal_gap_final_reinforces).reshape(-1)

        # primal_gap_final_regression_reinforces_talored = np.array(primal_gap_final_regression_reinforces_talored).reshape(-1)
        # primal_gap_final_reinforces_talored = np.array(primal_gap_final_reinforces_talored).reshape(-1)

        # avarage primal integral over test dataset
        primal_int_base_ave = primal_int_baselines.sum() / len(primal_int_baselines)
        primal_int_lns_lblp_ave = primal_int_lns_lblp_list.sum() / len(primal_int_lns_lblp_list)
        primal_int_lns_random_ave = primal_int_lns_random_list.sum() / len(primal_int_lns_random_list)
        primal_int_scip_baseline_ave = primal_int_scip_baselines_list.sum() / len(primal_int_scip_baselines_list)
        primal_int_lns_lblpmcts_ave = primal_int_lns_lblpmcts_list.sum() / len(primal_int_lns_lblpmcts_list)
        primal_int_reinforce_ave = primal_int_reinforces.sum() / len(
            primal_int_reinforces)

        # primal_int_regression_reinforce_talored_ave = primal_int_regression_reinforces_talored.sum() / len(primal_int_regression_reinforces_talored)
        # primal_int_reinforce_talored_ave = primal_int_reinforces_talored.sum() / len(
        #     primal_int_reinforces_talored)

        primal_gap_final_baseline_ave = primal_gap_final_baselines.sum() / len(primal_gap_final_baselines)
        primal_gap_final_lns_lblp_ave = primal_gap_final_lns_lblp_list.sum() / len(primal_gap_final_lns_lblp_list)
        primal_gap_final_lns_random_ave = primal_gap_final_lns_random_list.sum() / len(primal_gap_final_lns_random_list)
        primal_gap_final_scip_baseline_ave = primal_gap_final_scip_baselines_list.sum() / len(primal_gap_final_scip_baselines_list)
        primal_gap_final_lns_lblpmcts_ave = primal_gap_final_lns_lblpmcts_list.sum() / len(primal_gap_final_lns_lblpmcts_list)
        primal_gap_final_reinforce_ave = primal_gap_final_reinforces.sum() / len(
            primal_gap_final_reinforces)

        # primal_gap_final_regression_reinforce_talored_ave = primal_gap_final_regression_reinforces_talored.sum() / len(
        #     primal_gap_final_regression_reinforces_talored)
        # primal_gap_final_reinforce_talored_ave = primal_gap_final_reinforces_talored.sum() / len(
        #     primal_gap_final_reinforces_talored)

        print(instance_type)
        print(incumbent_mode + 'Solution')
        print('scip-baseline primal integral: ', primal_int_scip_baseline_ave)
        print('localbranch primal integral: ', primal_int_base_ave)
        print('lns-random primal integral: ', primal_int_lns_random_ave)
        print('lns_guided_by_lblp primal integral: ', primal_int_lns_lblp_ave)
        print('lns_guided_by_lblpmcts primal integral: ', primal_int_lns_lblpmcts_ave)

        print('rl primal integral: ', primal_int_reinforce_ave)


        print('\n')
        print('scip-baseline primal gap: ', primal_gap_final_scip_baseline_ave)
        print('localbranch primal gap: ', primal_gap_final_baseline_ave)
        print('lns-random primal gap: ', primal_gap_final_lns_random_ave)
        print('lns_guided_by_lblp primal gap: ', primal_gap_final_lns_lblp_ave)
        print('lns_guided_by_lblpmcts primal gap: ', primal_gap_final_lns_lblpmcts_ave)

        print('rl primal gap: ', primal_gap_final_reinforce_ave)


        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)

        primalgaps_baseline = None
        for n, stepline_baseline in enumerate(steplines_baseline):
            primal_gap = stepline_baseline(t)
            if n==0:
                primalgaps_baseline = primal_gap.reshape(1,-1)
            else:
                primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        primalgap_baseline_ave = np.average(primalgaps_baseline, axis=0)

        primalgaps_lns_lblp = None
        for n, stepline_lns_lblp in enumerate(steplines_lns_lblp_list):
            primal_gap = stepline_lns_lblp(t)
            if n == 0:
                primalgaps_lns_lblp = primal_gap.reshape(1,-1)
            else:
                primalgaps_lns_lblp = np.vstack((primalgaps_lns_lblp, primal_gap))
        primalgap_lns_lblp_ave = np.average(primalgaps_lns_lblp, axis=0)

        primalgaps_lns_random = None
        for n, stepline_lns_lblp in enumerate(steplines_lns_random_list):
            primal_gap = stepline_lns_lblp(t)
            if n == 0:
                primalgaps_lns_random = primal_gap.reshape(1,-1)
            else:
                primalgaps_lns_random = np.vstack((primalgaps_lns_random, primal_gap))
        primalgap_lns_random_ave = np.average(primalgaps_lns_random, axis=0)

        primalgaps_scip_baseline = None
        for n, stepline_scip in enumerate(steplines_scip_baseline_list):
            primal_gap = stepline_scip(t)
            if n == 0:
                primalgaps_scip_baseline = primal_gap.reshape(1,-1)
            else:
                primalgaps_scip_baseline = np.vstack((primalgaps_scip_baseline, primal_gap))
        primalgap_scip_baseline_ave = np.average(primalgaps_scip_baseline, axis=0)

        primalgaps_lns_lblpmcts = None
        for n, stepline_lns_lblpmcts in enumerate(steplines_lns_lblpmcts_list):
            primal_gap = stepline_lns_lblpmcts(t)
            if n == 0:
                primalgaps_lns_lblpmcts = primal_gap.reshape(1, -1)
            else:
                primalgaps_lns_lblpmcts = np.vstack((primalgaps_lns_lblpmcts, primal_gap))
        primalgap_lns_lblpmcts_ave = np.average(primalgaps_lns_lblpmcts, axis=0)

        #
        # primalgaps_reinforce = None
        # for n, stepline_reinforce in enumerate(steplines_reinforce):
        #     primal_gap = stepline_reinforce(t)
        #     if n == 0:
        #         primalgaps_reinforce = primal_gap
        #     else:
        #         primalgaps_reinforce = np.vstack((primalgaps_reinforce, primal_gap))
        # primalgap_reinforce_ave = np.average(primalgaps_reinforce, axis=0)

        # primalgaps_regression_reinforce_talored = None
        # for n, stepline_scip in enumerate(steplines_regression_reinforce_talored):
        #     primal_gap = stepline_scip(t)
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
        # fig.suptitle(instance_name + '-' + 'primal gap' , fontsize=13) # instance_name
        ax.set_title(instance_type + instance_size + '-' + incumbent_mode, fontsize=14)
        ax.plot(t, primalgap_scip_baseline_ave, '--', label='scip', color='tab:grey')
        ax.plot(t, primalgap_baseline_ave, label='expert policy', color='tab:blue')
        ax.plot(t, primalgap_lns_random_ave, label='LNS-mutation', color='tab:orange')
        ax.plot(t, primalgap_lns_lblp_ave, label='lns-expert', color='tab:red')
        # ax.plot(bits_lb, gamma_baseline_lb, label='expert policy', color='tab:blue')
        # ax.plot(bits_lns_random, gamma_lns_random, label='LNS-mutation', color='tab:orange')
        # ax.plot(t, primalgap_lns_lblp_ave, label='lns_guided_by_lblp', color='tab:red')
        ax.plot(t, primalgap_lns_lblpmcts_ave, label='lns-GNN', color='tab:green')
        # ax.plot(t, primalgap_reinforce_ave, '--', label='lb-rl', color='tab:green')
        #
        # ax.plot(t, primalgap_reinforce_talored_ave, ':', label='lb-rl-active', color='tab:green')
        # ax.plot(t, primalgap_regression_reinforce_talored_ave, ':', label='lb-regression-rl-active', color='tab:red')

        ax.set_xlabel('time /s', fontsize=12)
        ax.set_ylabel("scaled primal gap", fontsize=12)
        ax.legend()
        ax.grid()
        # fig.suptitle("Scaled primal gap", y=0.97, fontsize=13)
        # fig.tight_layout()
        plt.savefig('./plots/seed' + str(seed_mcts) + '_' + instance_type + '_' + incumbent_mode + '.png')
        plt.show()
        plt.clf()

        print("seed mcts: ", seed_mcts)

class Execute_LB_Baseline(ExecuteHeuristic):

    def __init__(self, instance_directory, solution_directory, result_derectory, lbconstraint_mode,
                 no_improve_iteration_limit=20, seed=100, enable_gpu=False,
                 is_heuristic=False):
        super().__init__(instance_directory, solution_directory, result_derectory,
                         no_improve_iteration_limit=no_improve_iteration_limit, seed=seed, enable_gpu=enable_gpu)

        self.lbconstraint_mode = lbconstraint_mode
        self.is_heuristic = is_heuristic
        self.k_0 = 20

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_0 = self.k_0 / 2

    def execute_heuristic_per_instance(self, MIP_model, incumbent, node_time_limit, total_time_limit):
        """
        call the underlying heuristic method over an MIP instance, this is the basic method by directly running scip to solve the problem
        :param MIP_model:
        :param incumbent:
        :param node_time_limit:
        :param total_time_limit:
        :return:
        """

        objs = []
        times = []
        MIP_obj_best = MIP_model.getSolObjVal(incumbent)
        times.append(0.0)
        objs.append(MIP_obj_best)

        primalbound_handler = PrimalBoundChangeEventHandler()
        primalbound_handler.primal_times = []
        primalbound_handler.primal_bounds = []

        MIP_model.includeEventhdlr(primalbound_handler, 'primal_bound_update_handler',
                                   'store every new primal bound and its time stamp')

        heuristic = HeurLocalbranch(k_0=self.k_0, node_time_limit=node_time_limit, total_time_limit=total_time_limit, is_symmetric=self.is_symmetric, is_heuristic=self.is_heuristic, reset_k_at_2nditeration=False, no_improve_iteration_limit = self.no_improve_iteration_limit,  device=self.device)
        MIP_model.includeHeur(heuristic,
                              "PyHeur_LB_baseline",
                              "Localbranching baseline heuristic implemented in python",
                              "Y",
                              priority=130000,
                              freq=0,
                              freqofs=0,
                              maxdepth=-1,
                              timingmask=SCIP_HEURTIMING.BEFORENODE, # SCIP_HEURTIMING.AFTERLPNODE
                              usessubscip=True
                              )

        MIP_model.setParam('limits/time', total_time_limit)
        MIP_model.setParam("display/verblevel", 0)
        MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.FAST)
        MIP_model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
        MIP_model.optimize()
        status = MIP_model.getStatus()
        n_sols_MIP = MIP_model.getNSols()
        # MIP_model.freeTransform()
        # feasible = MIP_model.checkSol(solution=MIP_model.getBestSol())
        elapsed_time = MIP_model.getSolvingTime()


        if n_sols_MIP > 0:
            feasible, MIP_sol_incumbent, MIP_obj_incumbent = getBestFeasiSol(MIP_model)
            feasible = MIP_model.checkSol(solution=MIP_sol_incumbent)
            assert feasible, "Error: the best solution from current SCIP solving is not feasible!"
            # MIP_obj_incumbent = MIP_model.getSolObjVal(MIP_sol_incumbent)

            if MIP_obj_incumbent < MIP_obj_best:
                primal_bounds = primalbound_handler.primal_bounds
                primal_times = primalbound_handler.primal_times
                MIP_obj_best = MIP_obj_incumbent

                # for i in range(len(primal_times)):
                #     primal_times[i] += self.total_time_expired

                objs.extend(primal_bounds)
                times.extend(primal_times)

        obj_best = MIP_obj_best
        objs = np.array(objs).reshape(-1)
        times = np.array(times).reshape(-1)

        print("Instance:", MIP_model.getProbName())
        print("Status of SCIP_baseline: ", status)
        print("Best obj of SCIP_baseline: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        data = [objs, times]

        return data

class Execute_LB_Regression(ExecuteHeuristic):

    def __init__(self, instance_directory, solution_directory, result_derectory, lbconstraint_mode,
                 no_improve_iteration_limit=20, seed=100, enable_gpu=False,
                 is_heuristic=False, instance_type='miplib_39binary', incumbent_mode='firstsol', regression_model_gnn=None):
        super().__init__(instance_directory, solution_directory, result_derectory,
                         no_improve_iteration_limit=no_improve_iteration_limit, seed=seed, enable_gpu=enable_gpu)

        self.lbconstraint_mode = lbconstraint_mode
        self.is_heuristic = is_heuristic
        self.k_0 = 20

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_0 = self.k_0 / 2
        self.incumbent_mode = incumbent_mode
        self.instance_type = instance_type

        self.initialize_ecole_env()
        self.env.seed(self.seed)  # environment (SCIP)
        self.regression_model_gnn = regression_model_gnn
        self.regression_model_gnn.to(self.device)


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

    def execute_heuristic_per_instance(self, MIP_model, incumbent, node_time_limit, total_time_limit):
        """
        call the underlying heuristic method over an MIP instance, this is the basic method by directly running scip to solve the problem
        :param MIP_model:
        :param incumbent:
        :param node_time_limit:
        :param total_time_limit:
        :return:
        """

        # MIP_model_copy2._freescip = True
        instance = ecole.scip.Model.from_pyscipopt(MIP_model)
        observation, _, _, done, _ = self.env.reset(instance)

        # variable features: only incumbent solution
        variable_features = observation.variable_features[:, -1:]
        graph = BipartiteNodeData(observation.constraint_features,
                                  observation.edge_features.indices,
                                  observation.edge_features.values,
                                  variable_features)
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = observation.constraint_features.shape[0] + \
                          observation.variable_features.shape[
                              0]
        # solve the root node and get the LP solution, compute k_prime
        k_prime = self.compute_k_prime(MIP_model, incumbent)

        k_model = self.regression_model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
                                            graph.variable_features)
        k_pred = k_model.item() * k_prime
        print('GNN prediction: ', k_model.item())
        k_pred = np.ceil(k_pred)
        print('GNN k_0: ', k_pred)

        if k_pred < 10:
            k_pred = 10

        MIP_model.resetParams()
        MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
            problemName='gnn-copy',
            origcopy=False)
        MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent,
                                                  MIP_copy_vars2)


        objs = []
        times = []
        MIP_obj_best = MIP_model_copy2.getSolObjVal(sol_MIP_copy2)
        times.append(0.0)
        objs.append(MIP_obj_best)



        primalbound_handler = PrimalBoundChangeEventHandler()
        primalbound_handler.primal_times = []
        primalbound_handler.primal_bounds = []

        MIP_model_copy2.includeEventhdlr(primalbound_handler, 'primal_bound_update_handler',
                                   'store every new primal bound and its time stamp')

        heuristic = HeurLocalbranch(k_0=k_pred, node_time_limit=node_time_limit, total_time_limit=total_time_limit, is_symmetric=self.is_symmetric, is_heuristic=self.is_heuristic, reset_k_at_2nditeration=False, no_improve_iteration_limit = self.no_improve_iteration_limit,  device=self.device)
        MIP_model_copy2.includeHeur(heuristic,
                                    "PyHeur_LB_baseline",
                                    "Localbranching baseline heuristic implemented in python",
                                    "Y",
                                    priority=130000,
                                    freq=0,
                                    freqofs=0,
                                    maxdepth=-1,
                                    timingmask=SCIP_HEURTIMING.BEFORENODE,  # SCIP_HEURTIMING.AFTERLPNODE
                                    usessubscip=True
                                    )

        MIP_model_copy2.setParam('limits/time', total_time_limit)
        MIP_model_copy2.setParam("display/verblevel", 0)
        MIP_model_copy2.setSeparating(pyscipopt.SCIP_PARAMSETTING.FAST)
        MIP_model_copy2.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
        MIP_model_copy2.optimize()
        status = MIP_model_copy2.getStatus()
        n_sols_MIP = MIP_model_copy2.getNSols()
        # MIP_model.freeTransform()
        # feasible = MIP_model.checkSol(solution=MIP_model.getBestSol())
        elapsed_time = MIP_model_copy2.getSolvingTime()

        if n_sols_MIP > 0:
            feasible, MIP_sol_incumbent, MIP_obj_incumbent = getBestFeasiSol(MIP_model_copy2)
            feasible = MIP_model_copy2.checkSol(solution=MIP_sol_incumbent)
            assert feasible, "Error: the best solution from current SCIP solving is not feasible!"
            # MIP_obj_incumbent = MIP_model.getSolObjVal(MIP_sol_incumbent)

            if MIP_obj_incumbent < MIP_obj_best:
                primal_bounds = primalbound_handler.primal_bounds
                primal_times = primalbound_handler.primal_times
                MIP_obj_best = MIP_obj_incumbent

                # for i in range(len(primal_times)):
                #     primal_times[i] += self.total_time_expired

                objs.extend(primal_bounds)
                times.extend(primal_times)

        obj_best = MIP_obj_best
        objs = np.array(objs).reshape(-1)
        times = np.array(times).reshape(-1)

        print("Instance:", MIP_model_copy2.getProbName())
        print("Status of SCIP_baseline: ", status)
        print("Best obj of SCIP_baseline: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        data = [objs, times]

        return data

class Execute_LB_RL(ExecuteHeuristic):

    def __init__(self, instance_directory, solution_directory, result_derectory, lbconstraint_mode,
                 no_improve_iteration_limit=20, seed=100, enable_gpu=False,
                 is_heuristic=False, agent_k=None, optim_k=None):
        super().__init__(instance_directory, solution_directory, result_derectory,
                         no_improve_iteration_limit=no_improve_iteration_limit, seed=seed, enable_gpu=enable_gpu)

        self.lbconstraint_mode = lbconstraint_mode
        self.is_heuristic = is_heuristic
        self.k_0 = 20

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_0 = self.k_0 / 2

        self.agent_k = agent_k
        self.optim_k = optim_k

    def execute_heuristic_per_instance(self, MIP_model, incumbent, node_time_limit, total_time_limit):
        """
        call the underlying heuristic method over an MIP instance, this is the basic method by directly running scip to solve the problem
        :param MIP_model:
        :param incumbent:
        :param node_time_limit:
        :param total_time_limit:
        :return:
        """

        objs = []
        times = []
        MIP_obj_best = MIP_model.getSolObjVal(incumbent)
        times.append(0.0)
        objs.append(MIP_obj_best)

        primalbound_handler = PrimalBoundChangeEventHandler()
        primalbound_handler.primal_times = []
        primalbound_handler.primal_bounds = []

        MIP_model.includeEventhdlr(primalbound_handler, 'primal_bound_update_handler',
                                   'store every new primal bound and its time stamp')

        heuristic = HeurLocalbranch(k_0=self.k_0, node_time_limit=node_time_limit, total_time_limit=total_time_limit, is_symmetric=self.is_symmetric, is_heuristic=self.is_heuristic, reset_k_at_2nditeration=False, no_improve_iteration_limit = self.no_improve_iteration_limit, device=self.device, agent_k=self.agent_k, optim_k=self.optim_k)
        MIP_model.includeHeur(heuristic,
                              "PyHeur_LB_baseline",
                              "Localbranching baseline heuristic implemented in python",
                              "Y",
                              priority=130000,
                              freq=0,
                              freqofs=0,
                              maxdepth=-1,
                              timingmask=SCIP_HEURTIMING.BEFORENODE,  # SCIP_HEURTIMING.AFTERLPNODE
                              usessubscip=True
                              )

        MIP_model.setParam('limits/time', total_time_limit)
        MIP_model.setParam("display/verblevel", 0)
        MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.FAST)
        MIP_model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
        MIP_model.optimize()
        status = MIP_model.getStatus()
        n_sols_MIP = MIP_model.getNSols()
        # MIP_model.freeTransform()
        # feasible = MIP_model.checkSol(solution=MIP_model.getBestSol())
        elapsed_time = MIP_model.getSolvingTime()


        if n_sols_MIP > 0:
            feasible, MIP_sol_incumbent, MIP_obj_incumbent = getBestFeasiSol(MIP_model)
            feasible = MIP_model.checkSol(solution=MIP_sol_incumbent)
            assert feasible, "Error: the best solution from current SCIP solving is not feasible!"
            # MIP_obj_incumbent = MIP_model.getSolObjVal(MIP_sol_incumbent)

            if MIP_obj_incumbent < MIP_obj_best:
                primal_bounds = primalbound_handler.primal_bounds
                primal_times = primalbound_handler.primal_times
                MIP_obj_best = MIP_obj_incumbent

                # for i in range(len(primal_times)):
                #     primal_times[i] += self.total_time_expired

                objs.extend(primal_bounds)
                times.extend(primal_times)

        obj_best = MIP_obj_best
        objs = np.array(objs).reshape(-1)
        times = np.array(times).reshape(-1)

        print("Instance:", MIP_model.getProbName())
        print("Status of SCIP_baseline: ", status)
        print("Best obj of SCIP_baseline: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        data = [objs, times]

        return data

class Execute_LB_Regression_RL(ExecuteHeuristic):

    def __init__(self, instance_directory, solution_directory, result_derectory, lbconstraint_mode,
                 no_improve_iteration_limit=20, seed=100, enable_gpu=False,
                 is_heuristic=False, instance_type='miplib_39binary', incumbent_mode='firstsol', regression_model_gnn=None, agent_k=None, optim_k=None):
        super().__init__(instance_directory, solution_directory, result_derectory,
                         no_improve_iteration_limit=no_improve_iteration_limit, seed=seed, enable_gpu=enable_gpu)

        self.lbconstraint_mode = lbconstraint_mode
        self.is_heuristic = is_heuristic
        self.k_0 = 20

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_0 = self.k_0 / 2
        self.incumbent_mode = incumbent_mode
        self.instance_type = instance_type

        self.initialize_ecole_env()
        self.env.seed(self.seed)  # environment (SCIP)
        self.regression_model_gnn = regression_model_gnn
        self.regression_model_gnn.to(self.device)
        self.agent_k = agent_k
        self.optim_k = optim_k

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

    def execute_heuristic_per_instance(self, MIP_model, incumbent, node_time_limit, total_time_limit):
        """
        call the underlying heuristic method over an MIP instance, this is the basic method by directly running scip to solve the problem
        :param MIP_model:
        :param incumbent:
        :param node_time_limit:
        :param total_time_limit:
        :return:
        """

        # MIP_model_copy2._freescip = True
        instance = ecole.scip.Model.from_pyscipopt(MIP_model)
        observation, _, _, done, _ = self.env.reset(instance)

        # variable features: only incumbent solution
        variable_features = observation.variable_features[:, -1:]
        graph = BipartiteNodeData(observation.constraint_features,
                                  observation.edge_features.indices,
                                  observation.edge_features.values,
                                  variable_features)
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = observation.constraint_features.shape[0] + \
                          observation.variable_features.shape[
                              0]
        # solve the root node and get the LP solution, compute k_prime
        k_prime = self.compute_k_prime(MIP_model, incumbent)

        k_model = self.regression_model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
                                            graph.variable_features)
        k_pred = k_model.item() * k_prime
        print('GNN prediction: ', k_model.item())
        k_pred = np.ceil(k_pred)
        print('GNN k_0: ', k_pred)

        if k_pred < 10:
            k_pred = 10

        MIP_model.resetParams()
        MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
            problemName='gnn-copy',
            origcopy=False)
        MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent,
                                                  MIP_copy_vars2)

        objs = []
        times = []
        MIP_obj_best = MIP_model_copy2.getSolObjVal(sol_MIP_copy2)
        times.append(0.0)
        objs.append(MIP_obj_best)



        primalbound_handler = PrimalBoundChangeEventHandler()
        primalbound_handler.primal_times = []
        primalbound_handler.primal_bounds = []

        MIP_model_copy2.includeEventhdlr(primalbound_handler, 'primal_bound_update_handler',
                                   'store every new primal bound and its time stamp')

        heuristic = HeurLocalbranch(k_0=k_pred, node_time_limit=node_time_limit, total_time_limit=total_time_limit, is_symmetric=self.is_symmetric, is_heuristic=self.is_heuristic, reset_k_at_2nditeration=False, no_improve_iteration_limit = self.no_improve_iteration_limit,  device=self.device, agent_k=self.agent_k, optim_k=self.optim_k)
        MIP_model_copy2.includeHeur(heuristic,
                                    "PyHeur_LB_baseline",
                                    "Localbranching baseline heuristic implemented in python",
                                    "Y",
                                    priority=130000,
                                    freq=0,
                                    freqofs=0,
                                    maxdepth=-1,
                                    timingmask=SCIP_HEURTIMING.BEFORENODE,  # SCIP_HEURTIMING.AFTERLPNODE
                                    usessubscip=True
                                    )

        MIP_model_copy2.setParam('limits/time', total_time_limit)
        MIP_model_copy2.setParam("display/verblevel", 0)
        MIP_model_copy2.setSeparating(pyscipopt.SCIP_PARAMSETTING.FAST)
        MIP_model_copy2.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
        MIP_model_copy2.optimize()
        status = MIP_model_copy2.getStatus()
        n_sols_MIP = MIP_model_copy2.getNSols()
        # MIP_model.freeTransform()
        # feasible = MIP_model.checkSol(solution=MIP_model.getBestSol())
        elapsed_time = MIP_model_copy2.getSolvingTime()

        if n_sols_MIP > 0:
            feasible, MIP_sol_incumbent, MIP_obj_incumbent = getBestFeasiSol(MIP_model_copy2)
            feasible = MIP_model_copy2.checkSol(solution=MIP_sol_incumbent)
            assert feasible, "Error: the best solution from current SCIP solving is not feasible!"
            # MIP_obj_incumbent = MIP_model.getSolObjVal(MIP_sol_incumbent)

            if MIP_obj_incumbent < MIP_obj_best:
                primal_bounds = primalbound_handler.primal_bounds
                primal_times = primalbound_handler.primal_times
                MIP_obj_best = MIP_obj_incumbent

                # for i in range(len(primal_times)):
                #     primal_times[i] += self.total_time_expired

                objs.extend(primal_bounds)
                times.extend(primal_times)

        obj_best = MIP_obj_best
        objs = np.array(objs).reshape(-1)
        times = np.array(times).reshape(-1)

        print("Instance:", MIP_model_copy2.getProbName())
        print("Status of SCIP_baseline: ", status)
        print("Best obj of SCIP_baseline: ", obj_best)
        print("Solving time: ", elapsed_time)
        print('\n')

        data = [objs, times]

        return data



