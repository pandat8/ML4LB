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
from utilities import lbconstraint_modes, instancetypes, instancesizes, generator_switcher, binary_support, copy_sol, mean_filter,mean_forward_filter, imitation_accuracy, haming_distance_solutions, haming_distance_solutions_asym
from localbranching import addLBConstraint, addLBConstraintAsymmetric
from ecole_extend.environment_extend import SimpleConfiguring, SimpleConfiguringEnablecuts, SimpleConfiguringEnableheuristics
from models import GraphDataset, GNNPolicy, BipartiteNodeData
import torch.nn.functional as F
import torch_geometric
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.interpolate import interp1d

from localbranching import LocalBranching

import gc
import sys
from memory_profiler import profile

from models_rl import SimplePolicy, ImitationLbDataset, AgentReinforce

class InstanceGeneration:
    def __init__(self, instance_type, instance_size, lbconstraint_mode, incumbent_mode='firstsol', seed=100):
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
            elif self.instance_type == 'combinatorialauction':
                heuristics_off = True
                cuts_off = True
            elif self.instance_type == 'generalized_independentset':
                heuristics_off = True
                cuts_off = True
            elif self.instance_type == 'miplib2017_binary':
                heuristics_off = False
                cuts_off = False
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
        else:
            MIP_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.FAST)

        if cuts_off:
            MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
        else:
            MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.FAST)

        if incumbent_mode == 'firstsol':
            MIP_model.setParam('limits/solutions', 1)
        elif incumbent_mode == 'rootsol':
            MIP_model.setParam("limits/nodes", 1)

        MIP_model.setParam("display/verblevel", 0)
        MIP_model.setParam('limits/time', 3600)
        MIP_model.optimize()

        t = MIP_model.getSolvingTime()
        status = MIP_model.getStatus()
        lp_status = MIP_model.getLPSolstat()
        stage = MIP_model.getStage()
        n_sols = MIP_model.getNSols()
        if n_sols == 0:
            MIP_model.freeTransform()
            MIP_model.resetParams()
            if preprocess_off:
                MIP_model.setParam('presolving/maxrounds', 0)
                MIP_model.setParam('presolving/maxrestarts', 0)
            MIP_model.setParam('limits/time', 3600*2)
            # MIP_model.setParam("limits/nodes", -1)
            MIP_model.setParam('limits/solutions', 2)
            MIP_model.setParam("display/verblevel", 0)
            MIP_model.optimize()
            n_sols = MIP_model.getNSols()
            t = MIP_model.getSolvingTime()
            status = MIP_model.getStatus()

        print("* Model status: %s" % status)
        # print("* LP status: %s" % lp_status)
        # print("* Solve stage: %s" % stage)
        print("* Solving time: %s" % t)
        print('* number of sol : ', n_sols)

        if n_sols == 0:
            feasible = False
            incumbent_solution = None
        else:
            incumbent_solution = MIP_model.getBestSol()
            feasible = MIP_model.checkSol(solution=incumbent_solution)

        return status, feasible, MIP_model, incumbent_solution

    def initialize_MIP(self, MIP_model):

        MIP_model_2, MIP_2_vars, success = MIP_model.createCopy(origcopy=True)

        incumbent_mode = self.incumbent_mode
        if self.incumbent_mode == 'firstsol':
            incumbent_mode_2 = 'rootsol'
        elif self.incumbent_mode == 'rootsol':
            incumbent_mode_2 = 'firstsol'

        status, feasible, MIP_model, incumbent_solution = self.set_and_optimize_MIP(MIP_model, incumbent_mode)
        status_2, feasible_2, MIP_model_2, incumbent_solution_2 = self.set_and_optimize_MIP(MIP_model_2,
                                                                                            incumbent_mode_2)

        feasible = (feasible and feasible_2)

        if (not status == 'optimal') and (not status_2 == 'optimal'):
            not_optimal = True
        else:
            not_optimal = False

        if not_optimal and feasible:
            valid = True
        else:
            valid = False

        return valid, MIP_model, incumbent_solution

    def generate_instances(self, instance_type, instance_size):

        directory = './data/generated_instances/' + instance_type + '/' + instance_size + '/'
        directory_transformedmodel = directory + 'transformedmodel' + '/'
        directory_firstsol = directory +'firstsol' + '/'
        directory_rootsol = directory + 'rootsol' + '/'
        pathlib.Path(directory_transformedmodel).mkdir(parents=True, exist_ok=True)
        pathlib.Path(directory_firstsol).mkdir(parents=True, exist_ok=True)
        pathlib.Path(directory_rootsol).mkdir(parents=True, exist_ok=True)

        generator = generator_switcher(instance_type + instance_size)
        generator.seed(self.seed)

        index_instance = 0
        while index_instance < 200: # 200
            instance = next(generator)
            MIP_model = instance.as_pyscipopt()
            MIP_model.setProbName(instance_type + '-' + str(index_instance))
            instance_name = MIP_model.getProbName()
            print('\n')
            print(instance_name)

            # initialize MIP
            MIP_model_2, MIP_2_vars, success = MIP_model.createCopy(
                problemName='Baseline', origcopy=True)

            # MIP_model_orig, MIP_vars_orig, success = MIP_model.createCopy(
            #     problemName='Baseline', origcopy=True)

            incumbent_mode = 'firstsol'
            incumbent_mode_2 = 'rootsol'

            status, feasible, MIP_model, incumbent_solution = self.set_and_optimize_MIP(MIP_model, incumbent_mode)

            status_2, feasible_2, MIP_model_2, incumbent_solution_2 = self.set_and_optimize_MIP(MIP_model_2,
                                                                                                incumbent_mode_2)

            feasible = feasible and feasible_2

            if (not status == 'optimal') and (not status_2 == 'optimal'):
                not_optimal = True
            else:
                not_optimal = False

            if not_optimal and feasible:
                valid = True
            else:
                valid = False

            if valid:

                MIP_model.resetParams()
                MIP_model_transformed, MIP_copy_vars, success = MIP_model.createCopy(
                    problemName='transformed', origcopy=False)
                MIP_model_transformed, sol_MIP_first = copy_sol(MIP_model, MIP_model_transformed, incumbent_solution,
                                                        MIP_copy_vars)
                MIP_model_transformed, sol_MIP_root = copy_sol(MIP_model_2, MIP_model_transformed, incumbent_solution_2,
                                                        MIP_copy_vars)

                n_vars = MIP_model_transformed.getNVars()
                n_binvars = MIP_model_transformed.getNBinVars()
                print("N of variables: {}".format(n_vars))
                print("N of binary vars: {}".format(n_binvars))
                print("N of constraints: {}".format(MIP_model_transformed.getNConss()))

                transformed_model_name = MIP_model_transformed.getProbName()
                filename = f'{directory_transformedmodel}{transformed_model_name}.cip'
                MIP_model_transformed.writeProblem(filename=filename, trans=False)

                firstsol_filename = f'{directory_firstsol}firstsol-{transformed_model_name}.sol'
                MIP_model_transformed.writeSol(solution=sol_MIP_first, filename=firstsol_filename)

                rootsol_filename = f'{directory_rootsol}rootsol-{transformed_model_name}.sol'
                MIP_model_transformed.writeSol(solution=sol_MIP_root, filename=rootsol_filename)

                model = Model()
                model.readProblem(filename)
                sol = model.readSolFile(rootsol_filename)

                feas = model.checkSol(sol)
                if not feas:
                    print('the root solution of '+ model.getProbName()+ 'is not feasible!')

                model.addSol(sol, False)
                print(model.getSolObjVal(sol))
                instance = ecole.scip.Model.from_pyscipopt(model)
                scipMIP = instance.as_pyscipopt()
                sol2 = scipMIP.getBestSol()
                print(scipMIP.getSolObjVal(sol2))

                # MIP_model_2.resetParams()
                # MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model_2.createCopy(
                #     problemName='rootsol',
                #     origcopy=False)
                # MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model_2, MIP_model_copy2, incumbent_solution_2,
                #                                           MIP_copy_vars2)

                MIP_model.freeProb()
                MIP_model_2.freeProb()
                MIP_model_transformed.freeProb()
                model.freeProb()
                del MIP_model
                del MIP_model_2
                del MIP_model_transformed
                del model

                index_instance += 1

    def generate_instances_generalized_independentset(self, instance_type='generalized_independentset', instance_size='-small'):

        directory = './data/generated_instances/' + instance_type + '/' + instance_size + '/'
        directory_transformedmodel = directory + 'transformedmodel' + '/'
        directory_firstsol = directory +'firstsol' + '/'
        directory_rootsol = directory + 'rootsol' + '/'
        pathlib.Path(directory_transformedmodel).mkdir(parents=True, exist_ok=True)
        pathlib.Path(directory_firstsol).mkdir(parents=True, exist_ok=True)
        pathlib.Path(directory_rootsol).mkdir(parents=True, exist_ok=True)

        instance_directory = './data/generated_instances/generalized_independentset/original_lp_instances/'
        filename = '*.lp'
        # print(filename)
        sample_files = [str(path) for path in pathlib.Path(instance_directory).glob(filename)]

        index_instance = 0
        for i, instance in enumerate(sample_files):
            print(instance)
            MIP_model = Model()
            MIP_model.readProblem(instance)

            MIP_model.setProbName(instance_type + '-' + str(index_instance))
            instance_name = MIP_model.getProbName()
            print('\n')
            print(instance_name)
            print('Number of variables', MIP_model.getNVars())
            print('Number of binary variables', MIP_model.getNBinVars())

            # initialize MIP
            MIP_model_2, MIP_2_vars, success = MIP_model.createCopy(
                problemName='Baseline', origcopy=True)

            # MIP_model_orig, MIP_vars_orig, success = MIP_model.createCopy(
            #     problemName='Baseline', origcopy=True)

            incumbent_mode = 'firstsol'
            incumbent_mode_2 = 'rootsol'

            status, feasible, MIP_model, incumbent_solution = self.set_and_optimize_MIP(MIP_model, incumbent_mode)

            status_2, feasible_2, MIP_model_2, incumbent_solution_2 = self.set_and_optimize_MIP(MIP_model_2,
                                                                                                incumbent_mode_2)

            feasible = feasible and feasible_2

            if (not status == 'optimal') and (not status_2 == 'optimal'):
                not_optimal = True
            else:
                not_optimal = False

            if not_optimal and feasible:
                valid = True
            else:
                valid = False

            if valid:

                MIP_model.resetParams()
                MIP_model_transformed, MIP_copy_vars, success = MIP_model.createCopy(
                    problemName='transformed', origcopy=False)
                MIP_model_transformed, sol_MIP_first = copy_sol(MIP_model, MIP_model_transformed, incumbent_solution,
                                                        MIP_copy_vars)
                MIP_model_transformed, sol_MIP_root = copy_sol(MIP_model_2, MIP_model_transformed, incumbent_solution_2,
                                                        MIP_copy_vars)

                transformed_model_name = MIP_model_transformed.getProbName()
                filename = f'{directory_transformedmodel}{transformed_model_name}.cip'
                MIP_model_transformed.writeProblem(filename=filename, trans=False)

                firstsol_filename = f'{directory_firstsol}firstsol-{transformed_model_name}.sol'
                MIP_model_transformed.writeSol(solution=sol_MIP_first, filename=firstsol_filename)

                rootsol_filename = f'{directory_rootsol}rootsol-{transformed_model_name}.sol'
                MIP_model_transformed.writeSol(solution=sol_MIP_root, filename=rootsol_filename)

                model = Model()
                model.readProblem(filename)
                sol = model.readSolFile(rootsol_filename)

                feas = model.checkSol(sol)
                if not feas:
                    print('the root solution of '+ model.getProbName()+ 'is not feasible!')

                model.addSol(sol, False)
                print(model.getSolObjVal(sol))
                instance = ecole.scip.Model.from_pyscipopt(model)
                scipMIP = instance.as_pyscipopt()
                sol2 = scipMIP.getBestSol()
                print(scipMIP.getSolObjVal(sol2))

                # MIP_model_2.resetParams()
                # MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model_2.createCopy(
                #     problemName='rootsol',
                #     origcopy=False)
                # MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model_2, MIP_model_copy2, incumbent_solution_2,
                #                                           MIP_copy_vars2)

                MIP_model.freeProb()
                MIP_model_2.freeProb()
                MIP_model_transformed.freeProb()
                model.freeProb()
                del MIP_model
                del MIP_model_2
                del MIP_model_transformed
                del model

                index_instance += 1

    def generate_instances_miplib2017_binary(self, instance_type='miplib2017_binary', instance_size='-small'):

        directory = './data/generated_instances/' + instance_type + '/' + instance_size + '/'
        directory_transformedmodel = directory + 'transformedmodel' + '/'
        directory_firstsol = directory +'firstsol' + '/'
        directory_rootsol = directory + 'rootsol' + '/'
        pathlib.Path(directory_transformedmodel).mkdir(parents=True, exist_ok=True)
        pathlib.Path(directory_firstsol).mkdir(parents=True, exist_ok=True)
        pathlib.Path(directory_rootsol).mkdir(parents=True, exist_ok=True)

        file_directory = './result/miplib2017/miplib2017_purebinary_solved.txt'
        print(file_directory)
        index_instance = 34 # 0
        with open(file_directory) as fp:
            Lines = fp.readlines()
            i = 1
            for line in Lines:
                if i > 55: #  start from i==56
                    instance_str = line.strip()
                    MIP_model = Loader().load_instance(instance_str)
                    original_name = MIP_model.getProbName()
                    print(original_name)

                    MIP_model.setProbName(instance_type + '-' + str(index_instance))
                    instance_name = MIP_model.getProbName()
                    print('\n')
                    print(instance_name)
                    print('Number of variables', MIP_model.getNVars())
                    print('Number of binary variables', MIP_model.getNBinVars())

                    # initialize MIP
                    MIP_model_2, MIP_2_vars, success = MIP_model.createCopy(
                        problemName='Baseline', origcopy=True)

                    # MIP_model_orig, MIP_vars_orig, success = MIP_model.createCopy(
                    #     problemName='Baseline', origcopy=True)

                    incumbent_mode = 'firstsol'
                    incumbent_mode_2 = 'rootsol'

                    status, feasible, MIP_model, incumbent_solution = self.set_and_optimize_MIP(MIP_model, incumbent_mode)

                    status_2, feasible_2, MIP_model_2, incumbent_solution_2 = self.set_and_optimize_MIP(MIP_model_2,
                                                                                                        incumbent_mode_2)

                    feasible = feasible and feasible_2

                    if (not status == 'optimal') and (not status_2 == 'optimal'):
                        not_optimal = True
                    else:
                        not_optimal = False

                    if not_optimal and feasible:
                        valid = True
                    else:
                        valid = False

                    if valid:

                        MIP_model.resetParams()
                        MIP_model_transformed, MIP_copy_vars, success = MIP_model.createCopy(
                            problemName='transformed', origcopy=False)
                        MIP_model_transformed, sol_MIP_first = copy_sol(MIP_model, MIP_model_transformed, incumbent_solution,
                                                                MIP_copy_vars)
                        MIP_model_transformed, sol_MIP_root = copy_sol(MIP_model_2, MIP_model_transformed, incumbent_solution_2,
                                                                MIP_copy_vars)

                        transformed_model_name = MIP_model_transformed.getProbName()
                        MIP_model_transformed.setProbName(transformed_model_name + '_' + original_name)

                        filename = f'{directory_transformedmodel}{transformed_model_name}.mps'
                        MIP_model_transformed.writeProblem(filename=filename, trans=False)

                        firstsol_filename = f'{directory_firstsol}firstsol-{transformed_model_name}.sol'
                        MIP_model_transformed.writeSol(solution=sol_MIP_first, filename=firstsol_filename)

                        rootsol_filename = f'{directory_rootsol}rootsol-{transformed_model_name}.sol'
                        MIP_model_transformed.writeSol(solution=sol_MIP_root, filename=rootsol_filename)

                        model = Model()
                        model.readProblem(filename)
                        sol = model.readSolFile(rootsol_filename)

                        feas = model.checkSol(sol)
                        if not feas:
                            print('the root solution of '+ model.getProbName()+ 'is not feasible!')

                        model.addSol(sol, False)
                        print(model.getSolObjVal(sol))
                        instance = ecole.scip.Model.from_pyscipopt(model)
                        scipMIP = instance.as_pyscipopt()
                        sol2 = scipMIP.getBestSol()
                        print(scipMIP.getSolObjVal(sol2))

                        # MIP_model_2.resetParams()
                        # MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model_2.createCopy(
                        #     problemName='rootsol',
                        #     origcopy=False)
                        # MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model_2, MIP_model_copy2, incumbent_solution_2,
                        #                                           MIP_copy_vars2)

                        MIP_model.freeProb()
                        MIP_model_2.freeProb()
                        MIP_model_transformed.freeProb()
                        model.freeProb()
                        del MIP_model
                        del MIP_model_2
                        del MIP_model_transformed
                        del model

                        index_instance += 1

                i += 1

    def generate_instances_miplib_39binary(self, instance_type='miplib_39binary', instance_size='-small'):

        directory = './data/generated_instances/' + instance_type + '/' + instance_size + '/'
        directory_transformedmodel = directory + 'transformedmodel' + '/'
        directory_firstsol = directory +'firstsol' + '/'
        directory_rootsol = directory + 'rootsol' + '/'
        pathlib.Path(directory_transformedmodel).mkdir(parents=True, exist_ok=True)
        pathlib.Path(directory_firstsol).mkdir(parents=True, exist_ok=True)
        pathlib.Path(directory_rootsol).mkdir(parents=True, exist_ok=True)

        data = np.load('./result/miplib2017/miplib2017_binary39.npz')
        miplib2017_binary39 = data['miplib2017_binary39']

        index_instance = 0
        for p in range(0, len(miplib2017_binary39)):
            if not p == 26:
                MIP_model = Loader().load_instance(miplib2017_binary39[p] + '.mps.gz')

                original_name = MIP_model.getProbName()
                print(original_name)

                MIP_model.setProbName(instance_type + '-' + str(index_instance))
                instance_name = MIP_model.getProbName()
                print('\n')
                print(instance_name)
                print('Number of variables', MIP_model.getNVars())
                print('Number of binary variables', MIP_model.getNBinVars())

                # initialize MIP
                MIP_model_2, MIP_2_vars, success = MIP_model.createCopy(
                    problemName='Baseline', origcopy=True)

                # MIP_model_orig, MIP_vars_orig, success = MIP_model.createCopy(
                #     problemName='Baseline', origcopy=True)

                incumbent_mode = 'firstsol'
                incumbent_mode_2 = 'rootsol'

                status, feasible, MIP_model, incumbent_solution = self.set_and_optimize_MIP(MIP_model, incumbent_mode)

                status_2, feasible_2, MIP_model_2, incumbent_solution_2 = self.set_and_optimize_MIP(MIP_model_2,
                                                                                                    incumbent_mode_2)

                feasible = feasible and feasible_2

                if (not status == 'optimal') and (not status_2 == 'optimal'):
                    not_optimal = True
                else:
                    not_optimal = False

                if not_optimal and feasible:
                    valid = True
                else:
                    valid = False

                if valid:

                    MIP_model.resetParams()
                    MIP_model_transformed, MIP_copy_vars, success = MIP_model.createCopy(
                        problemName='transformed', origcopy=False)
                    MIP_model_transformed, sol_MIP_first = copy_sol(MIP_model, MIP_model_transformed, incumbent_solution,
                                                            MIP_copy_vars)
                    MIP_model_transformed, sol_MIP_root = copy_sol(MIP_model_2, MIP_model_transformed, incumbent_solution_2,
                                                            MIP_copy_vars)

                    transformed_model_name = MIP_model_transformed.getProbName()
                    MIP_model_transformed.setProbName(transformed_model_name + '_' + original_name)

                    filename = f'{directory_transformedmodel}{transformed_model_name}.cip'
                    MIP_model_transformed.writeProblem(filename=filename, trans=False)

                    firstsol_filename = f'{directory_firstsol}firstsol-{transformed_model_name}.sol'
                    MIP_model_transformed.writeSol(solution=sol_MIP_first, filename=firstsol_filename)

                    rootsol_filename = f'{directory_rootsol}rootsol-{transformed_model_name}.sol'
                    MIP_model_transformed.writeSol(solution=sol_MIP_root, filename=rootsol_filename)

                    model = Model()
                    model.readProblem(filename)
                    sol = model.readSolFile(rootsol_filename)

                    feas = model.checkSol(sol)
                    if not feas:
                        print('the root solution of '+ model.getProbName()+ 'is not feasible!')

                    model.addSol(sol, False)
                    print(model.getSolObjVal(sol))
                    instance = ecole.scip.Model.from_pyscipopt(model)
                    scipMIP = instance.as_pyscipopt()
                    sol2 = scipMIP.getBestSol()
                    print(scipMIP.getSolObjVal(sol2))

                    # MIP_model_2.resetParams()
                    # MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model_2.createCopy(
                    #     problemName='rootsol',
                    #     origcopy=False)
                    # MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model_2, MIP_model_copy2, incumbent_solution_2,
                    #                                           MIP_copy_vars2)

                    MIP_model.freeProb()
                    MIP_model_2.freeProb()
                    MIP_model_transformed.freeProb()
                    model.freeProb()
                    del MIP_model
                    del MIP_model_2
                    del MIP_model_transformed
                    del model

                    index_instance += 1

    def initialize_instances_floorplan(self, instance_directory='./', instance_type='floorplan_gsrc',
                                       instance_size='-small'):

        directory = './data/generated_instances/' + instance_type + '/' + instance_size + '/'
        directory_transformedmodel = directory + 'transformedmodel' + '/'
        directory_firstsol = directory + 'firstsol' + '/'
        directory_rootsol = directory + 'rootsol' + '/'
        pathlib.Path(directory_transformedmodel).mkdir(parents=True, exist_ok=True)
        pathlib.Path(directory_firstsol).mkdir(parents=True, exist_ok=True)
        pathlib.Path(directory_rootsol).mkdir(parents=True, exist_ok=True)

        instance_directory = './data/generated_instances/floorplan_gsrc/original_mip_instances/'
        filename = '*.cip'
        # print(filename)
        sample_files = [str(path) for path in pathlib.Path(instance_directory).glob(filename)]

        index_instance = 3
        for i, instance in enumerate(sample_files):
            print(instance)
            MIP_model = Model()
            MIP_model.readProblem(instance)
            original_name = MIP_model.getProbName()

            MIP_model.setProbName(instance_type + '-' + str(index_instance))
            instance_name = MIP_model.getProbName()
            print('\n')
            print(instance_name)
            print('Number of variables', MIP_model.getNVars())
            print('Number of binary variables', MIP_model.getNBinVars())

            # initialize MIP
            MIP_model_2, MIP_2_vars, success = MIP_model.createCopy(
                problemName='Baseline', origcopy=True)

            # MIP_model_orig, MIP_vars_orig, success = MIP_model.createCopy(
            #     problemName='Baseline', origcopy=True)

            incumbent_mode = 'firstsol'
            incumbent_mode_2 = 'rootsol'

            status, feasible, MIP_model, incumbent_solution = self.set_and_optimize_MIP(MIP_model, incumbent_mode)

            status_2, feasible_2, MIP_model_2, incumbent_solution_2 = self.set_and_optimize_MIP(MIP_model_2,
                                                                                                incumbent_mode_2)

            feasible = feasible and feasible_2

            if (not status == 'optimal') and (not status_2 == 'optimal'):
                not_optimal = True
            else:
                not_optimal = False

            if not_optimal and feasible:
                valid = True
            else:
                valid = False

            if valid:

                MIP_model.resetParams()
                MIP_model_transformed, MIP_copy_vars, success = MIP_model.createCopy(
                    problemName='transformed', origcopy=False)
                MIP_model_transformed, sol_MIP_first = copy_sol(MIP_model, MIP_model_transformed, incumbent_solution,
                                                                MIP_copy_vars)
                MIP_model_transformed, sol_MIP_root = copy_sol(MIP_model_2, MIP_model_transformed, incumbent_solution_2,
                                                               MIP_copy_vars)

                transformed_model_name = MIP_model_transformed.getProbName()
                MIP_model_transformed.setProbName(transformed_model_name + '_' + original_name)

                filename = f'{directory_transformedmodel}{transformed_model_name}.cip'
                MIP_model_transformed.writeProblem(filename=filename, trans=False)

                firstsol_filename = f'{directory_firstsol}firstsol-{transformed_model_name}.sol'
                MIP_model_transformed.writeSol(solution=sol_MIP_first, filename=firstsol_filename)

                rootsol_filename = f'{directory_rootsol}rootsol-{transformed_model_name}.sol'
                MIP_model_transformed.writeSol(solution=sol_MIP_root, filename=rootsol_filename)

                model = Model()
                model.readProblem(filename)
                sol = model.readSolFile(rootsol_filename)

                feas = model.checkSol(sol)
                if not feas:
                    print('the root solution of ' + model.getProbName() + 'is not feasible!')

                model.addSol(sol, False)
                print(model.getSolObjVal(sol))
                instance = ecole.scip.Model.from_pyscipopt(model)
                scipMIP = instance.as_pyscipopt()
                sol2 = scipMIP.getBestSol()
                print(scipMIP.getSolObjVal(sol2))

                # MIP_model_2.resetParams()
                # MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model_2.createCopy(
                #     problemName='rootsol',
                #     origcopy=False)
                # MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model_2, MIP_model_copy2, incumbent_solution_2,
                #                                           MIP_copy_vars2)

                MIP_model.freeProb()
                MIP_model_2.freeProb()
                MIP_model_transformed.freeProb()
                model.freeProb()
                del MIP_model
                del MIP_model_2
                del MIP_model_transformed
                del model

                index_instance += 1










