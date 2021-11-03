import pyscipopt
from pyscipopt import Model
import numpy as np
import sys
from memory_profiler import profile

from models_rl import SimplePolicy
import torch
from utility import imitation_accuracy
import pathlib
import gzip
import pickle
from event import PrimalBoundChangeEventHandler


class LocalBranching:

    def __init__(self, MIP_model, MIP_sol_bar, k=20,  node_time_limit=30, total_time_limit=3600, is_symmetric=True):
        self.MIP_model = MIP_model
        self.MIP_sol_best = self.copy_solution( self.MIP_model, MIP_sol_bar)
        self.MIP_obj_best = self.MIP_model.getSolObjVal(self.MIP_sol_best)
        self.MIP_obj_init = self.MIP_obj_best # initial obj before adapting k
        self.MIP_sol_bar = self.copy_solution( self.MIP_model, MIP_sol_bar)
        self.subMIP_sol_best = self.copy_solution(self.MIP_model, MIP_sol_bar)
        self.MIP_obj_bar = self.MIP_obj_best
        self.n_vars = self.MIP_model.getNVars()
        self.n_binvars = self.MIP_model.getNBinVars()

        self.default_node_time_limit = node_time_limit
        self.default_initial_node_time_limit = node_time_limit
        self.primal_no_improvement_account = 0
        self.total_time_limit = total_time_limit
        self.total_time_available = self.total_time_limit
        self.total_time_expired = 0
        self.div_max = 3
        self.default_k = k
        self.eps = eps = .0000001
        self.t_node = self.default_node_time_limit
        self.k = k
        self.first = False
        self.diversify = False
        self.div = 0
        self.is_symmetric = is_symmetric
        # if not self.is_symmetric:
        #     self.default_k = self.default_k / 2
        self.reset_k_at_2nditeration = False

        self.rightbranch_index = 0

        self.actions = {'reset': 0, 'unchange':1, 'increase': 2, 'decrease':3, 'free':4}

        self.k_stepsize = 1/2
        self.t_stepsize = 3
        self.alpha = 0.01

        self.primal_objs = []
        self.primal_times = []
        self.primal_times.append(self.total_time_limit - self.total_time_available)
        self.primal_objs.append(self.MIP_obj_best)

        self.primalbound_handler = PrimalBoundChangeEventHandler()
        self.MIP_model.includeEventhdlr(self.primalbound_handler, 'primal_bound_update_handler',
                                        'store every new primal bound and its time stamp')

    def create_subMIP(self):

        # self.subMIP_model, subMIP_vars, success = self.MIP_model.createCopy(problemName='subMIPmodel', origcopy=False)
        #
        # # create a primal solution for the copy MIP by copying the solution of original MIP
        # self.subMIP_sol_bar = self.subMIP_model.createSol()
        # self.n_vars = self.MIP_model.getNVars()
        # MIP_vars = self.MIP_model.getVars()
        #
        # for j in range(self.n_vars):
        #     val = self.MIP_model.getSolVal(self.MIP_sol_bar, MIP_vars[j])
        #     self.subMIP_model.setSolVal(self.subMIP_sol_bar, subMIP_vars[j], val)

        self.subMIP_model = self.MIP_model
        self.subMIP_model.resetParams()
        self.subMIP_sol_bar = self.MIP_sol_bar

        # feasible = self.subMIP_model.checkSol(solution=self.subMIP_sol_bar)
        # if feasible:
        #     self.subMIP_model.addSol(subMIP_sol_bar, False)
        #     print("the incumbent solution of subMIP for local branching is added to subMIP")
        # else:
        #     print("Error: the incumbent solution of subMIP for local branching is not feasible!")

        if not self.first == True:
            self.subMIP_ub = self.subMIP_model.getSolObjVal(self.subMIP_sol_bar)
        else:
            self.subMIP_ub = self.subMIP_model.infinity()

        if self.subMIP_ub >=0:
            self.subMIP_model.setObjlimit(0.999 * self.subMIP_ub)
        else:
            self.subMIP_model.setObjlimit(1.001 * self.subMIP_ub)

        self.primalbound_handler.primal_times = []
        self.primalbound_handler.primal_bounds = []

        # print("Initial obj before LB: {}".format(self.subMIP_obj_bar))

    def copy_solution(self, model, solution):
        """create a copy of solution for MIP_model"""
        solution_copy = model.createSol()
        MIP_vars = model.getVars()
        self.n_vars = model.getNVars()

        for j in range(self.n_vars):
            val = model.getSolVal(solution, MIP_vars[j])
            model.setSolVal(solution_copy, MIP_vars[j], val)
        return solution_copy

    def copy_solution_subMIP_to_MIP(self, subMIP_sol, MIP_sol):
        """copy a solution of subMIP to MIP"""
        subMIP_vars = self.subMIP_model.getVars()
        MIP_vars = self.MIP_model.getVars()
        for j in range(self.n_vars):
            val = self.subMIP_model.getSolVal(subMIP_sol, subMIP_vars[j])
            self.MIP_model.setSolVal(MIP_sol, MIP_vars[j], val)

    def left_branch(self, t_node, is_symmetric=True):
        self.create_subMIP()

        if is_symmetric:
            self.add_LBconstraint()
        else:
            self.add_LBconstraintAsym()
        self.subMIP_model.setParam('limits/time', t_node)
        self.subMIP_model.setParam("display/verblevel", 0)
        # for strong diversify(first==True), abort as soon as finding first feasible solution.
        if self.first:
            self.subMIP_model.setParam('limits/solutions', 1)

        self.subMIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.FAST)
        self.subMIP_model.setPresolve (pyscipopt.SCIP_PARAMSETTING.FAST)

        self.subMIP_model.optimize()

    def step_localbranch(self, k_action, t_action, lb_bits, enable_adapt_t=False):

        self.k = self.update_k(k_action, self.k_stepsize)
        self.t_node = self.update_t(t_action, self.t_stepsize)

        # reset k_stand and k at the 2nd iteration if reset option is enable
        if (lb_bits == 2) and self.reset_k_at_2nditeration:
            self.default_k = 20
            if not self.is_symmetric:
                self.default_k = 10
            self.k = self.default_k

            self.diversify = False
            self.first = False

        t_node = np.minimum(self.t_node, self.total_time_available)
        self.left_branch(t_node, is_symmetric=self.is_symmetric)  # execute 1 iteration of lb

        # node_time_limit = self.node_time_limit

        self.primal_no_improvement_account += 1

        t_leftbranch = self.subMIP_model.getSolvingTime()
        self.total_time_available -= t_leftbranch
        subMIP_status = self.subMIP_model.getStatus()

        # update best obj of original MIP before printing
        # subMIP_obj_best = self.subMIP_model.getObjVal()
        # if subMIP_obj_best < self.MIP_obj_best:
        #     self.MIP_obj_best = subMIP_obj_best

        div_pre = self.div
        k_pre = self.k
        MIP_obj_best_pre = self.MIP_obj_best

        state = np.zeros((7, ))

        n_sols_subMIP = self.subMIP_model.getNSols()
        subMIP_obj_best = None

        if n_sols_subMIP > 0:
            subMIP_sol_best = self.subMIP_model.getBestSol()
            subMIP_obj_best = self.subMIP_model.getSolObjVal(subMIP_sol_best)
            if subMIP_obj_best < self.MIP_obj_best:
                primal_bounds = self.primalbound_handler.primal_bounds
                primal_times = self.primalbound_handler.primal_times
                self.primal_no_improvement_account = 0

                for i in range(len(primal_times)):
                    primal_times[i] += self.total_time_expired

                self.primal_objs.extend(primal_bounds)
                self.primal_times.extend(primal_times)

        # case 1
        if subMIP_status == "optimal" or subMIP_status == "bestsollimit":

            subMIP_sol_best = self.subMIP_model.getBestSol()
            self.copy_solution_subMIP_to_MIP(subMIP_sol_best, self.subMIP_sol_best)
            subMIP_obj_best = self.subMIP_model.getSolObjVal(subMIP_sol_best)
            # assert subMIP_obj_best < self.subMIP_ub, "SubMIP is optimal and improved solution of subMIP is expected! But no improved solution found!"

            self.subMIP_model.freeTransform()

            # add the reversed right branch constraint to MIP_model
            if self.is_symmetric == True:
                self.rightbranch_reverse(k=self.k)
            else:
                self.rightbranch_reverse_asym(k=self.k)

            # update best MIP_sol_bar
            self.copy_solution_subMIP_to_MIP(self.subMIP_sol_best, self.MIP_sol_bar)

            self.MIP_obj_bar = subMIP_obj_best
            # update MIP_sol_best and best obj of original MIP
            if subMIP_obj_best < self.MIP_obj_best:
                self.copy_solution_subMIP_to_MIP(self.subMIP_sol_best, self.MIP_sol_best)
                self.MIP_obj_best = subMIP_obj_best

            self.diversify = False
            self.first = False

            state[0:5] = [1, 0, 0, 0, 0]
            # self.k = self.k_standard

        # case 2
        elif subMIP_status == "infeasible" or subMIP_status == "inforunbd":

            self.subMIP_model.freeTransform()
            # add the reversed right branch constraint to MIP_model
            if self.is_symmetric == True:
                self.rightbranch_reverse(k=self.k)
            else:
                self.rightbranch_reverse_asym(k=self.k)

            state[0:5] = [0, 1, 0, 0, 0]

            if self.diversify:
                self.div += 1
                # node_time_limit = self.subMIP_model.infinity()
                self.first = True
                state[4] = 1 # set state[first]=1 when sol not improved for successive 2 iterations
            # self.k += np.ceil(self.k_standard / 2)
            self.diversify = True

        elif subMIP_status == "timelimit" or subMIP_status == "sollimit":
            n_sols = self.subMIP_model.getNSols()
            subMIP_sol_best = self.subMIP_model.getBestSol()
            self.copy_solution_subMIP_to_MIP(subMIP_sol_best, self.subMIP_sol_best)
            subMIP_obj_best = self.subMIP_model.getSolObjVal(subMIP_sol_best)

            # case 3
            if n_sols > 0 and subMIP_obj_best < self.subMIP_ub:

                self.subMIP_model.freeTransform()
                if not self.first:
                    # add the reversed right branch constraint to exclude MIP_sol_bar
                    if self.is_symmetric == True:
                        self.rightbranch_reverse(k=0.0)
                    else:
                        self.rightbranch_reverse_asym(k=0.0)

                # to do: refine best solution

                # assert subMIP_obj_best < self.subMIP_ub, "SubMIP has feasible solutions and an improved solution of subMIP is expected! But no improved solution found!"

                self.copy_solution_subMIP_to_MIP(self.subMIP_sol_best, self.MIP_sol_bar)
                self.MIP_obj_bar = subMIP_obj_best
                # update best obj of original MIP
                if subMIP_obj_best < self.MIP_obj_best:
                    self.copy_solution_subMIP_to_MIP(self.subMIP_sol_best, self.MIP_sol_best)
                    self.MIP_obj_best = subMIP_obj_best

                self.diversify = False
                self.first = False
                # self.k = self.k_standard

                state[0:5] = [0, 0, 1, 0, 0]

            # case 4
            else:

                self.subMIP_model.freeTransform()

                state[0:5] = [0, 0, 0, 1, 0]

                if self.diversify:
                    # to do: add the reversed right branch constraint to exclude MIP_sol_bar
                    if self.is_symmetric == True:
                        self.rightbranch_reverse(k=0.0)
                    else:
                        self.rightbranch_reverse_asym(k=0.0)

                    self.div += 1
                    # node_time_limit = self.subMIP_model.infinity()
                    # self.k += np.ceil(self.k_standard / 2)
                    self.first = True
                    state[4] = 1
                # else:
                #     self.k -= np.ceil(self.k_standard / 2)
                self.diversify = True

        self.subMIP_model.delCons(self.constraint_LB)
        self.subMIP_model.releasePyCons(self.constraint_LB)
        del self.constraint_LB

        # simple t-adpatation algorithm
        if enable_adapt_t:
            if self.primal_no_improvement_account > 0 and self.primal_no_improvement_account % 5 == 0:
                self.default_node_time_limit *= self.t_stepsize

            if self.default_node_time_limit > self.default_initial_node_time_limit and self.primal_no_improvement_account == 0:
                self.default_node_time_limit /= self.t_stepsize


        print('LB round: {:.0f}'.format(lb_bits),
              'Solving time: {:.4f}'.format(self.total_time_limit - self.total_time_available),
              'Best Obj: {:.4f}'.format(self.MIP_obj_best),
              # 'Obj_subMIP: {:.4f}'.format(str(subMIP_obj_best)),
              'n_sols_subMIP: {:.0f}'.format(n_sols_subMIP),
              'K: {:.0f}'.format(k_pre),
              'self.div: {:.0f}'.format(div_pre),
              'LB Status: {}'.format(subMIP_status)
              )


        # avoid negative time reward
        if t_leftbranch > t_node:
            t_leftbranch = t_node

        # calculate rewards reward = alpha * reward_t + (1-alpha * reward_obj)
        obj_norm = np.abs(MIP_obj_best_pre - self.MIP_obj_best)/ np.maximum(np.abs(MIP_obj_best_pre), np.abs(self.MIP_obj_best))
        t_norm = 1 - t_leftbranch / t_node
        state[5:7] = [t_norm, obj_norm]

        # # calculate rewards reward = alpha * reward_t + (1-alpha * reward_obj)
        # reward_obj = obj_norm
        # if obj_norm > 0:
        #     if t_leftbranch >= t_node:
        #         reward_t = -1
        #     else:
        #         reward_t = t_norm
        # else:
        #     reward_t = 0
        # reward = self.alpha * reward_t + (1-self.alpha)*reward_obj

        # calculate reward: reward = obj_improve_bit * t_leftbranch_bit

        # # reward 1
        # obj_best_local = self.MIP_obj_best # reward option 1: use the best obj after running that lb iteration
        # obj_improve = np.abs(self.MIP_obj_init - obj_best_local) / np.abs(self.MIP_obj_init)
        # reward = obj_improve * t_leftbranch

        # # reward 2
        # obj_best_local = MIP_obj_best_pre # reward option 2: use the best obj before running that lb iteration
        # obj_improve = np.abs(self.MIP_obj_init - obj_best_local) / np.abs(self.MIP_obj_init)
        # reward = obj_improve * t_leftbranch

        # reward 3
        obj_improve_local = np.abs(MIP_obj_best_pre - self.MIP_obj_best) / np.abs(self.MIP_obj_init)
        reward = obj_improve_local * self.total_time_available

        done = (self.total_time_available <= 0) or (self.k >= self.n_binvars)
        info = None

        self.total_time_expired += t_leftbranch
        return state, reward, done, info

    def solve_rightbranch(self):
        """
        solve the MIP of right branch with the time available.
        :return:
        """
        self.MIP_model.addSol(self.MIP_sol_best)

        self.primalbound_handler.primal_bounds = []
        self.primalbound_handler.primal_times = []
        if self.total_time_available > 0:
            self.MIP_model.setObjlimit(self.MIP_obj_best - self.eps)
            self.MIP_model.setParam('limits/time', self.total_time_available)
            self.MIP_model.optimize()

            best_obj = self.MIP_model.getObjVal()
            if best_obj < self.MIP_obj_best:
                self.MIP_obj_best = best_obj

                if self.subMIP_model.getNSols() > 0:
                    subMIP_sol_best = self.subMIP_model.getBestSol()
                    subMIP_obj_best = self.subMIP_model.getSolObjVal(subMIP_sol_best)


                    primal_bounds = self.primalbound_handler.primal_bounds
                    primal_times = self.primalbound_handler.primal_times

                    for i in range(len(primal_times)):
                        primal_times[i] += self.total_time_expired

                    self.primal_objs.extend(primal_bounds)
                    self.primal_times.extend(primal_times)

            self.total_time_available -= self.MIP_model.getSolvingTime()
            self.total_time_expired += self.MIP_model.getSolvingTime()

    def policy_vanilla(self, state):

        lb_status = state[0:4].argmax()
        if lb_status == 0: #[1, 0, 0, 0]
            k_action = self.actions['reset']
            t_action = self.actions['reset']
        elif lb_status == 1: # state[0:4] == [0, 1, 0, 0]:
            if state[4] == 0:
                k_action = self.actions['increase']
                t_action = self.actions['reset']
            elif state[4] == 1:
                k_action = self.actions['increase']
                t_action = self.actions['free']
        elif lb_status == 2: # state[0:4] == [0, 0, 1, 0]:
            k_action = self.actions['reset']
            t_action = self.actions['reset']
        elif lb_status == 3:# state[0:4] == [0, 0, 0, 1]:
            if state[4] == 0:
                k_action = self.actions['decrease']
                t_action = self.actions['reset']
            elif state[4] == 1:
                k_action = self.actions['increase']
                t_action = self.actions['free']

        return k_action, t_action

    def update_k(self, action, k_stepsize):
        switcher = {
            self.actions['reset']: self.default_k,
            self.actions['unchange']: self.k,
            self.actions['decrease']: np.ceil(self.k - k_stepsize * self.k),
            self.actions['increase']: np.ceil(self.k + k_stepsize * self.k)
        }
        return switcher.get(action, 'Error: Invilid k action!')

    def update_t(self, action, t_stepsize):
        switcher = {
            self.actions['reset']: self.default_node_time_limit,
            self.actions['unchange']: self.t_node,
            self.actions['decrease']: self.t_node - t_stepsize * self.t_node,
            self.actions['increase']: self.t_node + t_stepsize * self.t_node,
            self.actions['free']: self.MIP_model.infinity()
        }
        return switcher.get(action, 'Error: Invilid k action!')

    def mdp_localbranch(self, is_symmetric=True, reset_k_at_2nditeration=False, policy=None, optimizer=None, criterion=None, device=None, samples_dir=None):

        # self.total_time_limit = total_time_limit
        self.total_time_available = self.total_time_limit
        self.first = False
        self.diversify = False
        self.t_node = self.default_node_time_limit
        self.div = 0
        self.is_symmetric = is_symmetric
        self.reset_k_at_2nditeration = reset_k_at_2nditeration
        lb_bits = 0
        t_list = []
        obj_list = []
        lb_bits_list = []

        lb_bits_list.append(lb_bits)
        t_list.append(self.total_time_limit - self.total_time_available)
        obj_list.append(self.MIP_obj_best)

        accu_instance = 0
        loss_instance = 0
        k_action = self.actions['unchange']
        t_action = self.actions['unchange']
        done = (self.total_time_available <= 0) or (self.k >= self.n_binvars)

        while not done:  # and self.div < self.div_max
            lb_bits += 1

            # execute one iteration of LB and get the state and rewards
            state, rewards, done, _ = self.step_localbranch(k_action=k_action, t_action=t_action, lb_bits=lb_bits)

            # k_vanilla, t_action = self.policy_vanilla(state)
            # k_action = k_vanilla

            if policy is not None:
                # state_torch = torch.FloatTensor(state).view(1, -1)
                # k_vanilla_torch = torch.LongTensor(np.array(k_vanilla).reshape(-1))
                # if device is not None:
                #     state_torch.to(device)
                #     k_vanilla_torch.to(device)


                # k_pred = policy(state_torch)

                # loss = criterion(k_pred, k_vanilla_torch)
                # accu = imitation_accuracy(k_pred, k_vanilla_torch)

                # # for online learning, update policy
                # if optimizer is not None:
                #     optimizer.zero_grad()
                #     loss.backward()
                #     optimizer.step()

                # loss_instance += loss.item()
                # accu_instance += accu.item()

                # k_action = k_pred.argmax(1, keepdim=False).item()

                k_action = policy.select_action(state)
            else:
                k_vanilla, t_action = self.policy_vanilla(state)
                k_action = k_vanilla

                # data_sample = [state, k_vanilla]
                #
                # filename = f'{samples_dir}imitation_{self.MIP_model.getProbName()}_{lb_bits}.pkl'
                #
                # with gzip.open(filename, 'wb') as f:
                #     pickle.dump(data_sample, f)

            lb_bits_list.append(lb_bits)
            t_list.append(self.total_time_limit - self.total_time_available)
            obj_list.append(self.MIP_obj_best)

        print(
            'K_final: {:.0f}'.format(self.k),
            'div_final: {:.0f}'.format(self.div)
        )

        self.solve_rightbranch()
        t_list.append(self.total_time_limit - self.total_time_available)
        obj_list.append(self.MIP_obj_best)

        status = self.MIP_model.getStatus()
        # if status == "optimal" or status == "bestsollimit":
        #     self.MIP_obj_best = self.MIP_model.getObjVal()

        elapsed_time = self.total_time_limit - self.total_time_available

        lb_bits_list = np.array(lb_bits_list).reshape(-1)
        times_list = np.array(t_list).reshape(-1)
        objs_list = np.array(obj_list).reshape(-1)

        del self.subMIP_sol_best
        del self.MIP_sol_bar
        del self.MIP_sol_best

        loss_instance = loss_instance / lb_bits
        accu_instance = accu_instance / lb_bits

        return status, self.MIP_obj_best, elapsed_time, lb_bits_list, times_list, objs_list, loss_instance, accu_instance

    def search_localbranch(self, is_symmetric=True, reset_k_at_2nditeration=False):
        # self.total_time_limit = total_time_limit
        self.total_time_available = self.total_time_limit
        self.first = False
        self.diversify = False
        node_time_limit = self.default_node_time_limit
        self.div = 0
        self.is_symmetric = is_symmetric
        lb_bits = 0
        t_list = []
        obj_list = []
        lb_bits_list = []

        lb_bits_list.append(lb_bits)
        t_list.append(self.total_time_limit - self.total_time_available)
        obj_list.append(self.MIP_obj_best)

        while self.total_time_available > 0 and self.k < self.n_binvars: # and self.div < self.div_max
            lb_bits += 1

            # reset k_stand and k at the 2nd iteration if reset option is enable
            if lb_bits == 2 and reset_k_at_2nditeration == True:
                self.default_k = 20
                if not self.is_symmetric:
                    self.default_k = 10
                self.k = self.default_k

                self.diversify = False
                self.first = False

            node_time_limit = np.minimum(node_time_limit ,self.total_time_available)
            self.left_branch(node_time_limit, is_symmetric=self.is_symmetric) # execute 1 iteration of lb

            node_time_limit = self.default_node_time_limit
            self.total_time_available -= self.subMIP_model.getSolvingTime()
            subMIP_status = self.subMIP_model.getStatus()

            # update best obj of original MIP before printing
            # subMIP_obj_best = self.subMIP_model.getObjVal()
            # if subMIP_obj_best < self.MIP_obj_best:
            #     self.MIP_obj_best = subMIP_obj_best

            div_pre = self.div
            k_pre = self.k

            # case 1
            if subMIP_status == "optimal" or subMIP_status == "bestsollimit":

                subMIP_sol_best = self.subMIP_model.getBestSol()
                self.copy_solution_subMIP_to_MIP(subMIP_sol_best, self.subMIP_sol_best)
                subMIP_obj_best = self.subMIP_model.getSolObjVal(subMIP_sol_best)
                assert subMIP_obj_best < self.subMIP_ub, "SubMIP is optimal and improved solution of subMIP is expected! But no improved solution found!"

                self.subMIP_model.freeTransform()

                # add the reversed right branch constraint to MIP_model
                if self.is_symmetric == True:
                    self.rightbranch_reverse(k=self.k)
                else:
                    self.rightbranch_reverse_asym(k=self.k)

                # update best obj of original MIP
                if subMIP_obj_best < self.MIP_obj_best:
                    self.MIP_obj_best = subMIP_obj_best

                # update best MIP_sol_bar
                self.copy_solution_subMIP_to_MIP(self.subMIP_sol_best, self.MIP_sol_bar)

                self.MIP_obj_bar = subMIP_obj_best
                # update MIP_sol_best
                if subMIP_obj_best < self.MIP_obj_best:
                    self.copy_solution_subMIP_to_MIP(self.subMIP_sol_best, self.MIP_sol_best)
                    self.MIP_obj_best = subMIP_obj_best

                self.diversify = False
                self.first = False
                self.k = self.default_k

            # case 2
            elif subMIP_status == "infeasible" or subMIP_status == "inforunbd":

                self.subMIP_model.freeTransform()
                # add the reversed right branch constraint to MIP_model
                if self.is_symmetric == True:
                    self.rightbranch_reverse(k=self.k)
                else:
                    self.rightbranch_reverse_asym(k=self.k)

                if self.diversify:
                    self.div += 1
                    node_time_limit = self.subMIP_model.infinity()
                    self.first =True
                self.k += np.ceil(self.default_k / 2)
                self.diversify = True

            elif subMIP_status == "timelimit" or subMIP_status == "sollimit":
                n_sols = self.subMIP_model.getNSols()
                subMIP_sol_best = self.subMIP_model.getBestSol()
                self.copy_solution_subMIP_to_MIP(subMIP_sol_best, self.subMIP_sol_best)
                subMIP_obj_best = self.subMIP_model.getSolObjVal(subMIP_sol_best)

                # case 3
                if n_sols >0 and subMIP_obj_best < self.subMIP_ub:

                    self.subMIP_model.freeTransform()
                    if not self.first:
                        # add the reversed right branch constraint to exclude MIP_sol_bar
                        if self.is_symmetric == True:
                            self.rightbranch_reverse(k=0.0)
                        else:
                            self.rightbranch_reverse_asym(k=0.0)

                    # to do: refine best solution

                    # assert subMIP_obj_best < self.subMIP_ub, "SubMIP has feasible solutions and an improved solution of subMIP is expected! But no improved solution found!"

                    # update best obj of original MIP
                    if subMIP_obj_best < self.MIP_obj_best:
                        self.MIP_obj_best = subMIP_obj_best

                    self.copy_solution_subMIP_to_MIP(self.subMIP_sol_best, self.MIP_sol_bar)
                    self.MIP_obj_bar = subMIP_obj_best
                    if subMIP_obj_best < self.MIP_obj_best:
                        self.copy_solution_subMIP_to_MIP(self.subMIP_sol_best, self.MIP_sol_best)
                        self.MIP_obj_best = subMIP_obj_best


                    self.diversify = False
                    self.first = False
                    self.k = self.default_k

                # case 4
                else:

                    self.subMIP_model.freeTransform()
                    if self.diversify:
                        # to do: add the reversed right branch constraint to exclude MIP_sol_bar
                        if self.is_symmetric == True:
                            self.rightbranch_reverse(k=0.0)
                        else:
                            self.rightbranch_reverse_asym(k=0.0)

                        self.div += 1
                        node_time_limit = self.subMIP_model.infinity()
                        self.k += np.ceil(self.default_k / 2)
                        self.first = True
                    else:
                        self.k -= np.ceil(self.default_k / 2)
                    self.diversify = True


            print('LB round: {:.0f}'.format(lb_bits),
                  'Solving time: {:.4f}'.format(self.total_time_limit - self.total_time_available),
                  'Best Obj: {:.4f}'.format(self.MIP_obj_best),
                  # 'Best Obj subMIP: {:.4f}'.format(self.subMIP_model.getObjVal()),
                  'K: {:.0f}'.format(k_pre),
                  'self.div: {:.0f}'.format(div_pre),
                  'LB Status: {}'.format(subMIP_status)
                  )


            lb_bits_list.append(lb_bits)
            t_list.append(self.total_time_limit - self.total_time_available)
            obj_list.append(self.MIP_obj_best)

            self.subMIP_model.delCons(self.constraint_LB)
            self.subMIP_model.releasePyCons(self.constraint_LB)
            del self.constraint_LB

            # self.subMIP_model.freeSol(self.subMIP_sol_bar)
            # self.subMIP_model.freeProb()
            # del self.subMIP_sol_bar
            # del self.subMIP_model

        print(
              'K_final: {:.0f}'.format(self.k),
              'div_final: {:.0f}'.format(self.div)
              )

        self.MIP_model.setObjlimit(self.MIP_obj_best - self.eps)
        self.MIP_model.addSol(self.MIP_sol_best)
        if self.total_time_available > 0:
            self.MIP_model.setParam('limits/time', self.total_time_available)
            self.MIP_model.optimize()

        status = self.MIP_model.getStatus()
        if status == "optimal" or status == "bestsollimit":
            self.MIP_obj_best = self.MIP_model.getObjVal()

        self.total_time_available -= self.MIP_model.getSolvingTime()
        elapsed_time = self.total_time_limit - self.total_time_available

        t_list.append(self.total_time_limit - self.total_time_available)
        obj_list.append(self.MIP_obj_best)

        lb_bits = np.array(lb_bits_list).reshape(-1)
        times = np.array(t_list).reshape(-1)
        objs = np.array(obj_list).reshape(-1)

        del lb_bits_list
        del t_list
        del obj_list

        del self.subMIP_sol_best
        del self.MIP_sol_bar
        del self.MIP_sol_best

        return status, self.MIP_obj_best, elapsed_time, lb_bits, times, objs

    def rightbranch_reverse(self, k):
        """
        add the reversed right branch constraint to MIP_model
        :param k:
        :return:
        """
        vars = self.MIP_model.getVars()
        n_binvars = self.MIP_model.getNBinVars()

        rhs = self.MIP_model.infinity()
        lhs = k + 1

        cons_vars = np.empty(n_binvars, dtype=np.object)
        cons_vals = np.empty(n_binvars)

        # compute coefficient for reversed LB constraint
        for i in range(0, n_binvars):
            val = self.MIP_model.getSolVal(self.MIP_sol_bar, vars[i])
            assert self.MIP_model.isFeasIntegral(val), "Error: Solution passed to LB is not integral!"

            if self.MIP_model.isFeasEQ(val, 1.0):
                cons_vals[i] = -1.0
                lhs -=1.0
                rhs -=1.0
            else:
                cons_vals[i] = 1.0
            cons_vars[i] = vars[i]
            assert cons_vars[i].vtype() == "BINARY", "Error: local branching constraint uses a non-binary variable!"

        # create right local branch constraints
        constraint_rightbranch = self.MIP_model.createConsBasicLinear(self.MIP_model.getProbName() + '_rightbranching_'+ str(self.rightbranch_index), n_binvars,
                                                                      cons_vars, cons_vals, lhs, rhs)
        self.MIP_model.addPyCons(constraint_rightbranch)
        self.MIP_model.releasePyCons(constraint_rightbranch)

        del constraint_rightbranch
        del vars
        del cons_vars
        del cons_vals

    def rightbranch_reverse_asym(self, k):
        """
        add the reversed right branch constraint to MIP_model
        :param k:
        :return:
        """
        vars = self.MIP_model.getVars()
        n_binvars = self.MIP_model.getNBinVars()

        rhs = self.MIP_model.infinity()
        lhs = k + 1

        cons_vars = np.empty(n_binvars, dtype=np.object)
        cons_vals = np.empty(n_binvars)

        # compute coefficient for reversed LB constraint
        for i in range(0, n_binvars):
            val = self.MIP_model.getSolVal(self.MIP_sol_bar, vars[i])
            assert self.MIP_model.isFeasIntegral(val), "Error: Solution passed to LB is not integral!"

            if self.MIP_model.isFeasEQ(val, 1.0):
                cons_vals[i] = -1.0
                lhs -=1.0
                rhs -=1.0
            else:
                cons_vals[i] = 0.0
            cons_vars[i] = vars[i]
            assert cons_vars[i].vtype() == "BINARY", "Error: local branching constraint uses a non-binary variable!"

        # create right local branch constraints
        constraint_rightbranch = self.MIP_model.createConsBasicLinear(self.MIP_model.getProbName() + '_rightbranching_'+ str(self.rightbranch_index), n_binvars,
                                                                      cons_vars, cons_vals, lhs, rhs)
        self.MIP_model.addPyCons(constraint_rightbranch)
        self.MIP_model.releasePyCons(constraint_rightbranch)

        del constraint_rightbranch
        del vars
        del cons_vars
        del cons_vals

    def add_LBconstraint(self):
        """symmetric local branching constraint over all binary variables"""

        vars = self.subMIP_model.getVars()
        n_binvars = self.subMIP_model.getNBinVars()

        lhs = 0
        rhs = self.k
        cons_vars = np.empty(n_binvars, dtype=np.object)
        cons_vals = np.empty(n_binvars)

        # compute coefficients for LB constraint
        for i in range(0, n_binvars):
            val = self.subMIP_model.getSolVal(self.subMIP_sol_bar, vars[i])
            assert self.subMIP_model.isFeasIntegral(val), "Error: Solution passed to LB is not integral!"

            if self.subMIP_model.isFeasEQ(val, 1.0):
                cons_vals[i] = -1.0
                lhs -= 1.0
                rhs -= 1.0
            else:
                cons_vals[i] = 1.0
            cons_vars[i] = vars[i]
            assert cons_vars[i].vtype() == "BINARY", "Error: local branching constraint uses a non-binary variable!"

        # create and add LB constraint to mip_model
        self.constraint_LB = self.subMIP_model.createConsBasicLinear(self.subMIP_model.getProbName() + "_localbranching", n_binvars,
                                                                cons_vars, cons_vals, lhs, rhs)
        self.subMIP_model.addPyCons(self.constraint_LB)
        # self.subMIP_model.releasePyCons(self.constraint_LB)

        del vars
        del cons_vars
        del cons_vals

        # for j in range(0, n_binvars):  # release cons_vars variables after creating a constraint
        #     self.subMIP_model.releaseVar(cons_vars[j])


    def add_LBconstraintAsym(self):
        """symmetric local branching constraint over all binary variables"""

        vars = self.subMIP_model.getVars()
        n_binvars = self.subMIP_model.getNBinVars()

        lhs = 0
        rhs = self.k
        cons_vars = np.empty(n_binvars, dtype=np.object)
        cons_vals = np.empty(n_binvars)

        # compute coefficients for LB constraint
        for i in range(0, n_binvars):
            val = self.subMIP_model.getSolVal(self.subMIP_sol_bar, vars[i])
            assert self.subMIP_model.isFeasIntegral(val), "Error: Solution passed to LB is not integral!"

            if self.subMIP_model.isFeasEQ(val, 1.0):
                cons_vals[i] = -1.0
                lhs -= 1.0
                rhs -= 1.0
            else:
                cons_vals[i] = 0.0
            cons_vars[i] = vars[i]
            assert cons_vars[i].vtype() == "BINARY", "Error: local branching constraint uses a non-binary variable!"

        # create and add LB constraint to mip_model
        self.constraint_LB = self.subMIP_model.createConsBasicLinear(self.subMIP_model.getProbName() + "_localbranching", n_binvars,
                                                                cons_vars, cons_vals, lhs, rhs)
        self.subMIP_model.addPyCons(self.constraint_LB)
        # self.subMIP_model.releasePyCons(self.constraint_LB)

        del vars
        del cons_vars
        del cons_vals
        # for j in range(0, n_binvars):  # release cons_vars variables after creating a constraint
        #     self.subMIP_model.releaseVar(cons_vars[j])


def addLBConstraint(mip_model, mip_sol, neighborhoodsize):
    """symmetric local branching constraint over all binary variables"""
    vars = mip_model.getVars()
    n_binvars = mip_model.getNBinVars()

    lhs = 0
    rhs = neighborhoodsize
    cons_vars = np.empty(n_binvars, dtype=np.object)
    cons_vals = np.empty(n_binvars)

    # compute coefficients for LB constraint
    for i in range(0, n_binvars):
        val = mip_model.getSolVal(mip_sol, vars[i])
        assert mip_model.isFeasIntegral(val), "Error: Solution passed to LB is not integral!"

        if mip_model.isFeasEQ(val, 1.0):
            cons_vals[i] = -1.0
            lhs -= 1.0
            rhs -= 1.0
        else:
            cons_vals[i] = 1.0
        cons_vars[i] = vars[i]
        assert cons_vars[i].vtype() == "BINARY", "Error: local branching constraint uses a non-binary variable!"

    # create and add LB constraint to mip_model
    constraint_LB = mip_model.createConsBasicLinear(mip_model.getProbName()+"_localbranching", n_binvars, cons_vars, cons_vals, lhs, rhs)
    mip_model.addPyCons(constraint_LB)
    # mip_model.releasePyCons(constraint_LB)
    # for j in range(0, n_binvars):  # release cons_vars variables after creating a constraint
    #     mip_model.releaseVar(cons_vars[j])
    del vars
    del cons_vars
    del cons_vals

    return mip_model, constraint_LB


def addLBConstraintAsymmetric(mip_model, mip_sol, neighborhoodsize):
    """asymmetric local branching variables over the support of binary variables"""
    vars = mip_model.getVars()
    n_binvars = mip_model.getNBinVars()

    lhs = 0
    rhs = neighborhoodsize
    cons_vars = np.empty(n_binvars, dtype=np.object)
    cons_vals = np.empty(n_binvars)

    # compute coefficients for LB constraint
    for i in range(0, n_binvars):
        val = mip_model.getSolVal(mip_sol, vars[i])
        assert mip_model.isFeasIntegral(val), "Error: Solution passed to LB is not integral!"

        if mip_model.isFeasEQ(val, 1.0):
            cons_vals[i] = -1.0
            lhs -= 1.0
            rhs -= 1.0
        else:
            cons_vals[i] = 0.0
        cons_vars[i] = vars[i]
        assert cons_vars[i].vtype() == "BINARY", "Error: local branching constraint uses a non-binary variable!"

    # create and add LB constraint to mip_model
    constraint_LB = mip_model.createConsBasicLinear(mip_model.getProbName()+"_localbranching", n_binvars, cons_vars, cons_vals, lhs, rhs)
    mip_model.addPyCons(constraint_LB)
    # mip_model.releasePyCons(constraint_LB)
    # for j in range(0, n_binvars):  # release cons_vars variables after creating a constraint
    #     mip_model.releaseVar(cons_vars[j])

    del vars
    del cons_vars
    del cons_vals

    return mip_model, constraint_LB


def addLBConstraintAsymJustslackvars(mip_model, mip_sol, neighborhoodsize, indexlist_slackvars):
    """asymmetric local branching constraints over the support of slack variables"""

    vars = mip_model.getVars()
    n_slackvars = len(indexlist_slackvars)

    lhs = 0
    rhs = neighborhoodsize
    cons_vars = np.empty(n_slackvars, dtype=np.object)
    cons_vals = np.empty(n_slackvars)

    # compute coefficients for LB constraint
    for i in range(0, n_slackvars):
        val = mip_model.getSolVal(mip_sol, vars[indexlist_slackvars[i]])
        assert mip_model.isFeasIntegral(val), "Error: Solution passed to LB is not integral!"

        if mip_model.isFeasEQ(val, 1.0):
            cons_vals[i] = -1.0
            lhs -= 1.0
            rhs -= 1.0
        else:
            cons_vals[i] = 0.0
        cons_vars[i] = vars[indexlist_slackvars[i]]
        assert cons_vars[i].vtype() == "BINARY", "Error: local branching constraint uses a non-binary variable!"

    # create and add LB constraint to mip_model
    constraint_LB = mip_model.createConsBasicLinear(mip_model.getProbName()+"_localbranching", n_slackvars, cons_vars, cons_vals, lhs, rhs)
    mip_model.addPyCons(constraint_LB)
    # for j in range(0, n_binvars):  # release cons_vars variables after creating a constraint
    #     mip_model.releaseVar(cons_vars[j])
    del constraint_LB
    del vars
    del cons_vars
    del cons_vals

    return mip_model
