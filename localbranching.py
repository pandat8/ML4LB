"""The local branching (LB) heuristic algorithm implemented in Python.

This file implements the basic local branching heuristic algorithm of
Fischetti and Lodi. It calls SCIP as the off-the-shelf MIP solver for solving
the local branching sub-problems. It also includes the necessary methods for
extending the LB algorithm with ML (states, actions and rewards of the LB
Markov decision process used by the RL policies).
"""

import pyscipopt
import numpy as np

from event import PrimalBoundChangeEventHandler


class LocalBranching:
    """One local branching search over a MIP instance.

    The search state consists of the current incumbent (MIP_sol_bar), the
    best solution found so far (MIP_sol_best), the neighborhood size k and
    the node time limit t_node. Each call of step_localbranch() executes one
    LB iteration (solving the left branch sub-MIP) and updates k and t_node
    according to the given actions.
    """

    def __init__(self, MIP_model, MIP_sol_bar, MIP_vars=None, k=20,  node_time_limit=10, total_time_limit=3600, is_symmetric=True, is_heuristic=False):
        self.MIP_model = MIP_model
        self.MIP_vars = MIP_vars
        self.MIP_sol_best = self.copy_solution( self.MIP_model, MIP_sol_bar)
        self.MIP_obj_best = self.MIP_model.getSolObjVal(self.MIP_sol_best)
        self.MIP_obj_init = self.MIP_obj_best # initial obj before adapting k
        self.MIP_sol_bar = self.copy_solution( self.MIP_model, MIP_sol_bar)
        self.subMIP_sol_best = self.copy_solution(self.MIP_model, MIP_sol_bar)
        self.MIP_obj_bar = self.MIP_obj_best
        self.n_vars = self.MIP_model.getNVars()
        self.n_binvars = self.MIP_model.getNBinVars()

        # time management of the LB search
        self.default_node_time_limit = node_time_limit
        self.default_initial_node_time_limit = node_time_limit
        self.primal_no_improvement_account = 0
        self.total_time_limit = total_time_limit
        self.total_time_available = self.total_time_limit
        self.total_time_expired = 0

        # neighborhood size k and node time limit t_node, with their bounds
        self.div_max = 2               # max number of strong diversifications
        self.default_k = k
        self.eps = .0000001            # objective-limit tolerance
        self.t_node = self.default_node_time_limit
        self.t_node_lowerbound = 0.1
        self.t_node_upperbound = 20
        self.k = k
        self.k_lowerbound = 10
        self.first = False             # strong diversification flag
        self.diversify = False         # (weak) diversification flag
        self.div = 0                   # number of diversifications performed
        self.is_symmetric = is_symmetric
        self.reset_k_at_2nditeration = False

        self.rightbranch_index = 0

        # actions of the k- and t-policies
        self.actions = {'reset': 0, 'unchange': 1, 'increase': 2, 'decrease': 3, 'free': 4}

        self.k_stepsize = 1/2          # multiplicative step for k updates
        self.t_stepsize = 2            # multiplicative step for t updates
        self.t_default_stepsize = 3    # step for the hand-crafted t adaptation
        self.alpha = 0.01              # weight of the time reward

        self.primal_objs = []
        self.primal_times = []
        self.primal_times.append(self.total_time_limit - self.total_time_available)
        self.primal_objs.append(self.MIP_obj_best)

        self.primalbound_handler = PrimalBoundChangeEventHandler()
        self.MIP_model.includeEventhdlr(self.primalbound_handler, 'primal_bound_update_handler',
                                        'store every new primal bound and its time stamp')

        self.is_heuristic = is_heuristic

    def create_subMIP(self):
        """Prepare the LB sub-MIP: reset parameters and set the objective limit.

        The sub-MIP shares the model with the original MIP; an objective
        limit slightly better than the current incumbent objective enforces
        that only improving solutions are accepted (unless a strong
        diversification is performed, in which case no limit is set).
        """
        self.subMIP_model = self.MIP_model
        self.subMIP_model.resetParams()
        self.subMIP_sol_bar = self.MIP_sol_bar

        if not self.first:
            self.subMIP_ub = self.subMIP_model.getSolObjVal(self.subMIP_sol_bar)
        else:
            self.subMIP_ub = self.subMIP_model.infinity()

        if self.subMIP_ub >= 0:
            self.subMIP_model.setObjlimit(0.999 * self.subMIP_ub)
        else:
            self.subMIP_model.setObjlimit(1.001 * self.subMIP_ub)

        self.primalbound_handler.primal_times = []
        self.primalbound_handler.primal_bounds = []

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
        """Solve the left-branch sub-MIP (incumbent neighborhood of size k).

        :param t_node: time limit for solving the sub-MIP.
        :param is_symmetric: use the symmetric or asymmetric LB constraint.
        """
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
        self.subMIP_model.setPresolve(pyscipopt.SCIP_PARAMSETTING.FAST)

        self.subMIP_model.optimize()

    def step_localbranch(self, k_action, t_action, lb_bits, enable_adapt_t=False):
        """Execute one iteration (MDP step) of the local branching search.

        Applies the k and t actions, solves the left-branch sub-MIP, updates
        the incumbent and diversification flags according to the sub-MIP
        status, and computes the state and reward signals for the RL policies.

        :param k_action: action for the neighborhood size k (see self.actions).
        :param t_action: action for the node time limit t_node.
        :param lb_bits: index of the current LB iteration (1-based).
        :param enable_adapt_t: enable the hand-crafted t adaptation rule.
        :return: (state, reward_k, reward_t, done, success), where state is
            the 7-dimensional LB state (4 status bits, strong-diversification
            bit, normalized time and objective improvement), done flags the
            end of the search, and success flags an improved best solution.
        """
        success = False

        self.k = self.update_k(k_action, self.k_stepsize)
        self.t_node = self.update_t(t_action, self.t_stepsize)

        # reset default_k and k at the 2nd iteration if the reset option is enabled
        if (lb_bits == 2) and self.reset_k_at_2nditeration:
            self.default_k = 20
            if not self.is_symmetric:
                self.default_k = 10
            self.k = self.default_k

            self.diversify = False
            self.first = False

        # control k and t_node within the bound limits.
        if self.k < self.k_lowerbound:
            self.k = self.k_lowerbound
        if self.t_node < self.t_node_lowerbound:
            self.t_node = self.t_node_lowerbound
        elif self.t_node > self.t_node_upperbound:
            self.t_node = self.t_node_upperbound

        t_node = np.minimum(self.t_node, self.total_time_available)
        self.left_branch(t_node, is_symmetric=self.is_symmetric)  # solve the LB sub-MIP
        n_nodes_subMIP = self.subMIP_model.getNNodes()

        self.primal_no_improvement_account += 1

        t_leftbranch = self.subMIP_model.getSolvingTime()
        self.total_time_available -= t_leftbranch
        subMIP_status = self.subMIP_model.getStatus()

        # snapshot of the search state before processing this iteration
        div_pre = self.div
        k_pre = self.k
        t_pre = self.t_node
        MIP_obj_best_pre = self.MIP_obj_best

        state = np.zeros((7, ))

        n_sols_subMIP = self.subMIP_model.getNSols()
        subMIP_obj_best = None

        # case 1: sub-MIP solved to optimality -> improved incumbent found
        if subMIP_status == "optimal" or subMIP_status == "bestsollimit":

            subMIP_sol_best = self.subMIP_model.getBestSol()
            self.copy_solution_subMIP_to_MIP(subMIP_sol_best, self.subMIP_sol_best)
            subMIP_obj_best = self.subMIP_model.getSolObjVal(subMIP_sol_best)

            self.subMIP_model.freeTransform()

            # add the reversed right branch constraint to MIP_model
            if not self.is_heuristic:
                if self.is_symmetric == True:
                    self.rightbranch_reverse(k=self.k)
                else:
                    self.rightbranch_reverse_asym(k=self.k)

            # update best MIP_sol_bar
            self.copy_solution_subMIP_to_MIP(self.subMIP_sol_best, self.MIP_sol_bar)

            self.MIP_obj_bar = subMIP_obj_best

            self.diversify = False
            self.first = False

            state[0:5] = [1, 0, 0, 0, 0]

        # case 2: sub-MIP proven infeasible -> no improving solution in the
        # neighborhood; diversify
        elif subMIP_status == "infeasible" or subMIP_status == "inforunbd":

            self.subMIP_model.freeTransform()
            # add the reversed right branch constraint to MIP_model
            if not self.is_heuristic:
                if self.is_symmetric == True:
                    self.rightbranch_reverse(k=self.k)
                else:
                    self.rightbranch_reverse_asym(k=self.k)

            state[0:5] = [0, 1, 0, 0, 0]

            if self.diversify:
                self.div += 1
                self.first = True
                state[4] = 1 # set state[first]=1 when sol not improved for successive 2 iterations
            self.diversify = True

        elif subMIP_status == "timelimit" or subMIP_status == "sollimit":
            n_sols = self.subMIP_model.getNSols()
            subMIP_sol_best = self.subMIP_model.getBestSol()
            self.copy_solution_subMIP_to_MIP(subMIP_sol_best, self.subMIP_sol_best)
            subMIP_obj_best = self.subMIP_model.getSolObjVal(subMIP_sol_best)

            # case 3: time/solution limit reached with an improved solution
            if n_sols > 0 and subMIP_obj_best < self.subMIP_ub:

                self.subMIP_model.freeTransform()
                if not self.first:
                    # add the reversed right branch constraint to exclude MIP_sol_bar
                    if not self.is_heuristic:
                        if self.is_symmetric == True:
                            self.rightbranch_reverse(k=0.0)
                        else:
                            self.rightbranch_reverse_asym(k=0.0)

                self.copy_solution_subMIP_to_MIP(self.subMIP_sol_best, self.MIP_sol_bar)
                self.MIP_obj_bar = subMIP_obj_best

                self.diversify = False
                self.first = False

                state[0:5] = [0, 0, 1, 0, 0]

            # case 4: time/solution limit reached without improvement -> diversify
            else:

                self.subMIP_model.freeTransform()

                state[0:5] = [0, 0, 0, 1, 0]

                if self.diversify:
                    # add the reversed right branch constraint to exclude MIP_sol_bar
                    if not self.is_heuristic:
                        if self.is_symmetric == True:
                            self.rightbranch_reverse(k=0.0)
                        else:
                            self.rightbranch_reverse_asym(k=0.0)

                    self.div += 1
                    self.first = True
                    state[4] = 1
                self.diversify = True

        # update the best solution/objective and record the primal bound trajectory
        if n_sols_subMIP > 0:
            subMIP_sol_best = self.subMIP_model.getBestSol()
            subMIP_obj_best = self.subMIP_model.getSolObjVal(subMIP_sol_best)
            feasible = True

            if feasible and subMIP_obj_best < self.MIP_obj_best:
                self.MIP_sol_best = subMIP_sol_best
                self.MIP_obj_best = subMIP_obj_best
                success = True

                primal_bounds = self.primalbound_handler.primal_bounds
                primal_times = self.primalbound_handler.primal_times
                self.primal_no_improvement_account = 0

                for i in range(len(primal_times)):
                    primal_times[i] += self.total_time_expired

                self.primal_objs.extend(primal_bounds)
                self.primal_times.extend(primal_times)


        self.subMIP_model.delCons(self.constraint_LB)
        self.subMIP_model.releasePyCons(self.constraint_LB)
        del self.constraint_LB

        # hand-crafted t adaptation: enlarge the default node time limit after
        # every 5 consecutive non-improving iterations, shrink it back after
        # an improving one
        if enable_adapt_t:
            if self.primal_no_improvement_account > 0 and self.primal_no_improvement_account % 5 == 0:
                self.default_node_time_limit *= self.t_default_stepsize

            if self.default_node_time_limit > self.default_initial_node_time_limit and self.primal_no_improvement_account == 0:
                self.default_node_time_limit /= self.t_default_stepsize


        print('LB round: {:.0f}'.format(lb_bits),
              'Solving time: {:.4f}'.format(self.total_time_limit - self.total_time_available),
              'Best Obj: {:.4f}'.format(self.MIP_obj_best),
              'n_sols_subMIP: {:.0f}'.format(n_sols_subMIP),
              'K: {:.0f}'.format(k_pre),
              't_node: {:.1f}'.format(t_pre),
              'self.div: {:.0f}'.format(div_pre),
              'LB Status: {}'.format(subMIP_status),
              'Number of Nodes: {}'.format(n_nodes_subMIP)
              )


        # avoid negative time reward
        if t_leftbranch > t_node:
            t_leftbranch = t_node

        # last two state entries: normalized remaining node time and
        # normalized objective improvement of this iteration
        obj_norm = np.abs(MIP_obj_best_pre - self.MIP_obj_best)/ np.maximum(np.abs(MIP_obj_best_pre), np.abs(self.MIP_obj_best))
        t_norm = 1 - t_leftbranch / t_node
        state[5:7] = [t_norm, obj_norm]

        # reward for the k-policy: objective improvement of this iteration
        # (normalized by the initial objective) times the time still available
        obj_improve_local = np.abs(MIP_obj_best_pre - self.MIP_obj_best) / np.abs(self.MIP_obj_init)
        reward_k = obj_improve_local * self.total_time_available

        # reward for the t-policy: -1 when the sub-MIP hit the time limit
        # without improvement
        reward_t = 0
        if state[3] == 1:
            reward_t = -1

        done = (self.total_time_available <= 0) or (self.k >= self.n_binvars)

        self.total_time_expired += t_leftbranch
        return state, reward_k, reward_t, done, success

    def solve_rightbranch(self):
        """Solve the right-branch MIP with the remaining time budget.

        After the LB iterations are finished, the original MIP (with all
        reversed right-branch constraints added so far) is solved for the
        remaining time, looking only for solutions better than the incumbent.
        """
        self.primalbound_handler.primal_bounds = []
        self.primalbound_handler.primal_times = []
        if self.total_time_available > 0.1:
            self.MIP_model.setObjlimit(self.MIP_obj_best - self.eps)
            self.MIP_model.setParam('limits/time', self.total_time_available)

            self.MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.FAST)
            self.MIP_model.setPresolve(pyscipopt.SCIP_PARAMSETTING.FAST)

            print('try to run optimize()')
            self.MIP_model.optimize()
            print('right branch optimize() is finished with no error.')

            if self.MIP_model.getNSols() > 0:
                best_obj = self.MIP_model.getObjVal()
                if best_obj < self.MIP_obj_best:
                    self.MIP_obj_best = best_obj
                    if self.MIP_model.getNSols() > 0:
                        primal_bounds = self.primalbound_handler.primal_bounds
                        primal_times = self.primalbound_handler.primal_times

                        for i in range(len(primal_times)):
                            primal_times[i] += self.total_time_expired

                        self.primal_objs.extend(primal_bounds)
                        self.primal_times.extend(primal_times)


            self.total_time_available -= self.MIP_model.getSolvingTime()
            self.total_time_expired += self.MIP_model.getSolvingTime()

    def policy_vanilla(self, state):
        """Hand-crafted baseline policy mapping the LB state to (k, t) actions.

        Implements the classic LB update rules of Fischetti and Lodi: reset k
        after an improving iteration, enlarge the neighborhood when it is
        proven to contain no improving solution, and shrink it when the time
        limit is hit without improvement.
        """
        lb_status = state[0:4].argmax()
        if lb_status == 0:  # state[0:4] == [1, 0, 0, 0]
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
        """Return the new neighborhood size k resulting from the given action."""
        switcher = {
            self.actions['reset']: self.default_k,
            self.actions['unchange']: self.k,
            self.actions['decrease']: np.ceil(self.k - k_stepsize * self.k),
            self.actions['increase']: np.ceil(self.k + k_stepsize * self.k)
        }
        return switcher.get(action, 'Error: Invalid k action!')

    def update_t(self, action, t_stepsize):
        """Return the new node time limit t_node resulting from the given action."""
        switcher = {
            self.actions['reset']: self.default_node_time_limit,
            self.actions['unchange']: self.t_node,
            self.actions['decrease']: self.t_node / t_stepsize,
            self.actions['increase']: self.t_node * t_stepsize,
            self.actions['free']: self.MIP_model.infinity()
        }
        return switcher.get(action, 'Error: Invalid t action!')

    def mdp_localbranch(self, is_symmetric=True, reset_k_at_2nditeration=False, policy=None, optimizer=None, criterion=None, device=None, samples_dir=None):
        """Run the full LB search as an MDP, selecting k actions by a policy.

        The k action of each iteration is chosen by the given policy (an
        Agent from models_rl) or, if no policy is given, by the hand-crafted
        vanilla policy. Afterwards the right branch is solved with the
        remaining time.

        :return: (status, best objective, elapsed time, iteration indices,
            time stamps, objective values, loss, accuracy); the last two are
            kept for interface compatibility and remain zero.
        """
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

        while not done:
            lb_bits += 1

            # execute one iteration of LB and get the state and rewards
            state, reward_k, reward_t, done, _ = self.step_localbranch(k_action=k_action, t_action=t_action, lb_bits=lb_bits)

            # select the next k action: by the given policy (t stays
            # unchanged), or by the vanilla policy (which also sets t)
            if policy is not None:
                k_action = policy.select_action(state)
            else:
                k_vanilla, t_action = self.policy_vanilla(state)
                k_action = k_vanilla

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
        """Run the classic (self-contained) LB search with the vanilla rules.

        This is the plain LB baseline: the k and t updates of Fischetti and
        Lodi are applied inline, without the MDP interface. Afterwards the
        original MIP is solved with the remaining time.

        :return: (status, best objective, elapsed time, iteration indices,
            time stamps, objective values).
        """
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

        while self.total_time_available > 0 and self.k < self.n_binvars:
            lb_bits += 1

            # reset default_k and k at the 2nd iteration if the reset option is enabled
            if lb_bits == 2 and reset_k_at_2nditeration == True:
                self.default_k = 20
                if not self.is_symmetric:
                    self.default_k = 10
                self.k = self.default_k

                self.diversify = False
                self.first = False

            node_time_limit = np.minimum(node_time_limit, self.total_time_available)
            self.left_branch(node_time_limit, is_symmetric=self.is_symmetric)  # solve the LB sub-MIP

            node_time_limit = self.default_node_time_limit
            self.total_time_available -= self.subMIP_model.getSolvingTime()
            subMIP_status = self.subMIP_model.getStatus()

            div_pre = self.div
            k_pre = self.k

            # case 1: sub-MIP solved to optimality -> improved incumbent found
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

            # case 2: sub-MIP proven infeasible -> enlarge the neighborhood
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
                    self.first = True
                self.k += np.ceil(self.default_k / 2)
                self.diversify = True

            elif subMIP_status == "timelimit" or subMIP_status == "sollimit":
                n_sols = self.subMIP_model.getNSols()
                subMIP_sol_best = self.subMIP_model.getBestSol()
                self.copy_solution_subMIP_to_MIP(subMIP_sol_best, self.subMIP_sol_best)
                subMIP_obj_best = self.subMIP_model.getSolObjVal(subMIP_sol_best)

                # case 3: time/solution limit reached with an improved solution
                if n_sols > 0 and subMIP_obj_best < self.subMIP_ub:

                    self.subMIP_model.freeTransform()
                    if not self.first:
                        # add the reversed right branch constraint to exclude MIP_sol_bar
                        if self.is_symmetric == True:
                            self.rightbranch_reverse(k=0.0)
                        else:
                            self.rightbranch_reverse_asym(k=0.0)

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

                # case 4: time/solution limit reached without improvement
                else:

                    self.subMIP_model.freeTransform()
                    if self.diversify:
                        # add the reversed right branch constraint to exclude MIP_sol_bar
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
        """Add the reversed (symmetric) right-branch constraint to MIP_model.

        The constraint excludes the neighborhood of MIP_sol_bar of size k
        from the feasible region: Delta(x, MIP_sol_bar) >= k + 1.

        :param k: neighborhood size to exclude.
        """
        vars = self.MIP_model.getVars()
        n_binvars = self.MIP_model.getNBinVars()

        rhs = self.MIP_model.infinity()
        lhs = k + 1

        cons_vars = np.empty(n_binvars, dtype=object)
        cons_vals = np.empty(n_binvars)

        # compute coefficient for reversed LB constraint
        for i in range(0, n_binvars):
            val = self.MIP_model.getSolVal(self.MIP_sol_bar, vars[i])
            assert self.MIP_model.isFeasIntegral(val), "Error: Solution passed to LB is not integral!"

            if self.MIP_model.isFeasEQ(val, 1.0):
                cons_vals[i] = -1.0
                lhs -= 1.0
                rhs -= 1.0
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
        """Add the reversed asymmetric right-branch constraint to MIP_model.

        Asymmetric variant of rightbranch_reverse: the Hamming distance is
        measured only over the support of MIP_sol_bar.

        :param k: neighborhood size to exclude.
        """
        vars = self.MIP_model.getVars()
        n_binvars = self.MIP_model.getNBinVars()

        rhs = self.MIP_model.infinity()
        lhs = k + 1

        cons_vars = np.empty(n_binvars, dtype=object)
        cons_vals = np.empty(n_binvars)

        # compute coefficient for reversed LB constraint
        for i in range(0, n_binvars):
            val = self.MIP_model.getSolVal(self.MIP_sol_bar, vars[i])
            assert self.MIP_model.isFeasIntegral(val), "Error: Solution passed to LB is not integral!"

            if self.MIP_model.isFeasEQ(val, 1.0):
                cons_vals[i] = -1.0
                lhs -= 1.0
                rhs -= 1.0
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
        """Add the symmetric LB constraint (over all binary variables) to the sub-MIP."""

        vars = self.subMIP_model.getVars()
        n_binvars = self.subMIP_model.getNBinVars()

        lhs = 0
        rhs = self.k
        cons_vars = np.empty(n_binvars, dtype=object)
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

        del vars
        del cons_vars
        del cons_vals


    def add_LBconstraintAsym(self):
        """Add the asymmetric LB constraint (over the incumbent support) to the sub-MIP."""

        vars = self.subMIP_model.getVars()
        n_binvars = self.subMIP_model.getNBinVars()

        lhs = 0
        rhs = self.k
        cons_vars = np.empty(n_binvars, dtype=object)
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

        del vars
        del cons_vars
        del cons_vals


def addLBConstraint(mip_model, mip_sol, neighborhoodsize):
    """Add a symmetric LB constraint over all binary variables to the model.

    Restricts the search to solutions within Hamming distance
    `neighborhoodsize` of `mip_sol`.

    :return: (mip_model, the created constraint).
    """
    vars = mip_model.getVars()
    n_binvars = mip_model.getNBinVars()

    lhs = 0
    rhs = neighborhoodsize
    cons_vars = np.empty(n_binvars, dtype=object)
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
    del vars
    del cons_vars
    del cons_vals

    return mip_model, constraint_LB


def addLBConstraintAsymmetric(mip_model, mip_sol, neighborhoodsize):
    """Add an asymmetric LB constraint over the support of `mip_sol` to the model.

    Only variables at value 1 in `mip_sol` contribute to the distance.

    :return: (mip_model, the created constraint).
    """
    vars = mip_model.getVars()
    n_binvars = mip_model.getNBinVars()

    lhs = 0
    rhs = neighborhoodsize
    cons_vars = np.empty(n_binvars, dtype=object)
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

    del vars
    del cons_vars
    del cons_vals

    return mip_model, constraint_LB


def addLBConstraintAsymJustslackvars(mip_model, mip_sol, neighborhoodsize, indexlist_slackvars):
    """Add an asymmetric LB constraint over the given slack variables only.

    :param indexlist_slackvars: indices of the (binary) slack variables that
        contribute to the LB distance.
    :return: mip_model.
    """

    vars = mip_model.getVars()
    n_slackvars = len(indexlist_slackvars)

    lhs = 0
    rhs = neighborhoodsize
    cons_vars = np.empty(n_slackvars, dtype=object)
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
    del constraint_LB
    del vars
    del cons_vars
    del cons_vals

    return mip_model
