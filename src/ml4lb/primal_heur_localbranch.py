"""The ML-based local branching algorithm as a SCIP primal heuristic.

This file implements the ML-based local branching algorithm as a primal
heuristic integrated into SCIP (via the PySCIPOpt Heur plugin interface).
"""

from pyscipopt import Heur, SCIP_RESULT
from ml4lb.utilities import copy_sol, copy_sol_from_subMIP_to_MIP, t_reward_types
import numpy as np
import torch

from ml4lb.localbranching import LocalBranching


class HeurLocalbranch(Heur):
    """LB primal heuristic executed once inside the SCIP solving process.

    On execution, the heuristic copies the current MIP and its incumbent,
    runs the (ML-guided) local branching search on the copy, and injects any
    improved solution back into the main SCIP model.

    :param k_0: initial neighborhood size (e.g. predicted by the GNN).
    :param agent_k: optional RL agent selecting the k action per iteration.
    :param agent_t: optional RL agent selecting the t action per iteration.
    :param optim_k: optional optimizer for updating agent_k online.
    """

    def __init__(self, k_0, node_time_limit, total_time_limit, is_symmetric, is_heuristic, reset_k_at_2nditeration, no_improve_iteration_limit, device, agent_k=None, agent_t=None, optim_k=None):
        super().__init__()
        self.k_0 = k_0
        self.agent_k = agent_k
        self.agent_t = agent_t
        self.optim_k = optim_k
        self.node_time_limit = node_time_limit
        self.total_time_limit = total_time_limit
        self.is_symmetric = is_symmetric
        self.is_heuristic = is_heuristic
        self.reset_k_at_2nditeration = reset_k_at_2nditeration
        self.device = device
        self.no_improve_iteration_limit = no_improve_iteration_limit

        self.alpha = 0.01
        self.gamma = 0.99                          # discount factor of the RL return
        self.eps = np.finfo(np.float32).eps.item()  # numerical safeguard

    def heurexec(self, heurtiming, nodeinfeasible):
        """SCIP heuristic callback: run one LB search from the current incumbent."""

        print('LB heuristic is starting..')
        lb_start_time = self.model.getSolvingTime()
        print('LB heuristic starts at {} s'.format(str(lb_start_time)))

        incumbent_solution = self.model.getBestSol()

        assert (incumbent_solution is not None), 'initial solution of LB is None'
        assert self.model.checkSol(incumbent_solution), 'initial solution of LB is not feasible'

        lb_start_obj = self.model.getSolObjVal(incumbent_solution)
        print('LB initial objective:', lb_start_obj)

        feas = self.model.checkSol(incumbent_solution)
        if feas:
            print('The init sol of original MIP is feasible')
        else:
            print('Error: The init sol of original MIP is not feasible!')

        n_binvars = self.model.getNBinVars()
        fixed_vals = np.empty(n_binvars)
        fixed_vars = np.empty(n_binvars, dtype=object)
        MIP_model_copy, MIP_copy_vars, success = self.model.createCopyMipLns(fixed_vars, fixed_vals, 0, uselprows=False,
                                                                  copycuts=True)


        MIP_model_copy, sol_MIP_copy = copy_sol(self.model, MIP_model_copy, incumbent_solution,
                                                  MIP_copy_vars)

        # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
        lb = LocalBranching(MIP_model=MIP_model_copy,
                            MIP_sol_bar=sol_MIP_copy,
                            MIP_vars=MIP_copy_vars,
                            k=self.k_0,
                            node_time_limit=self.node_time_limit,
                            total_time_limit=self.total_time_limit,
                            is_symmetric=self.is_symmetric,
                            is_heuristic=self.is_heuristic
                            )

        status, obj_best, elapsed_time, agent_k, _, success_lb = self.mdp_localbranch(
            localbranch=lb,
            is_symmetric=self.is_symmetric,
            reset_k_at_2nditeration=self.reset_k_at_2nditeration,
            agent_k=self.agent_k,
            agent_t=self.agent_t,
            optimizer_k=None,
            device=self.device)

        if agent_k is not None:
            self.agent_k, self.optim_k, R = self.update_agent(agent_k, self.optim_k)

        lb_start_time = self.model.getSolvingTime()
        print('LB heuristic finishes at {} s'.format(str(lb_start_time)))

        if success_lb:
            print('LB heuristic succeeds, finds an improving solution')
            incumbent_solution = self.model.getBestSol()
            lb_final_obj = self.model.getSolObjVal(incumbent_solution)
            print('LB final objective:', lb_final_obj)

            return {"result": SCIP_RESULT.FOUNDSOL}
        else:
            print('LB heuristic fails, can not find an improving solution')
            return {"result": SCIP_RESULT.DIDNOTFIND}


    def mdp_localbranch(self, localbranch=None, is_symmetric=True, reset_k_at_2nditeration=False, agent_k=None,
                        optimizer_k=None, agent_t=None, optimizer_t=None, device=None, enable_adapt_t=False,
                        t_reward_type=t_reward_types[0]):
        """Run the LB search as an MDP and copy improving solutions to SCIP.

        Same LB loop as RlLocalbranch.mdp_localbranch, with two additions:
        every improved solution is copied back to the main SCIP model, and
        the search additionally stops after no_improve_iteration_limit
        consecutive non-improving iterations.

        :return: (status, best objective, elapsed time, agent_k, agent_t,
            success flag of injecting an improved solution into SCIP).
        """
        success = False

        localbranch.total_time_available = localbranch.total_time_limit
        localbranch.first = False
        localbranch.diversify = False
        localbranch.t_node = localbranch.default_node_time_limit
        localbranch.div = 0
        localbranch.is_symmetric = is_symmetric
        localbranch.reset_k_at_2nditeration = reset_k_at_2nditeration
        lb_bits = 0


        k_action = localbranch.actions['unchange']
        t_action = localbranch.actions['unchange']

        # initialize the env to state_0
        lb_bits += 1
        state, reward_k, reward_time, done, success_step = localbranch.step_localbranch(k_action=k_action, t_action=t_action,
                                                                             lb_bits=lb_bits)
        done = done or (localbranch.primal_no_improvement_account > self.no_improve_iteration_limit - 1)

        if success_step and (localbranch.MIP_vars is not None):
            _, _, feasible = copy_sol_from_subMIP_to_MIP(localbranch.MIP_model, self.model,
                                                                  localbranch.MIP_sol_best, localbranch.MIP_vars)
            if feasible:
                success = True

        localbranch.MIP_obj_init = localbranch.MIP_obj_best

        if (not done) and reset_k_at_2nditeration:
            lb_bits += 1
            localbranch.default_k = 20
            if not localbranch.is_symmetric:
                localbranch.default_k = 10
            localbranch.k = localbranch.default_k
            localbranch.diversify = False
            localbranch.first = False

            state, reward_k, reward_time, done, success_step = localbranch.step_localbranch(k_action=k_action,
                                                                                 t_action=t_action,
                                                                                 lb_bits=lb_bits)
            done = done or (localbranch.primal_no_improvement_account > self.no_improve_iteration_limit - 1)
            if success_step and (localbranch.MIP_vars is not None) :
                _, _, feasible = copy_sol_from_subMIP_to_MIP(localbranch.MIP_model, self.model, localbranch.MIP_sol_best, localbranch.MIP_vars)
                if feasible:
                    success = True

            localbranch.MIP_obj_init = localbranch.MIP_obj_best

        while (not done) and localbranch.div < localbranch.div_max :
            lb_bits += 1

            k_vanilla, t_action = localbranch.policy_vanilla(state)


            k_action = k_vanilla
            if agent_k is not None:
                k_action = agent_k.select_action(state)

            if agent_t is not None:
                t_action = agent_t.select_action(state)


            # execute one iteration of LB, get the state and rewards

            state, reward_k, reward_time, done, success_step = localbranch.step_localbranch(k_action=k_action,
                                                                                 t_action=t_action, lb_bits=lb_bits,
                                                                                 enable_adapt_t=enable_adapt_t)
            done = done or (localbranch.primal_no_improvement_account > self.no_improve_iteration_limit - 1)
            if success_step and (localbranch.MIP_vars is not None) :
                _, _, feasible = copy_sol_from_subMIP_to_MIP(localbranch.MIP_model, self.model, localbranch.MIP_sol_best, localbranch.MIP_vars)

                if feasible:
                    success = True


            if agent_k is not None:
                agent_k.rewards.append(reward_k)
            if agent_t is not None:
                if t_reward_type == t_reward_types[1]:
                    reward_t = reward_k + reward_time
                else:
                    reward_t = reward_k
                agent_t.rewards.append(reward_t)


        print(
            'K_final: {:.0f}'.format(localbranch.k),
            'div_final: {:.0f}'.format(localbranch.div)
        )


        status = localbranch.MIP_model.getStatus()

        elapsed_time = localbranch.total_time_limit - localbranch.total_time_available


        del localbranch.subMIP_sol_best
        del localbranch.MIP_sol_bar
        del localbranch.MIP_sol_best

        return status, localbranch.MIP_obj_best, elapsed_time, agent_k, agent_t, success

    def update_agent(self, agent, optimizer):
        """REINFORCE update of the agent policy from the collected rewards.

        Computes the discounted, normalized returns of the episode and, if an
        optimizer is given, performs one policy-gradient step. The reward and
        log-probability buffers of the agent are cleared afterwards.

        :return: (agent, optimizer, undiscounted final return R).
        """
        R = 0
        policy_losses = []
        returns = []
        # calculate the return
        for r in agent.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
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


class HeurLocalbranchMulticall(Heur):
    """LB primal heuristic that may be executed multiple times by SCIP.

    Variant of HeurLocalbranch for repeated calls within one solve: the LB
    search is only started on the first call, or when SCIP has improved the
    incumbent since the previous LB call (to avoid wasting time re-searching
    the same neighborhood).
    """

    def __init__(self, k_0, node_time_limit, total_time_limit, is_symmetric, is_heuristic, reset_k_at_2nditeration, no_improve_iteration_limit, device, agent_k=None, agent_t=None, optim_k=None):
        super().__init__()
        self.k_0 = k_0
        self.agent_k = agent_k
        self.agent_t = agent_t
        self.optim_k = optim_k
        self.node_time_limit = node_time_limit
        self.total_time_limit = total_time_limit
        self.is_symmetric = is_symmetric
        self.is_heuristic = is_heuristic
        self.reset_k_at_2nditeration = reset_k_at_2nditeration
        self.device = device
        self.no_improve_iteration_limit = no_improve_iteration_limit

        self.alpha = 0.01
        self.gamma = 0.99                          # discount factor of the RL return
        self.eps = np.finfo(np.float32).eps.item()  # numerical safeguard
        self.n_lb_calls = 0
        self.lb_last_obj = 0.0

    def heurexec(self, heurtiming, nodeinfeasible):
        """SCIP heuristic callback: run an LB search if the incumbent is new."""

        print('LB heuristic is starting..')
        lb_start_time = self.model.getSolvingTime()
        print('LB heuristic starts at {} s'.format(str(lb_start_time)))

        incumbent_solution = self.model.getBestSol()

        assert (incumbent_solution is not None), 'initial solution of LB is None'

        lb_start_obj = self.model.getSolObjVal(incumbent_solution)
        if self.n_lb_calls == 0:
            self.lb_last_obj = lb_start_obj

        # call local branching when 1. first time when call it 2. when the incumbent is better than the incumbent after the last call of LB
        if self.n_lb_calls == 0 or lb_start_obj < self.lb_last_obj:
            self.n_lb_calls += 1
            print('No. of LB call: ', self.n_lb_calls)
            print('LB initial objective:', lb_start_obj)

            feas = self.model.checkSol(incumbent_solution)
            if feas:
                print('The init sol of original MIP is feasible')
            else:
                print('LB heuristic exits, since the initial incumbent solution passed to LB is not feasible!')
                return {"result": SCIP_RESULT.DIDNOTFIND}

            n_binvars = self.model.getNBinVars()
            fixed_vals = np.empty(n_binvars)
            fixed_vars = np.empty(n_binvars, dtype=object)

            MIP_model_copy, MIP_copy_vars, success = self.model.createCopyMipLns(fixed_vars, fixed_vals, 0, uselprows=False,
                                                                      copycuts=True)


            MIP_model_copy, sol_MIP_copy = copy_sol(self.model, MIP_model_copy, incumbent_solution,
                                                      MIP_copy_vars)

            # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
            lb = LocalBranching(MIP_model=MIP_model_copy,
                                MIP_sol_bar=sol_MIP_copy,
                                MIP_vars=MIP_copy_vars,
                                k=self.k_0,
                                node_time_limit=self.node_time_limit,
                                total_time_limit=self.total_time_limit,
                                is_symmetric=self.is_symmetric,
                                is_heuristic=self.is_heuristic
                                )

            status, obj_best, elapsed_time, agent_k, _, success_lb = self.mdp_localbranch(
                localbranch=lb,
                is_symmetric=self.is_symmetric,
                reset_k_at_2nditeration=self.reset_k_at_2nditeration,
                agent_k=self.agent_k,
                agent_t=self.agent_t,
                optimizer_k=None,
                device=self.device)


            if agent_k is not None:
                self.agent_k, self.optim_k, R = self.update_agent(agent_k, self.optim_k)

            lb_start_time = self.model.getSolvingTime()
            print('LB heuristic finishes at {} s'.format(str(lb_start_time)))

            incumbent_solution = self.model.getBestSol()
            lb_final_obj = self.model.getSolObjVal(incumbent_solution)
            print('LB final objective:', lb_final_obj)
            self.lb_last_obj = lb_final_obj

            if success_lb:
                print('LB heuristic succeeds, finds an improving solution')
                return {"result": SCIP_RESULT.FOUNDSOL}
            else:
                print('LB heuristic fails, can not find an improving solution')
                return {"result": SCIP_RESULT.DIDNOTFIND}
        else:
            print('LB heuristic exits, since the incumbent is not updated by the solver after last LB call')
            return {"result": SCIP_RESULT.DIDNOTFIND}


    def mdp_localbranch(self, localbranch=None, is_symmetric=True, reset_k_at_2nditeration=False, agent_k=None,
                        optimizer_k=None, agent_t=None, optimizer_t=None, device=None, enable_adapt_t=False,
                        t_reward_type=t_reward_types[0]):
        """Run the LB search as an MDP; see HeurLocalbranch.mdp_localbranch."""
        success = False

        localbranch.total_time_available = localbranch.total_time_limit
        localbranch.first = False
        localbranch.diversify = False
        localbranch.t_node = localbranch.default_node_time_limit
        localbranch.div = 0
        localbranch.is_symmetric = is_symmetric
        localbranch.reset_k_at_2nditeration = reset_k_at_2nditeration
        lb_bits = 0


        k_action = localbranch.actions['unchange']
        t_action = localbranch.actions['unchange']

        # initialize the env to state_0
        lb_bits += 1
        state, reward_k, reward_time, done, success_step = localbranch.step_localbranch(k_action=k_action, t_action=t_action,
                                                                             lb_bits=lb_bits)
        done = done or (localbranch.primal_no_improvement_account > self.no_improve_iteration_limit - 1)

        if success_step and (localbranch.MIP_vars is not None):
            _, _, feasible = copy_sol_from_subMIP_to_MIP(localbranch.MIP_model, self.model,
                                                                  localbranch.MIP_sol_best, localbranch.MIP_vars)
            if feasible:
                success = True

        localbranch.MIP_obj_init = localbranch.MIP_obj_best

        if (not done) and reset_k_at_2nditeration:
            lb_bits += 1
            localbranch.default_k = 20
            if not localbranch.is_symmetric:
                localbranch.default_k = 10
            localbranch.k = localbranch.default_k
            localbranch.diversify = False
            localbranch.first = False

            state, reward_k, reward_time, done, success_step = localbranch.step_localbranch(k_action=k_action,
                                                                                 t_action=t_action,
                                                                                 lb_bits=lb_bits)
            done = done or (localbranch.primal_no_improvement_account > self.no_improve_iteration_limit - 1)
            if success_step and (localbranch.MIP_vars is not None) :
                _, _, feasible = copy_sol_from_subMIP_to_MIP(localbranch.MIP_model, self.model, localbranch.MIP_sol_best, localbranch.MIP_vars)
                if feasible:
                    success = True

            localbranch.MIP_obj_init = localbranch.MIP_obj_best

        while (not done) and localbranch.div < localbranch.div_max :
            lb_bits += 1

            k_vanilla, t_action = localbranch.policy_vanilla(state)


            k_action = k_vanilla
            if agent_k is not None:
                k_action = agent_k.select_action(state)

            if agent_t is not None:
                t_action = agent_t.select_action(state)


            # execute one iteration of LB, get the state and rewards

            state, reward_k, reward_time, done, success_step = localbranch.step_localbranch(k_action=k_action,
                                                                                 t_action=t_action, lb_bits=lb_bits,
                                                                                 enable_adapt_t=enable_adapt_t)
            done = done or (localbranch.primal_no_improvement_account > self.no_improve_iteration_limit - 1)
            if success_step and (localbranch.MIP_vars is not None) :
                _, _, feasible = copy_sol_from_subMIP_to_MIP(localbranch.MIP_model, self.model, localbranch.MIP_sol_best, localbranch.MIP_vars)

                if feasible:
                    success = True


            if agent_k is not None:
                agent_k.rewards.append(reward_k)
            if agent_t is not None:
                if t_reward_type == t_reward_types[1]:
                    reward_t = reward_k + reward_time
                else:
                    reward_t = reward_k
                agent_t.rewards.append(reward_t)


        print(
            'K_final: {:.0f}'.format(localbranch.k),
            'div_final: {:.0f}'.format(localbranch.div)
        )


        status = localbranch.MIP_model.getStatus()

        elapsed_time = localbranch.total_time_limit - localbranch.total_time_available


        del localbranch.subMIP_sol_best
        del localbranch.MIP_sol_bar
        del localbranch.MIP_sol_best

        return status, localbranch.MIP_obj_best, elapsed_time, agent_k, agent_t, success

    def update_agent(self, agent, optimizer):
        """REINFORCE update of the agent policy; see HeurLocalbranch.update_agent."""
        R = 0
        policy_losses = []
        returns = []
        # calculate the return
        for r in agent.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        # calculate loss
        with torch.set_grad_enabled(optimizer is not None):
            for log_prob, Return in zip(agent.log_probs, returns):
                policy_losses.append(-log_prob * Return)

            # optimize policy network (skipped when the episode is empty)
            if (optimizer is not None) and (not len(policy_losses) == 0):
                optimizer.zero_grad()
                policy_losses = torch.cat(policy_losses).sum()
                policy_losses.backward()
                optimizer.step()

        del agent.rewards[:]
        del agent.log_probs[:]
        return agent, optimizer, R


