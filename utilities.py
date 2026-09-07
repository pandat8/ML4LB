"""Shared constants and helper functions used across the ML4LB code base.

This module defines:
- the canonical names of the benchmark datasets, instance sizes and the
  various algorithm modes used by the training/evaluation scripts;
- solution-copying helpers for moving primal solutions between a MIP and
  its (sub-)MIP copies (PySCIPOpt models);
- small numeric utilities (Hamming distances, mean filters, shifted means).
"""

import ecole
import numpy as np
from scipy.stats import gmean

# Benchmark datasets. The evaluation scripts select a dataset through the
# command-line argument --dataset_id, which indexes this list:
#   0: 'setcovering'                 (SC)
#   1: 'independentset'              (MIS)
#   2: 'combinatorialauction'        (CA)
#   3: 'generalized_independentset'  (GISP)
#   4: 'miplib_39binary'             (MIPLIB, 39 binary instances)
#   5: 'miplib2017_binary'           (MIPLIB 2017, binary instances)
#   6: 'miplib2017_binary_open'      (MIPLIB 2017, open binary instances)
instancetypes = ['setcovering', 'independentset', 'combinatorialauction',
                 'generalized_independentset', 'miplib_39binary',
                 'miplib2017_binary', 'miplib2017_binary_open']

# Dataset groups of Section 5: the models are trained on the synthetic
# datasets and additionally evaluated on the transfer datasets.
SYNTHETIC_DATASETS = instancetypes[0:3]   # 'setcovering', 'independentset', 'combinatorialauction'
TRANSFER_DATASETS = instancetypes[3:5]    # 'generalized_independentset', 'miplib_39binary'


def lbconstraint_mode_for(instance_type):
    """Return the LB constraint mode used in the paper for a dataset.

    The set covering instances use the asymmetric LB constraint; all other
    datasets use the symmetric one.
    """
    return 'asymmetric' if instance_type == 'setcovering' else 'symmetric'

# Instance size suffixes: transfer evaluations use '-small' (training size)
# and '-large' (scaled-up test size).
instancesizes = ['-small', '-large']

# The local branching (LB) constraint can be formulated symmetrically (over
# all binary variables) or asymmetrically (over the support of the incumbent).
lbconstraint_modes = ['symmetric', 'asymmetric']

# Type of incumbent solution the LB heuristic starts from:
# 'firstsol' = first solution found by SCIP, 'rootsol' = root-node solution.
incumbent_modes = ['firstsol', 'rootsol', 'firstrootsol']

# Variants of the regression model evaluated by evaluation_regression_k_prime:
# 'homo'     = model trained on the same (homogeneous) dataset,
# 'merged'   = model trained on the merged SC+MIS+CA dataset,
# 'baseline' = plain LB baseline without a regression model.
regression_modes = ['homo', 'merged', 'baseline']

# Reward signal used when training the RL policy for the node time limit t:
# 'reward_k' = objective-improvement reward of the k-policy,
# 'reward_k+t' = objective-improvement reward plus node-time reward,
# 'reward_t' = node-time reward only.
t_reward_types = ['reward_k', 'reward_k+t', 'reward_t']

# Averaging functions selectable through the --mean command-line argument.
mean_options = {'arithmetic': np.average, 'geometric': gmean}

# Average of the best k_0 values (normalized initial neighborhood size)
# observed on the training sets; used to initialize k for the
# 'lb_baseline_k0_average' variant.
k_0_bank = {'setcovering-firstsol': 0.8775714285714286,
            'setcovering-rootsol': 0.6008571428571429,
            'independentset-firstsol': 0.6095714285714284,
            'independentset-rootsol': 0.2434285714285714,
            'combinatorialauction-firstsol': 0.18371428571428572,
            'combinatorialauction-rootsol': 0.30142857142857143,
            'merged': 0.46942857142857136}


def generator_switcher(dataset):
    """Return an ecole instance generator for the synthetic dataset name.

    :param dataset: dataset name, i.e. instance type plus size suffix,
        e.g. 'setcovering-small'.
    :return: an ecole.instance generator object.
    """
    switcher = {
        'setcovering-small': lambda: ecole.instance.SetCoverGenerator(n_rows=5000, n_cols=2000, density=0.01),
        'setcovering-large': lambda: ecole.instance.SetCoverGenerator(n_rows=10000, n_cols=4000, density=0.01),
        'independentset-small': lambda: ecole.instance.IndependentSetGenerator(n_nodes=1000),
        'independentset-large': lambda: ecole.instance.IndependentSetGenerator(n_nodes=2000),
        'combinatorialauction-small': lambda: ecole.instance.CombinatorialAuctionGenerator(n_items=4000, n_bids=2000, add_item_prob=0.6),
        'combinatorialauction-large': lambda: ecole.instance.CombinatorialAuctionGenerator(n_items=8000, n_bids=4000, add_item_prob=0.60),
    }
    return switcher.get(dataset, lambda: "invalide argument")()


def copy_sol(mip_original, mip_target, sol, mip_target_vars):
    """Copy a solution of the original MIP to a copy of that MIP.

    The solution is checked for feasibility and, if feasible, added to the
    solution pool of the target model.

    :param mip_original: source PySCIPOpt model.
    :param mip_target: target PySCIPOpt model (a copy of mip_original).
    :param sol: solution of mip_original to copy.
    :param mip_target_vars: variables of mip_target, ordered as in mip_original.
    :return: (mip_target, copied solution).
    """
    sol_mip_target = mip_target.createSol()

    n_vars = mip_original.getNVars()
    mip_original_vars = mip_original.getVars()
    for j in range(n_vars):
        val = mip_original.getSolVal(sol, mip_original_vars[j])
        mip_target.setSolVal(sol_mip_target, mip_target_vars[j], val)
    feasible = mip_target.checkSol(solution=sol_mip_target)

    if feasible:
        mip_target.addSol(sol_mip_target, False)
    else:
        print("Error: the trivial solution of " + mip_target.getProbName() + " is not feasible!")
    return mip_target, sol_mip_target


def copy_sol_from_subMIP_to_MIP(subMIP_model, MIP_model, sol_subMIP, subMIP_vars, check_feasibility=True, add_sol=True):
    """Copy a solution of a sub-MIP back to the original MIP.

    :param subMIP_model: source PySCIPOpt model (the LB sub-MIP).
    :param MIP_model: target PySCIPOpt model (the original MIP).
    :param sol_subMIP: solution of subMIP_model to copy.
    :param subMIP_vars: variables of subMIP_model, ordered as in MIP_model.
    :param check_feasibility: if True, check the solution before/after copying.
    :param add_sol: if True, add the feasible copied solution to MIP_model.
    :return: (MIP_model, copied solution, feasibility flag).
    """
    print("start copying solution of subMIP to MIP")
    if check_feasibility:
        feasible = subMIP_model.checkSol(solution=sol_subMIP)
        print("check feasibility")
        assert feasible, "Error: the trivial solution of the subMIP model " + subMIP_model.getProbName() + " is not feasible!"

    print("try to initialize a new solution")
    sol_mip_target = MIP_model.createSol()
    print("a new solution is initialized!")

    n_vars = MIP_model.getNVars()
    MIP_vars = MIP_model.getVars()
    for j in range(n_vars):
        val = subMIP_model.getSolVal(sol_subMIP, subMIP_vars[j])
        MIP_model.setSolVal(sol_mip_target, MIP_vars[j], val)
    if check_feasibility:
        feasible = MIP_model.checkSol(solution=sol_mip_target)
    else:
        feasible = True

    if add_sol and feasible:
        MIP_model.addSol(sol_mip_target, False)

    return MIP_model, sol_mip_target, feasible


def copy_sol_from_subMIP_to_MIP_heur(heur, subMIP_model, MIP_model, sol_subMIP, subMIP_vars, check_feasibility=True, add_sol=True):
    """Copy a solution of a sub-MIP back to the original MIP inside a SCIP heuristic.

    Same as copy_sol_from_subMIP_to_MIP, but the new solution is linked to
    the calling primal heuristic (required by the SCIP heuristic callback).

    :param heur: the SCIP primal heuristic that found the solution.
    :param subMIP_model: source PySCIPOpt model (the LB sub-MIP).
    :param MIP_model: target PySCIPOpt model (the original MIP).
    :param sol_subMIP: solution of subMIP_model to copy.
    :param subMIP_vars: variables of subMIP_model, ordered as in MIP_model.
    :param check_feasibility: if True, check the solution before/after copying.
    :param add_sol: if True, add the feasible copied solution to MIP_model.
    :return: (MIP_model, copied solution, feasibility flag).
    """
    if check_feasibility:
        feasible = subMIP_model.checkSol(solution=sol_subMIP)
        assert feasible, "Error: the trivial solution of the subMIP model " + subMIP_model.getProbName() + " is not feasible!"

    sol_mip_target = MIP_model.createSol(heur)

    n_vars = MIP_model.getNVars()
    MIP_vars = MIP_model.getVars()
    for j in range(n_vars):
        val = subMIP_model.getSolVal(sol_subMIP, subMIP_vars[j])
        MIP_model.setSolVal(sol_mip_target, MIP_vars[j], val)
    if check_feasibility:
        feasible = MIP_model.checkSol(solution=sol_mip_target)
    else:
        feasible = True

    if add_sol and feasible:
        MIP_model.addSol(sol_mip_target, False)

    return MIP_model, sol_mip_target, feasible


def binary_support(mip, sol):
    """Count the binary variables set to 1 in the given solution.

    :param mip: PySCIPOpt model.
    :param sol: solution of the model.
    :return: number of binary variables at value 1 (the binary support).
    """
    n_binvars = mip.getNBinVars()
    vars = mip.getVars()
    n_supportbinvars = 0
    for i in range(n_binvars):
        val = mip.getSolVal(sol, vars[i])
        assert mip.isFeasIntegral(val), "Error: Value of a binary varialbe is not integral!"
        if mip.isFeasEQ(val, 1.0):
            n_supportbinvars += 1
    return n_supportbinvars


def mean_filter(a, kernal_size):
    """Smooth a 1-D array with a centered moving-average filter.

    Boundary entries that the kernel does not fully cover are left unchanged.

    :param a: 1-D numpy array.
    :param kernal_size: (odd) window size of the filter.
    :return: filtered array of the same shape.
    """
    a_mean = np.zeros(a.shape)

    k = int((kernal_size - 1) / 2)

    for i in range(a.shape[0]):
        if i < k or i > (a.shape[0] - 1 - k):
            a_mean[i] = a[i]
        else:
            for n in range(kernal_size):
                a_mean[i] += a[i - k + n]
            a_mean[i] = a_mean[i] / kernal_size
    return a_mean


def mean_forward_filter(a, kernal_size):
    """Smooth a 1-D array with a forward-looking moving-average filter.

    Entries too close to the end of the array are left unchanged.

    :param a: 1-D numpy array.
    :param kernal_size: window size of the filter.
    :return: filtered array of the same shape.
    """
    a_mean = np.zeros(a.shape)

    k = kernal_size

    for i in range(a.shape[0]):
        if i > (a.shape[0] - 1 - k):
            a_mean[i] = a[i]
        else:
            for n in range(kernal_size):
                a_mean[i] += a[i + n]
            a_mean[i] = a_mean[i] / kernal_size
    return a_mean


def imitation_accuracy(k_pred, k_label):
    """Compute the top-1 accuracy of predicted action logits against labels.

    :param k_pred: tensor of predicted logits, shape (batch, n_actions).
    :param k_label: tensor of ground-truth action labels, shape (batch,).
    :return: accuracy as a scalar tensor.
    """
    top_pred = k_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(k_label.view_as(top_pred)).sum()
    acc = correct.float() / k_label.shape[0]
    return acc


def haming_distance_solutions(mip_model, sol1, sol2):
    """Hamming distance between two solutions over all binary variables.

    :param mip_model: PySCIPOpt model the solutions belong to.
    :param sol1: first solution.
    :param sol2: second solution.
    :return: sum of absolute differences over the binary variables.
    """
    vars = mip_model.getVars()
    n_bins = mip_model.getNBinVars()
    delta = 0
    for i in range(0, n_bins):
        val1 = mip_model.getSolVal(sol1, vars[i])
        val2 = mip_model.getSolVal(sol2, vars[i])

        delta += np.abs(val1 - val2)

    return delta


def haming_distance_solutions_asym(mip_model, sol1, sol2):
    """Asymmetric Hamming distance between two solutions.

    Only binary variables at value 1 in sol1 (the support of sol1)
    contribute to the distance.

    :param mip_model: PySCIPOpt model the solutions belong to.
    :param sol1: first (reference) solution.
    :param sol2: second solution.
    :return: sum of absolute differences over the support of sol1.
    """
    vars = mip_model.getVars()
    n_bins = mip_model.getNBinVars()
    delta = 0
    for i in range(0, n_bins):
        val1 = mip_model.getSolVal(sol1, vars[i])
        val2 = mip_model.getSolVal(sol2, vars[i])
        assert mip_model.isFeasIntegral(val1), "Error: Solution passed to LB is not integral!"

        if mip_model.isFeasEQ(val1, 1.0):
            delta += np.abs(val1 - val2)

    return delta


def getBestFeasiSol(mip_model):
    """Return the best feasible solution stored in a SCIP model.

    For numerical reasons a solution reported by SCIP may fail the
    feasibility check; in that case the stored solutions are checked from
    best to worst and the first feasible one is returned.

    :param mip_model: PySCIPOpt model.
    :return: (feasibility flag, best feasible solution or None, its objective or None).
    """
    obj_best = None
    sol_best = None
    feasible = False
    n_sols = mip_model.getNSols()

    mip_model.freeTransform()

    if n_sols > 0:
        sol_best_candidate = mip_model.getBestSol()
        feasible = mip_model.checkSol(solution=sol_best_candidate)

        if not feasible:
            sols = mip_model.getSols()
            for i in range(len(sols)):
                feasible = mip_model.checkSol(solution=sols[i])
                if feasible:
                    sol_best = sols[i]
                    obj_best = mip_model.getSolObjVal(sol_best)
                    break
            del sols
        else:
            sol_best = sol_best_candidate
            obj_best = mip_model.getSolObjVal(sol_best)

    return feasible, sol_best, obj_best


def mean_shift(data, axis=None, mean_option='arithmetic', shift=100):
    """Compute the shifted arithmetic or geometric mean of the input data.

    The shift makes the geometric mean well defined for data containing
    zeros (the shift is added before and subtracted after averaging).

    :param data: numpy array of values.
    :param axis: axis over which to average (as in numpy).
    :param mean_option: 'arithmetic' or 'geometric', see mean_options.
    :param shift: shift value added before averaging.
    :return: the (shifted) mean of the data.
    """
    func_mean = mean_options[mean_option]
    data = data + shift
    data_mean = func_mean(data, axis=axis)
    data_mean = data_mean - shift

    return data_mean
