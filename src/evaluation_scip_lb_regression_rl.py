"""Evaluate SCIP integrated with the ML-based local branching heuristic (Section 6).

For the selected dataset (--dataset_id, default 'miplib_39binary'), SCIP
solves every test instance with the ML-based LB primal heuristic included:
the initial neighborhood size k_0 is predicted by the pre-trained regression
model and the k updates are selected by the pre-trained RL policy. The
heuristic is called in the branch-and-bound tree with frequency --freq:

- --freq=0   : scip-lb-regression-rl-single (root node only);
- --freq=1   : scip-lb-regression-rl-freq1;
- --freq=100 : scip-lb-regression-rl-freq100.

The primal bound trajectories are stored under ./result/ and are aggregated
afterwards by compute_evaluation_results_scip_seeds_averaged.py. See the
README for the exact commands reproducing Section 6.
"""

import ecole
import numpy as np
import pyscipopt
import argparse
from ml4lb.execute_heuristics import Execute_LB_Regression_RL
from ml4lb.utilities import instancetypes, instancesizes, TRANSFER_DATASETS, lbconstraint_mode_for
import torch
import random
import pathlib
from ml4lb.models import GNNPolicy
from ml4lb.models_rl import SimplePolicy, AgentReinforce

# Command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--regression_model_path', type=str,
                    default='./result/saved_models/regression/trained_params_mean_setcover-independentset-combinatorialauction_asymmetric_firstsol_k_prime_epoch163.pth',
                    help='path of the pre-trained regression model for predicting k_0')
parser.add_argument('--rl_model_path', type=str,
                    default='./result/saved_models/rl/reinforce/setcovering/checkpoint_trained_reward3_simplepolicy_rl4lb_reinforce_trainset_setcovering-small_lr0.01_epochs7.pth',
                    help='path of the pre-trained RL policy for adapting k')
parser.add_argument('--dataset_id', type=int, default=4,
                    help='dataset to evaluate, index into ml4lb.utilities.instancetypes '
                         "(4: 'miplib_39binary', 5: 'miplib2017_binary')")
parser.add_argument('--t_total', type=int, default=3600, help='total time limit (s) per instance')
parser.add_argument('--t_node', type=int, default=2, help='node time limit (s) per LB sub-MIP')
parser.add_argument('--freq', type=int, default=0,
                    help='frequency of calling the LB primal heuristic in the branch-and-bound tree')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--enable_gpu', action='store_true', help='Enable CUDA GPU acceleration')
args = parser.parse_args()

# Select the device for the ML models; the device tag is also part of the
# result directory name.
enable_gpu = args.enable_gpu
if enable_gpu and torch.cuda.is_available():
    device = torch.device('cuda')
    device_str = 'cuda'
else:
    device = torch.device('cpu')
    device_str = 'cpu'

# Experiment configuration from the command line.
regression_model_path = args.regression_model_path
rl_model_path = args.rl_model_path
print(regression_model_path)
print(rl_model_path)

freq = args.freq
print('The frequency of calling LB primal heuristic within SCIP BB tree is : ', freq)

# Fix all random seeds for reproducibility.
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

dataset_id = args.dataset_id

total_time_limit = args.t_total
node_time_limit = args.t_node

# Run the LB algorithm as a primal heuristic inside SCIP.
is_heuristic = True

# The LB search stops after this many consecutive non-improving iterations.
no_improve_iteration_limit = 2

# Learning rate of the (loaded) RL policy optimizer.
lr = 0.01

# Load the pre-trained regression model (GNN) predicting the initial
# neighborhood size k_0.
regression_model_gnn = GNNPolicy()
regression_model_gnn.load_state_dict(torch.load(regression_model_path))

# Load the pre-trained RL policy for adapting k during the LB search,
# together with its optimizer state.
rl_policy = SimplePolicy(7, 4)
checkpoint = torch.load(rl_model_path)
rl_policy.load_state_dict(checkpoint['model_state_dict'])

rl_policy = rl_policy.to(device)
rl_policy.train()

optim_k = torch.optim.Adam(rl_policy.parameters(), lr=lr)
optim_k.load_state_dict(checkpoint['optimizer_state_dict'])
# move the loaded optimizer state to the selected device
for state in optim_k.state.values():
    for state_key, state_value in state.items():
        if torch.is_tensor(state_value):
            state[state_key] = state_value.to(device)

# Wrap the policy into a (non-greedy) REINFORCE agent.
greedy = False
agent_k = AgentReinforce(rl_policy, device, greedy, optim_k, 0.0)

# Select the dataset and the LB constraint mode used for it in the paper.
instance_type = instancetypes[dataset_id]
lbconstraint_mode = lbconstraint_mode_for(instance_type)

# Main loop; Section 6 evaluates the runs started from the root solution.
for incumbent_mode in ['rootsol']:

    for instance_size in instancesizes:

        # Log the configuration of this run.
        print(instance_type + instance_size)
        print(incumbent_mode)

        # Input directories: test instances and their stored incumbents.
        source_directory = './data/generated_instances/' + instance_type + '/' + instance_size + '/'
        instance_directory = source_directory + 'transformedmodel' + '/' + 'test/'
        solution_directory = source_directory + incumbent_mode + '/' + 'test/'

        # Output directory for the primal bound trajectories of this run
        # (the comparison scripts rebuild exactly the same path).
        evaluation_directory = './result/generated_instances/' + instance_type + '/' + instance_size + '/' + incumbent_mode + '/' + 'scip/'

        if is_heuristic:
            evaluation_directory = evaluation_directory + 'heuristic_mode/'

        result_directory = evaluation_directory + 'lb-from-' + incumbent_mode + '-t_total' + str(
            total_time_limit) + 's' + '-t_node' + str(node_time_limit) + 's' + instance_size + '_lb_k0_regression_rl_beforenode_freq_' + str(freq) + '-' + device_str + '/seed' + str(seed) + '/'
        pathlib.Path(result_directory).mkdir(parents=True, exist_ok=True)

        print(result_directory)

        # Construct the runner that solves each test instance with SCIP plus
        # the ML-based LB primal heuristic.
        scip_with_lb_heuristic = Execute_LB_Regression_RL(instance_type,
                                                          instance_directory,
                                                          solution_directory,
                                                          result_directory,
                                                          lbconstraint_mode=lbconstraint_mode,
                                                          no_improve_iteration_limit=no_improve_iteration_limit,
                                                          seed=seed,
                                                          enable_gpu=enable_gpu,
                                                          freq=freq,
                                                          is_heuristic=is_heuristic,
                                                          incumbent_mode=incumbent_mode,
                                                          regression_model_gnn=regression_model_gnn,
                                                          agent_k=agent_k,
                                                          optim_k=optim_k,
                                                          )

        # The large sizes of the transfer datasets are not part of the evaluation.
        skip_evaluation = (instance_type in TRANSFER_DATASETS
                           and instance_size == instancesizes[1])

        # Solve the whole test set and store the primal bound trajectories.
        if not skip_evaluation:
            scip_with_lb_heuristic.execute_heuristic_baseline(
                total_time_limit=total_time_limit,
                node_time_limit=node_time_limit,
                )
