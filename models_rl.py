"""The RL policy models and agents for adapting k and t during LB search."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset
from torch.distributions import Categorical
import gzip
import pickle


class SimplePolicy(nn.Module):
    """Linear policy network mapping the LB state to action logits.

    Used for both the k-policy and the t-policy: the input is the
    7-dimensional LB state and the output is one logit per action.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.linear1 = torch.nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=1)

    def forward(self, x):

        output = self.linear1(x)
        return output


class Agent:
    """Base agent that evaluates the policy network on a state."""

    def __init__(self, policy, device, greedy=True):
        self.policy = policy
        self.device = device
        self.greedy = greedy

    def select_action(self, state):
        """Return the action logits of the policy for a numpy state vector."""
        state = torch.from_numpy(state).float().view(1, -1)

        if self.device is not None:
            state = state.to(self.device)

        preds = self.policy(state)
        return preds


class AgentReinforce(Agent):
    """REINFORCE agent with epsilon-greedy exploration.

    In non-greedy mode the action is sampled from the categorical
    distribution given by the policy logits (or uniformly at random with
    probability epsilon); log-probabilities and rewards are recorded for
    the policy-gradient update. In greedy mode the argmax action is chosen.
    """

    def __init__(self, policy, device, greedy, opt=None, epsilon=0):
        super().__init__(policy, device, greedy)
        self.opt = opt
        self.epsilon = torch.tensor(epsilon, dtype=torch.float)

        self.rewards = []
        self.log_probs = []

    def select_action(self, state):
        preds = super().select_action(state)
        if not self.greedy:
            probs = F.log_softmax(preds, dim=1)
            m = Categorical(logits=probs)

            # Explore with probability epsilon (or when the policy output
            # degenerates to NaN): sample an action uniformly at random.
            if torch.bernoulli(self.epsilon) == 1 or torch.isnan(probs.exp().sum()):
                random_choice = torch.ones(m._num_events)
                if self.device is not None:
                    random_choice = random_choice.to(self.device)

                m_rand = Categorical(random_choice)
                action = m_rand.sample()
            else:
                action = m.sample()

            if self.opt is not None:
                self.log_probs.append(m.log_prob(action))
            action = action.item()
        else:
            action = preds.argmax(1, keepdim=False).item()
        return action


class ImitationLbDataset(Dataset):
    """Dataset of (LB state, expert action) samples for imitation learning."""

    def __init__(self, sample_files, transform=None):
        self.sample_files = sample_files

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)
        state, label = sample

        state = torch.FloatTensor(state).view(-1)
        label = torch.LongTensor(np.array(label).reshape(-1))
        return state, label
