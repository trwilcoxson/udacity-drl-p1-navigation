import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """Dueling DQN Architecture.

    Separates the Q-value into a state-value stream V(s) and an
    advantage stream A(s,a), then combines them:
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in shared hidden layer
            fc2_units (int): Number of nodes in each stream's hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Shared feature layer
        self.fc1 = nn.Linear(state_size, fc1_units)

        # Value stream
        self.value_fc = nn.Linear(fc1_units, fc2_units)
        self.value_out = nn.Linear(fc2_units, 1)

        # Advantage stream
        self.advantage_fc = nn.Linear(fc1_units, fc2_units)
        self.advantage_out = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> Q-values via value and advantage streams.

        Params
        ======
            state (torch.Tensor): shape (batch_size, state_size)

        Returns
        =======
            q (torch.Tensor): shape (batch_size, action_size) — Q-values for each action
        """
        x = F.relu(self.fc1(state))

        value = F.relu(self.value_fc(x))
        value = self.value_out(value)

        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage_out(advantage)

        # Combine: Q = V + (A - mean(A))
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
