# Apply deep mccfr minimization
# Allow neural network to learn abstractions of the game through embeddings

import torch.nn as nn
from bots.base import PokerBase
import bots.action as action
import bots.state as state
from typing import Any

class FC(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(FC, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.fc(x))


class PolicyNetwork(nn.Module):
    def __init__(self, dim: int = 64):
        super(PolicyNetwork, self).__init__()

        # Embedding layers
        # 52 cards in a deck
        self.card_embedding = nn.Embedding(52, dim)
        self.yield_embedding = nn.Linear(3, dim)
        
        # Action history
        self.action_embedding = nn.Embedding(3, dim)

        # 2 layer LSTM for action history
        self.lstm = nn.LSTM(dim, dim, 2)

        # Fully connected layers
        self.fc1 = FC(dim, dim)
        self.fc2 = FC(dim, dim)
        self.fc3 = FC(dim, dim)

        # Policy head
        self.policy_head = nn.Linear(dim, 3)

        # Value head
        self.value_head = nn.Linear(dim, 2)


    def forward(self, state: State) -> Action:
        """
        Deep MCCFR forward pass
        General approach:
        1. Create abstraction of current state
            - Embed cards
            - Project yield (current pot, current bet, current stack)
            - Abstract action history
        2. Pass abstraction to neural network
        3. Get output from two heads: policy and value
        4. Use policy to select action from (Fold, Call, Raise)
        5. Sample action from policy distribution
            - If action is Raise, use value (mean, var) 
                to create distribution of raise amounts 
                (normal distribution)
            - Sample discrete raise amount from distribution with floor
                of continuously sampled raise amount
        """
        pass

class MCCFR(PokerBase):
    def __init__(self, start_money: int = 10000):
        super(MCCFR, self).__init__(start_money)

        self.policy = PolicyNetwork()

    def forward(self, state: State) -> Action:
       
        # TODO: Implement forward pass
        pass
        
