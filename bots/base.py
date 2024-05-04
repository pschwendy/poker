# Base class for poker bots

import numpy as np
from torch import nn

from actions import Action
from state import State, MiniState

"""
Basic strategy:
- Bet with pair
- Probabilistic winning (maybe game theory?)
- Bluffer (inverse of probabilistic winning)
- Fish for a flush
"""

class PokerBase(nn.Module):
    def __init__(self):
        super(PokerBase, self).__init__()
        # start with 1000 money
        self.money = np.array(1000)

        # Change to False when bot folds
        self.play = True

        # 2 cards, each has rank and suit
        self.cards = np.zeros((2, 2))

    def add_card(self, card, index):
        self.cards[index] = card

    def forward(self, x: State):
        """
        x: State
        """
        return Action()
    
