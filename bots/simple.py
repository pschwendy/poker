# Base class for poker bots

import numpy as np
from torch import nn

from bots.action import Action
from bots.state import State, MiniState
from bots.base import PokerBase

"""
Basic strategy:
- Bet with pair
- Probabilistic winning (maybe game theory?)
- Bluffer (inverse of probabilistic winning)
- Fish for a flush
"""

class SimpleBot(PokerBase):
    def __init__(self):
        super(SimpleBot, self).__init__()

    def add_card(self, card, index):
        self.cards[index] = card

    def forward(self, x: State):
        """
        x: State
        """
        action = Action()

        if len(x.mini_states[-1].history) == 0:
            action.type = 2
        else:
            action.type = 1
        
        action.bet = 100

        self.money -= action.bet

        return action
    
