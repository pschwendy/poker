# Base class for poker bots

import numpy as np
# from torch import nn

from bots.action import Action, ActionType
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
    def forward(self, state: State):
        action = Action()

        if len(state.mini_states[-1].history) == 0:
            action.type = ActionType.RAISE
        else:
            action.type = ActionType.CALL
        
        action.bet = 100

        self.money -= action.bet

        return action
    
