# Base class for poker bots

from typing import Any
import numpy as np
# from torch import nn

from bots.action import Action
from bots.state import State, MiniState
from evaluation.card import Card


"""
Basic strategy:
- Bet with pair
- Probabilistic winning (maybe game theory?)
- Bluffer (inverse of probabilistic winning)
- Fish for a flush
"""

# Not sure if we need to derive from nn.Module
class PokerBase():
    def __init__(self):
        # super(PokerBase, self).__init__()
        # start with 1000 money
        self.money = 1000

        # Change to False when bot folds
        self.play = True

        # 2 cards, each has rank and suit
        self.cards = []
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    @property
    def money(self):
        return self._money
    
    @money.setter
    def money(self, value: int):
        # Should we enforce this?
        self._money = value
    
    @property
    def play(self):
        return self._play
    
    @play.setter
    def play(self, value: bool):
        assert isinstance(value, bool), f"PokerBase.play must be of type 'bool'"
        self._play = value
    
    @property
    def cards(self):
        return self._cards

    @cards.setter
    def cards(self, value: list):
        assert isinstance(value, list), f"PokerBase.cards must be of type 'list'"
        self._cards = value

    def add_card(self, card):
        assert len(self.cards) < 2
        self.cards.append(Card(card))

    def get_eval_cards(self):
        return [self.cards[0].get_eval_card(), self.cards[1].get_eval_card()]
        
    def forward(self, state: State):
        raise NotImplementedError
    
