from eval_card import EvaluationCard
import numpy as np


class Card:
    _INT_SUIT_TO_STR = {
        0: "Spades",
        1: "Hearts",
        2: "Diamonds",
        3: "Clubs"
    }
    
    def __init__(self, rank, suit):
        self._card = np.array([rank, suit])

    def __str__(self):
        suit = self._INT_SUIT_TO_STR[self._card[1]]
        return f"{self._card[0]} of {suit}"

    def __repr__(self):
        suit = self._INT_SUIT_TO_STR[self._card[1]]
        return f"{self._card[0]} of {suit}"

    def rank(self):
        return self._card[0]

    def suit(self):
        return self._INT_SUIT_TO_STR[self._card[1]]
    
    def __eq__(self, other):
        return self._card == other._card
    
    def get_eval_card(self):
        return EvaluationCard.new(self._card)