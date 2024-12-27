from evaluation.eval_card import EvaluationCard
import numpy as np


class Card:
    _INT_RANK_TO_STR = {
        0: "Two",
        1: "Three",
        2: "Four",
        3: "Five",
        4: "Six",
        5: "Seven",
        6: "Eight",
        7: "Nine",
        8: "Ten",
        9: "Jack",
        10: "Queen",
        11: "King",
        12: "Ace"
    }

    _INT_SUIT_TO_STR = {
        1: "Spades",
        2: "Hearts",
        3: "Diamonds",
        4: "Clubs"
    }
    
    def __init__(self, card, from_encode=False):
        if from_encode:
            self._card = np.array([card % 13, card // 13 + 1])
        else:
            self._card = card

    def __str__(self):
        rank = self._INT_RANK_TO_STR[self._card[0]]
        suit = self._INT_SUIT_TO_STR[self._card[1]]
        return f"{rank} of {suit}"

    def __repr__(self):
        rank = self._INT_RANK_TO_STR[self._card[0]]
        suit = self._INT_SUIT_TO_STR[self._card[1]]
        return f"{rank} of {suit}"

    def rank(self):
        return self._card[0]

    def suit(self):
        return self._INT_SUIT_TO_STR[self._card[1]]
    
    def encode(self):
        return int(self._card[0] + 13 * (self._card[1] - 1))
    
    def __eq__(self, other):
        return self._card == other._card
    
    def to_numpy(self):
        return np.array(self._card).astype(int)

    def get_eval_card(self):
        return EvaluationCard.new(self.to_numpy())