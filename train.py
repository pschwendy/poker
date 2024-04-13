"""
(1) Initialize 6 bots
(2) Create a deck
(3) Shuffle the deck
(4) Deal 2 cards to each bot
(5) First round actions from each bot
(6) Deal 3 cards to the table
(7) Second round actions from each bot
(8) Deal 1 card to the table
(9) Third round actions from each bot
(10) Deal 1 card to the table
(11) Fourth round actions from each bot
(12) Evaluate the winner
(13) Update the money of each bot
(14) Repeat steps 2-13
(15) Objective function
(16) Train the bots
"""

import numpy as np
import itertools

from deck import Deck
from bots.base import PokerBase

import yaml

class PokerGame:
    def __init__(self, bots):
        self.bots = bots
        self.deck = Deck()
        self.deck._shuffle()
        
        self.table = []
        self.playback = []

        self.pot = 0
        self.small_blind = 1
        self.big_blind = 2
        self.round = 0
    
    def deal(self, n):
        for i in range(2):
            for bot in self.bots:
                bot.add_card(self.deck.deal_one(), i)
    
    def betting_round(self):
        top_bet = np.array([0])

        for bot in self.bots:
            if not bot.play:
                continue

            action = bot.forward(self.playback, top_bet)
            
            if action[0] == 0: # Fold
                bot.play = False
            elif action[0] == 2: # Call
                self.pot += action[1]
            elif action[0] == 2: # Raise
                self.pot += action[1]
                top_bet = action[1]
            
            self.playback.append(action)


    