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
from bots.state import State, Round

import yaml

class PokerGame:
    def __init__(self, bots):
        self.bots = bots
        self.deck = Deck()
        self.deck._shuffle()
        
        self.table = []
        self.state = State()

        self.small_blind = 1
        self.big_blind = 2
    
    def deal(self):
        for i in range(2):
            for bot in self.bots:
                bot.add_card(self.deck.deal_one(), i)
    
    def betting_round(self):
        index = 0
        while not self.state.end_round():
            index += 1
            index %= len(self.bots)

            bot = self.bots[index]

            if not bot.play:
                continue

            action = bot.forward(self.state)
            
            if action[0] == 0: # Fold
                bot.play = False
            
            self.state.update(action)
        
    def next_round(self):
        # Finish round
        if self.state.round == Round.PREFLOP:
            table += self.deck.deal(3)
        else:
            table += self.deck.deal()
        
    
    def play(self):
        for bot in self.bots:
            bot.play = True
        
        self.deal()
        self.betting_round()
        
        self.next_round()
        self.betting_round()
        
        self.next_round()
        self.betting_round()
        
        self.next_round()
        self.betting_round()
        
        # Evaluate winner
        eval_cards = [bot.get_eval_card() for bot in self.bots]
        eval_table = EvaluationCard.new(self.table)
        
        for bot, eval_card in zip(self.bots, eval_cards):
            eval_card.evaluate(eval_table)
        
        # Update money
        winner = max(self.bots, key=lambda bot: bot.eval_card)
        winner.money += self.state.pot
        
        for bot in self.bots:
            if bot != winner:
                bot.money -= self.state.pot
        
        self.state = State()
        self.table = []
        
        return winner
    

def objective_function():
    pass

def train():
    pass

def main():
    pass

if __name__ == "__main__":
    main()
    