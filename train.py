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
from bots.simple import SimpleBot
from bots.state import State, Round
from evaluation.card import Card
from evaluation.eval_card import EvaluationCard
from evaluation.lookup import LookupTable
from evaluation.evaluate import Evaluator

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

            if action.type == 2: # Raise
                print(f"Bot {index} raised {action.bet}")
            elif action.type == 1: # Call
                print(f"Bot {index} called {action.bet}")
            else: # Fold
                print(f"Bot {index} folded")
            
            if action.type == 0: # Fold
                bot.play = False
            
            self.state.update(action)
        
    def next_round(self):
        # Finish round
        if self.state.round == Round.PREFLOP:
            self.table += self.deck.deal(3)
        else:
            self.table += self.deck.deal()
        self.state.finish_round(table=self.table)
        print([Card(c) for c in self.table])
        
    
    def play(self):
        print("=========Starting game=========")
        for bot in self.bots:
            bot.play = True
        
        self.deal()
        self.betting_round()
        self.next_round()
        print("Pot is", self.state.pot)

        print("==============Flop==============")
        
        self.betting_round()
        self.next_round()
        print("Pot is", self.state.pot)

        print("==============Turn==============")
        
        self.betting_round()
        self.next_round()
        print("Pot is", self.state.pot)

        print("=============River==============")
        
        self.betting_round()
        self.state.finish_round(self.table)
        print("Pot is", self.state.pot)

    
        # Evaluate winner
        eval_cards = [bot.get_eval_cards() for bot in self.bots if bot.play]
        eval_table = [Card(c).get_eval_card() for c in self.table]

        
        evals = [Evaluator().evaluate(eval_card, eval_table) for eval_card in eval_cards]


        print("=========Summary=========")
        print("Bots had the following cards:")
        print([bot.get_cards() for bot in self.bots])

        print("The winner had:")
        print(self.bots[evals.index(min(evals))].get_cards())

        # Update money
        win_index = evals.index(min(evals))
        print(f"Bot {win_index} wins the pot")
        self.bots[win_index].money += self.state.pot
        
        self.state = State()
        self.table = []

        for i, bot in enumerate(self.bots):
            print(f"Bot {i} has {bot.money}")
        
        return self.bots[win_index]
    

def objective_function():
    pass

def train():
    pass

def main():
    bots = [SimpleBot() for _ in range(6)]
    game = PokerGame(bots)
    game.play()

if __name__ == "__main__":
    main()
    