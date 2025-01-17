import torch
import numpy as np
import itertools

from evaluation.evaluate import Evaluator
from evaluation.card import Card
from state.state import Round
from state.state import State

class WpRollout:
    def __init__(self, preflop_win_rates, preflop_tie_rates):
        self.preflop_win_rates = torch.Tensor(preflop_win_rates)
        self.preflop_tie_rates = torch.Tensor(preflop_tie_rates)
        self.flop_win_rates = None
        self.flop_tie_rates = None
        self.player = None
        self.index = None

    def fix(self, p_cards, i):
        self.player = i
        self.index = self.encode_hand([p_cards[0].encode(), p_cards[1].encode()])
        self.p_cards = [c.encode() for c in p_cards]
        self.eval_cards = [c.get_eval_card() for c in p_cards]
        self.flop_win_rates = None
        self.flop_tie_rates = None

    def win_rates(self, board, remaining_cards):
        win_rates = np.zeros(26 * 51)
        tie_rates = np.zeros(26 * 51)
        evaluator = Evaluator()
        
        hand_val = evaluator.evaluate(self.eval_cards, [Card(c).get_eval_card() for c in board])
        eval_board = [Card(c).get_eval_card() for c in board]
        
        remaining_hands = list(itertools.combinations(remaining_cards, 2))
        # remaining_hands = [h.sort() for h in remaining_hands] # enforce order

        eval_hands = [[Card(c, from_encode=True).get_eval_card() for c in h] for h in remaining_hands]
        remaining_vals = np.array([evaluator.evaluate(h, eval_board) for h in eval_hands])

        wins = (remaining_vals > hand_val).astype(float)
        ties = (remaining_vals == hand_val).astype(float)

        encoded_hands = [self.encode_hand(hand) for hand in remaining_hands]

        win_rates[encoded_hands] = wins
        tie_rates[encoded_hands] = ties

        return win_rates, tie_rates 
            
    def compute_flop(self, state: State, player_idx: int):
        board = state.deck.simulate_deal(3) if len(state.table) == 0 else state.table
        encoded_board = [Card(c).encode() for c in board]
        
        remaining_cards = [c for c in range(52) if c not in self.p_cards + encoded_board]
        self.flop_win_rates, self.flop_tie_rates = self.win_rates(board, remaining_cards)
        self.flop_win_rates = torch.Tensor(self.flop_win_rates)
        self.flop_tie_rates = torch.Tensor(self.flop_tie_rates)

    def encode_hand(self, hand):
        return (52 - hand[0]) * (52 - hand[0] - 1) // 2 - hand[1] + hand[0]

    def get_wins(self, state, pi):
        assert self.index is not None, "Fix index of WpRollout before getting win rates"
        self.preflop_win_rates = self.preflop_win_rates.to(pi.device)
        if state.round == Round.PREFLOP:
            return pi * self.preflop_win_rates[self.index]
        else:
            if self.flop_win_rates is None: self.compute_flop(state, self.player)
            return pi * self.flop_win_rates.to(pi.device)
    
    def get_ties(self, state, pi):
        assert self.index is not None, "Fix index of WpRollout before getting tie rates"
        self.preflop_tie_rates = self.preflop_tie_rates.to(pi.device)

        if state.round == Round.PREFLOP:
            return pi * self.preflop_tie_rates[self.index]
        else:
            if self.flop_tie_rates is None: self.compute_flop(state, self.player)
            return pi * self.flop_tie_rates.to(pi.device)