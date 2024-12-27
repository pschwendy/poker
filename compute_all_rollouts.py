import numpy as np
from evaluation.card import Card
from evaluation.evaluator import Evaluator
import itertools
import os
import argparse

class WpRollout:
    def __init__(self, p_cards):
        self.p_cards = [Card(c, from_encode=True) for c in p_cards]
        self.eval_cards = [c.get_eval_card() for c in self.p_cards]

        self.compute()

    def remaining_boards(self, board, remaining_cards):
        board_size = 3
        boards_left = list(itertools.combinations(remaining_cards, board_size))
        return [set(x) for x in boards_left]

    def encode_hand(self, hand):
        return (52 - hand[0]) * (52 - hand[0] - 1) // 2 - hand[1] + hand[0]

    def win_rates(self, board, remaining_cards, future_cards):
        win_rates = np.zeros(26 * 51)
        # return np.ones(26 * 51), (26 * 51)  # for now
        evaluator = Evaluator()

        remaining_boards = self.remaining_boards(board, remaining_cards)

        for board_left in remaining_boards:
            # avoid unnecessary computation of boards that have already been seen
            if preflop and [(card in board_left) for card in future_cards].count(True) >= len(board_left): 
                continue
            elif [(card in board_left) for card in future_cards].count(True) >= len(future_cards): 
                continue
            
            full_board = board + list(board_left)
            eval_board = [Card(c, from_encode=True).get_eval_card() for c in full_board]

            player_eval = evaluator.evaluate(self.eval_cards, eval_board)

            rem_cards = set(remaining_cards).difference(board_left)

            remaining_hands = list(itertools.combinations(rem_cards, 2))
            # remaining_hands = [h.sort() for h in remaining_hands] # enforce order

            eval_hands = [[Card(c, from_encode=True).get_eval_card() for c in h] for h in remaining_hands]
            evals = np.array([evaluator.evaluate(h, eval_board) for h in eval_hands])
            
            # evals = np.array([evaluator.evaluate([Card(c, from_encode=True) for c in h], eval_board) for h in remaining_hands])

            wins = (evals > player_eval).astype(int)
            ties = (evals == player_eval).astype(float) / 2
            encoded_hands = np.array([self.encode_hand(hand) for hand in remaining_hands])

            win_rates[encoded_hands] += wins
            win_rates[encoded_hands] += ties

        return win_rates, len(remaining_boards)

    def compute(self):
        future_cards = []

        board = []
        hand = [card.encode() for card in self.p_cards]
        
        # Pre Flop
        remaining_cards = [c for c in range(52) if c not in hand + board]
        self.preflop_win_rates, preflop_boards = self.win_rates(board, remaining_cards, future_cards, preflop=True)

        print("Preflop computed")
        
        # compute win probabilities given win rates and avoids
        self.preflop_win_rates /= preflop_boards

    def get(self, state, pi):
        return self.preflop_win_rates

def compute_all_rollouts(output_path, index=0):
    os.makedirs(output_path, exist_ok=True)

    hands = list(itertools.combinations(range(52), 2))
    win_rates = np.zeros((26 * 51, 26 * 51)) # opposing win rates for each hand
    
    if index != 0:
        win_rates = np.load(os.path.join(output_path, "win_rates.npy"))
    
    for i in range(index, len(hands)):
        hand = hands[i]
        print(f"Computing hand {i}: {hand}")
        wp = WpRollout(hand)
        win_rates[i] = wp.get()

        np.save(os.path.join(output_path, "win_rates.npy"), win_rates)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output_path", type=str, default="rollouts")
    argparser.add_argument("--index", type=int, default=0)
    
    args = argparser.parse_args()

    compute_all_rollouts(args.output_path, args.index)
