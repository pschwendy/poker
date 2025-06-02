import os
import sys
import itertools
import numpy as np

from evaluation.card import Card
from evaluation.evaluate import Evaluator
from rebel.rebel import encode_hand

TEST_HAND = [Card([1, 4]), Card([1, 3])]
TEST_HAND_ENCODED = [c.encode() for c in TEST_HAND]
ENCODED_TEST_HAND_IDX = encode_hand(TEST_HAND_ENCODED)

def hand_wins(board, hand):
    win = np.zeros(26 * 51)
    evaluator = Evaluator()
    
    eval_cards = [Card(c, from_encode=True).get_eval_card() for c in hand]
    eval_board = [Card(c, from_encode=True).get_eval_card() for c in board]
    hand_val = evaluator.evaluate(eval_cards, eval_board)
    
    # possible = TEST_HAND_ENCODED not in board
    # if possible: 
    #     eval_test_hand = [Card(c, from_encode=True).get_eval_card() for c in TEST_HAND_ENCODED]
    #     test_hand_val = evaluator.evaluate(eval_test_hand, eval_board)

    known_cards = set([c for c in hand] + board)
    open_cards = list(set(range(52)) - known_cards)
    remaining_hands = list(itertools.combinations(open_cards, 2))
    idx_remaining_hands = np.array([encode_hand(h) for h in remaining_hands])

    # Assert uniqueness in remaining_hands
    assert len(remaining_hands) == len(set(remaining_hands)), "Duplicate hands found in remaining_hands"
    assert len(idx_remaining_hands) == len(set(idx_remaining_hands)), "Duplicate indices found in idx_remaining_hands"

    eval_hands = [[Card(c, from_encode=True).get_eval_card() for c in h] for h in remaining_hands]
    remaining_vals = np.array([evaluator.evaluate(h, eval_board) for h in eval_hands])

    wins = (remaining_vals > hand_val).astype(float)
    ties = (remaining_vals == hand_val).astype(float)

    win[idx_remaining_hands] = np.where(wins == 1, 1, -1)
    win[idx_remaining_hands[ties == 1]] = 0
        
    # if possible:
    #     i = ENCODED_TEST_HAND_IDX
    #     raw_calc = test_hand_val > hand_val # True if hand beats test hand
    #     print(f"Expected hand {hand} {'beats' if raw_calc else 'loses to'} test hand {TEST_HAND_ENCODED}")
    #     print(f"{hand} vs Test hand {TEST_HAND}: {'Win' if win[i] == 1 else 'Loss' if win[i] == -1 else 'Tie'}")
            

    return win

"""from evaluation/card.py...
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
"""
if __name__ == "__main__":
    three_of_spaces = Card(np.array([1, 1]))
    seven_of_hearts = Card(np.array([5, 2]))
    ten_of_diamonds = Card(np.array([8, 3]))
    king_of_spades = Card(np.array([11, 1]))

    turn_board = [three_of_spaces, seven_of_hearts, ten_of_diamonds, king_of_spades]
    encoded_board = [c.encode() for c in turn_board]

    all_cards = range(52)
    remaining_cards = list(set(all_cards) - set(encoded_board))

    # win lookup: 52 rivers, 26 * 51 encoded hands, 26 * 51 win rates per hand
    win_lookup = np.zeros((52, 26 * 51, 26 * 51))

    for card in remaining_cards:
        print(f"Processing river card: {Card(card, from_encode=True)}")
        river_board = encoded_board + [card]
        re_cards_after_river = list(set(remaining_cards) - set([card]))
        for hand in itertools.combinations(re_cards_after_river, 2):
            # print("==" * 20)
            # print("River board:", [Card(c, from_encode=True) for c in river_board])
            # print(f"Processing hand: {Card(hand[0], from_encode=True)}, {Card(hand[1], from_encode=True)}")
            wins = hand_wins(river_board, hand)
            win_lookup[card, encode_hand(hand)] = wins
            # print(f"Number of wins: {np.sum(wins == 1)}, Ties: {np.sum(wins == 0)}, Losses: {np.sum(wins == -1)}")
            test_hand = [Card([1, 4]), Card([1, 3])]
            encoded_test_hand = [c.encode() for c in test_hand]
            # print(f"Vs {test_hand}: {wins[encode_hand(encoded_test_hand)]}")  # Example hand for testing
            # print("==" * 20)
    
    # Save the win lookup table
    output_dir = "win_lookup"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "win_lookup.npy")
    np.save(output_path, win_lookup)
    print(f"Win lookup table saved to {output_path}")
    print("Win lookup table generated successfully.")