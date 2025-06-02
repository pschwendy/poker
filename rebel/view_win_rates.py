import numpy as np

win_lookup = np.load("win_lookup/win_lookup.npy")

for win in win_lookup:
    for hand in win:
        if hand.size == 0:
            continue
        wins = np.where(hand == 1, True, False)
        ties = np.where(hand == 0, True, False)
        losses = np.where(hand == -1, True, False)

        print(f"Hand: {hand}, Wins: {np.sum(wins)}, Ties: {np.sum(ties)}, Losses: {np.sum(losses)}")