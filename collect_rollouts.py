import numpy as np
import os

def main():
    win_rates = np.zeros((26 * 51, 26 * 51))
    tie_rates = np.zeros((26 * 51, 26 * 51))

    i = 0
    dirs = [x for x in os.listdir('./rollouts') if x != '.DS_Store']
    dirs = sorted(dirs, key=lambda x: int(x.split('_')[0]))

    for direc in dirs:
        if direc == '.DS_Store':
            continue

        load_win_rates = np.load(f'./rollouts/{direc}/win_rates.npy')
        load_tie_rates = np.load(f'./rollouts/{direc}/tie_rates.npy')

        print(f"Loading {direc} with i={i}")
        print(f"Loaded win rates: {load_win_rates.shape}")

        # Make sure these are disjoint
        assert (win_rates[i: i+200] == 0).all()
        assert (tie_rates[i: i+200] == 0).all()

        assert (load_win_rates[0:i] == 0).all()
        assert (load_win_rates[i+200:] == 0).all()
        assert (load_tie_rates[0:i] == 0).all()
        assert (load_tie_rates[i+200:] == 0).all()

        win_rates += load_win_rates
        tie_rates += load_tie_rates

        i += 200

    special_win_rates = np.load('./special_rollouts/win_rates.npy')
    special_tie_rates = np.load('./special_rollouts/tie_rates.npy')

    assert (special_win_rates[0:599] == 0).all()
    assert (special_win_rates[600:799] == 0).all()
    assert (special_win_rates[800:] == 0).all()
    assert (special_tie_rates[0:599] == 0).all()
    assert (special_tie_rates[600:799] == 0).all()
    assert (special_tie_rates[800:] == 0).all()

    win_rates += special_win_rates
    tie_rates += special_tie_rates

    for i in range(26 * 51):
        if (win_rates[i] == np.zeros(26 * 51)).all():
            print("HI", i)
            print(win_rates[i])
    
    assert (tie_rates != 0).any()

    np.save('./collected/win_rates.npy', win_rates)
    np.save('./collected/tie_rates.npy', tie_rates)

if __name__ == "__main__":
    main()