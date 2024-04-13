import numpy as np
from bots.actions import Action

class MiniState:
    """
    Mini state of the game.
    Tracks:
        - table
        - history of bets
        - round
        - current player
    """
    def __init__(self, table=np.zeros(5, 2), n_players=6) -> None:
        self.table = table

        # Question - Should we track history as 1d list of bets or 2d history of bet states?
        self.history = [np.zeros(n_players)]
        self.pot = 0
        self.top_bet = 0
        self.index = 0
    
    def update(self, action: Action):
        t = action.type
        b = action.bet

        # Construct bet state from previous state and new bet
        bets = self.history[-1]
        if t == 0: # Fold
            bets[self.index] = 0
        elif t == 1: # Call
            bets[self.index] = self.top_bet
            pot += self.top_bet
        elif t == 2: # Raise
            bets[self.index] = self.top_bet + b
            self.pot += self.top_bet + b
            self.top_bet = self.bets[self.index]
        
        self.history.append(bets)

        self.index += 1
        self.index %= len(self.bets)
    
    def end_round(self) -> None:
        return self.bets[self.index] == self.top_bet


class State:
    """
    State of the game
        - could have mini state for each round
            - field for invidiual mini state (preflop)
            - consider each player's bets
        - pot
        - could have history of actions
    """
    def __init__(self) -> None:
        self.pot = 0
        self.round = 0

        self.mini_states = [MiniState()]

    def finish_round(self, table):
        self.round += 1
        self.mini_states.append(MiniState(table))

    def end_round(self) -> bool:
        return self.mini_states[-1].end_round()
    
    def update(self, action: Action):
        self.mini_states[-1].update(action)
