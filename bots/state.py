import numpy as np
from bots.action_pb2 import Action
from enum import Enum

# class syntax
class Round(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

class Turn:
    def __init__(self):
        self.actions = [] # List of actions on table during player turn
        self.top_bet = 0
        self.index = 0
    
    def update(self, action: Action) -> int:        
        self.actions.append(action)

        self.index += 1
        self.index %= len(self.bets)

        t = action.type
        if t == 0: # Fold
            return 0
        elif t == 1: # Call
            return self.top_bet
        elif t == 2: # Raise
            b = action.bet
            return self.top_bet + b

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
        self.table = table # cards on table

        # Question - Should we track history as 1d list of bets or 2d history of bet states?
        self.history = [] # List of Turns
        self.pot = 0
    
    def update(self, action: Action) -> None:
        self.history.append(Turn())
        self.pot += self.history[-1].update(action)
    
    def end_round(self) -> bool:
        return True # some logic to determine if round is over


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

        self.mini_states = [MiniState()]
        self.round = Round.PREFLOP

    def finish_round(self, table):
        self.round += 1
        self.mini_states.append(MiniState(table))

    def end_round(self) -> bool:
        return self.mini_states[-1].end_round()
    
    def update(self, action: Action):
        self.mini_states[-1].update(action)
