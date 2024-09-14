import numpy as np
from bots.action import Action
from enum import IntEnum

# class syntax
class Round(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

class Turn:
    def __init__(self, top_bet):
        # self.actions = [] # List of actions on table during player turn
        self.action = Action()
        self.top_bet = top_bet
        # self.index = 0
    
    def update(self, action: Action) -> int:        
        self.action = action

        # self.index += 1
        # self.index %= len(self.bets)

        t = action.type
        if t == 0: # Fold
            return 0
        elif t == 1: # Call
            return self.top_bet
        elif t == 2: # Raise
            b = action.bet
            return b

class MiniState:
    """
    Mini state of the game.
    Tracks:
        - table
        - history of bets
        - round
        - current player
    """
    def __init__(self, table, players_left, n_players) -> None:
        self.table = table # cards on table

        # Question - Should we track history as 1d list of bets or 2d history of bet states?
        self.history = [] # List of Turns
        self.n_players = n_players
        self.players_left = players_left
        self.folded = 0
        self.pot = 0
        self.top_bet = 0
    
    def update(self, action: Action) -> None:
        self.history.append(Turn(self.top_bet))
        
        bet = self.history[-1].update(action)
        self.pot += bet
        self.top_bet = max(self.top_bet, bet)

        if action.type == 0: self.folded += 1
        if action.type == 2: self.update_backlog()
    
    def update_backlog(self) -> None:
        self.players_left -= self.folded
        self.folded = 0

    def raised_once(self) -> int:
        return [turn.action.type == 2 for turn in self.history[-self.players_left:]].count(True)
    
    def end_round(self) -> bool:
        return len(self.history) >= self.players_left and self.raised_once() # some logic to determine if round is over


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

        self.total_players = 6
        self.active = 6 # number of active players
        self.mini_states = [MiniState(np.zeros((5, 2)), self.active, self.total_players)]
        self.round = Round.PREFLOP

    def finish_round(self, table):
        self.round += 1
        self.pot += self.mini_states[-1].pot
        self.mini_states.append(MiniState(table, self.active, self.total_players))

    def end_round(self) -> bool:
        return self.mini_states[-1].end_round()
    
    def update(self, action: Action):
        if action.type == 0: self.active -= 1 # for next mini state
        self.mini_states[-1].update(action)
