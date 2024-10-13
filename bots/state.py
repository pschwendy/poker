import numpy as np
from bots.action import Action, ActionType
from enum import IntEnum

# class syntax
class Round(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

class Turn:
    def __init__(self, top_bet):
        self.action = Action()
        self.top_bet = top_bet
    
    def update(self, action: Action) -> int:        
        self.action = action

        t = action.type
        if t == ActionType.FOLD:
            return 0
        elif t == ActionType.CALL:
            return self.top_bet
        elif t == ActionType.RAISE:
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

        self.history = [] # List of Turns
        self.n_players = n_players
        
        # We know the round is over when folded + called = players_left
        self.players_left = players_left
        self.folded = 0
        self.called = 0

        self.pot = 0
        self.top_bet = 0
    
    def update(self, action: Action) -> None:
        """Update the state of the game with the given action"""
        self.history.append(Turn(self.top_bet))
        
        bet = self.history[-1].update(action)
        self.pot += bet
        self.top_bet = max(self.top_bet, bet)

        if action.type == ActionType.FOLD: self.folded += 1
        if action.type == ActionType.CALL: self.called += 1
        if action.type == ActionType.RAISE: self.update_backlog()
    
    def update_backlog(self) -> None:
        """Reset folded and called players"""
        self.players_left -= self.folded
        self.folded = 0
        self.called = 0
    
    def end_round(self) -> bool:
        """Called every time a player makes an action to check if the round is over"""
        return self.called + self.folded == self.players_left


class State:
    """
    State of the game
        - could have mini state for each round
            - field for invidiual mini state (preflop)
            - consider each player's bets
        - pot
        - could have history of actions
    """
    def __init__(self, n_players=6) -> None:
        self.pot = 0

        self.total_players = n_players
        self.active = n_players

        self.mini_states = [MiniState(np.zeros((5, 2)), self.active, self.total_players)]
        self.round = Round.PREFLOP

    def finish_round(self, table):
        """Called at end of each round. Updates the pot and creates a new mini state"""
        self.round += 1
        self.pot += self.mini_states[-1].pot
        self.mini_states.append(MiniState(table, self.active, self.total_players))

    def end_round(self) -> bool:
        return self.mini_states[-1].end_round()
    
    def update(self, action: Action):
        if action.type == 0: self.active -= 1 # for next mini state
        self.mini_states[-1].update(action)
