#############################################
# General state file for all poker variants #
#############################################

import torch
import numpy as np
from bots.action import Action, ActionType
from enum import IntEnum
from deck import Deck
from evaluation.card import Card

START_MONEY = 10000

class Round(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

class BotState:
    def __init__(self) -> None:
        self._play = True
        self._money = START_MONEY
        self._total_bet = 0
        self._current_bet = 0

    @property
    def round_money(self) -> int:
        return self._money + self._current_bet
    @property
    def play(self) -> bool:
        return self._play
    
    @play.setter
    def play(self, value: bool) -> None:
        self._play = value

    @property
    def money(self) -> int:
        return self._money
    
    @money.setter
    def money(self, value: int) -> None:
        self._money = value
    
    @property
    def total_bet(self) -> int:
        return self._total_bet

    @total_bet.setter
    def total_bet(self, value: int) -> None:
        self._total_bet = value
    
    @property
    def current_bet(self) -> int:
        return self._current_bet
    
    @current_bet.setter
    def current_bet(self, value: int) -> None:
        self._current_bet = value 
    
    def build_stack(self) -> torch.Tensor:
        return torch.Tensor([self.money, self.total_bet, self.current_bet]) / START_MONEY
    
    def __str__(self) -> str:
        return f"BotState(play={self.play}, money={self.money}, total_bet={self.total_bet}, current_bet={self.current_bet})"

class MiniState:
    """
    Mini state of the game.
    Tracks:
        - table
        - history of bets
        - round
        - current player
    """
    def __init__(self, table, players_left, n_players, max_round_size) -> None:
        self.table = table # cards on table

        self.action_history = [] # List of Actions

        self.n_players = n_players
        
        # We know the round is over when folded + called = players_left
        self.players_left = players_left
        self.folded = 0
        self.other = 0
        self.num_raises = 0

        self.pot = 0
        self.top_bet = 0
        self.max_round_size = max_round_size

    def encode_single_action(self, action) -> np.array:
        """Encodes the action history (formatted for BrownNet)"""
        if action.type == ActionType.FOLD: return np.array([0])
        if action.type == ActionType.CALL: return np.array([0])
        if action.type == ActionType.RAISE: return np.array([action.bet])
            
    def encode_action(self) -> np.array:
        """Encodes the action history"""
        return np.array([self.encode_single_action(action) for action in self.action_history] + [np.zeros(1) for _ in range(self.max_round_size - len(self.action_history))])

    def update(self, action: Action, bots, curr_player, global_pot, blind=False) -> None:
        """Update the state of the game with the given action"""

        if action.type == ActionType.CALL: 
            difference = action.bet - bots[curr_player].current_bet
            bots[curr_player].current_bet = action.bet
            bots[curr_player].total_bet += difference
            bots[curr_player].money -= difference
            action.bet = difference / global_pot
            self.pot += difference
        elif action.type == ActionType.RAISE:
            bots[curr_player].current_bet += action.bet
            bots[curr_player].total_bet += action.bet
            bots[curr_player].money -= action.bet
            self.top_bet = max(self.top_bet, bots[curr_player].current_bet)
            self.pot += action.bet
            action.bet /= global_pot  
        
        self.action_history.append(action)

        # blinds do not count as part of the round
        if blind:
            return

        if action.type == ActionType.FOLD: 
            bots[curr_player].play = False
            self.folded += 1
        elif action.type == ActionType.CALL: self.other += 1
        elif action.type == ActionType.RAISE: self.update_backlog()
    
    def update_backlog(self) -> None:
        """Reset folded and called players"""
        self.players_left -= self.folded
        self.folded = 0
        self.other = 1
        self.num_raises += 1
    
    def end_round(self) -> bool:
        """Called every time a player makes an action to check if the round is over"""
        return self.other + self.folded == self.players_left 
    
    def __str__(self) -> str:
        build_str = f"MiniState: pot={self.pot}, top_bet={self.top_bet}\n"
        build_str += f"Table: {self.table}\n"
        build_str += f"Action history: \n"
        for i, action in enumerate(self.action_history):
            build_str += f"Player {(self.start_player + i) % self.total_players}: {bot.__str__()}\n"
        
        return build_str

class State:
    """
    State of the game
        - could have mini state for each round
            - field for invidiual mini state (preflop)
            - consider each player's bets
        - pot
        - could have history of actions
    """
    def __init__(self, 
        n_players=6,
        num_rounds=2, # 2 for FHP
        max_round_size=7 # 7 for FHP
    ) -> None:
        self.pot = 0
        self.start_player = np.random.randint(0, n_players)
        self.curr_player = self.start_player
        self.num_rounds = num_rounds
        self.max_round_size = max_round_size

        self.total_players = n_players
        self.active = n_players

        self.bots = [BotState() for _ in range(n_players)]

        self.table = []
        self.mini_states = [MiniState(self.table, self.active, self.total_players, self.max_round_size)]
        self.round = Round.PREFLOP
        self.deck = Deck()
        self.deck.reset()

        # little
        self.mini_states[-1].update(Action(ActionType.RAISE, 50), self.bots, self.curr_player, 150, blind=True)
        self.curr_player = (self.curr_player + 1) % self.total_players
        self.pot += 50

        # big
        self.mini_states[-1].update(Action(ActionType.RAISE, 100), self.bots, self.curr_player, 150, blind=True)
        self.curr_player = (self.curr_player + 1) % self.total_players
        self.pot += 100

        self.depth = 0

    def board_size(self):
        return self.num_rounds + 1

    def finish_round(self):
        """Called at end of each round. Updates the pot and creates a new mini state"""
        self.reset_money()

        if self.round == self.num_rounds - 1:
            return

        if self.round == Round.PREFLOP:
            self.table = self.deck.deal(3)
            self.round += 1
        elif self.round == Round.FLOP:
            self.table += self.deck.deal(1)
            self.round += 1
        elif self.round == Round.TURN:
            self.table += self.deck.deal(1)
            self.round += 1

        self.mini_states.append(MiniState(self.table, self.active, self.total_players, self.max_round_size))
            
        self.curr_player = self.start_player

    def deal_player(self):
        return self.deck.deal_one()
    
    def get_top_bet(self):
        return self.mini_states[-1].top_bet

    def is_terminal(self) -> bool:
        return (self.round == (self.num_rounds - 1)  and self.end_round()) or self.one_player_left() or len(self.mini_states) > self.num_rounds

    def one_player_left(self) -> bool:
        return sum([bot.play for bot in self.bots]) == 1

    def end_round(self) -> bool:
        return self.mini_states[-1].end_round()

    def to_dict(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        encoded_actions = [torch.Tensor(mini_state.encode_action()).to(device) for mini_state in self.mini_states]
        empty_actions = [torch.zeros(self.max_round_size, 1).to(device) for _ in range(self.num_rounds - len(self.mini_states))]
        return {
            "h_action": torch.stack(encoded_actions + empty_actions).view(-1),
            # encode cards between 0-51, -1 for unknown
            "cards": [Card(c).encode() for c in self.table] + [-1 for _ in range(self.board_size() - len(self.table))],
        }

    def reset_money(self):
        for player in self.bots:
            player.current_bet = 0
    
    def next_player(self):
        self.curr_player = (self.curr_player + 1) % self.total_players
        while not self.bots[self.curr_player].play:
            self.curr_player = (self.curr_player + 1) % self.total_players
            
    def update(self, action: Action):
        if action.type == 0: self.active -= 1 # for next mini state
        # update state pot with ministate pot difference after update
        self.pot -= self.mini_states[-1].pot
        self.mini_states[-1].update(action, self.bots, self.curr_player, self.pot + self.mini_states[-1].pot)
        self.pot += self.mini_states[-1].pot
        self.curr_player = (self.curr_player + 1) % self.total_players

        if self.end_round(): self.finish_round()

        self.depth += 1

    def round_to_str(self):
        return ["PREFLOP", "FLOP", "TURN", "RIVER"][int(self.round)]

    def print_history(self):
        print("State history:")
        for i, mini_state in enumerate(self.mini_states):
            print(f"Round {i}: {mini_state.__str__()}")
            
    def __str__(self) -> str:
        build_str = f"Sate: pot={self.pot}, round={self.round_to_str()}\n"
        for i, bot in enumerate(self.bots):
            build_str += f"Player {i}: {bot.__str__()}\n"
        
        build_str += f"Table: {self.table}\n"
        build_str += f"Top bet: {self.get_top_bet()}\n"
        return build_str