from enum import IntEnum

class ActionType(IntEnum):
    FOLD = 0
    CALL = 1
    RAISE = 2

class Action:
    def __init__(self):
        self.type = ActionType.FOLD
        self.bet = 0