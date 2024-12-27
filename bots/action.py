from enum import IntEnum

class ActionType(IntEnum):
    FOLD = 0
    CALL = 1
    RAISE = 2

class Action:
    def __init__(self, action_type: ActionType = ActionType.FOLD, bet: int = 0):
        self.type = action_type
        self.bet = bet

    def type_str(self):
        if self.type == ActionType.FOLD:
            return "FOLD"
        elif self.type == ActionType.CALL:
            return "CALL"
        elif self.type == ActionType.RAISE:
            return "RAISE"

    def __str__(self):
        return f"Action({self.type_str(self.type)}, {self.bet})"