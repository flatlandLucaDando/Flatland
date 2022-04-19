from enum import IntEnum
from typing import NamedTuple
from flatland.core.grid.grid4 import Grid4TransitionsEnum

class RailEnvActions(IntEnum):
    DO_NOTHING = 0  # implies change of direction in a dead-end!
    MOVE_LEFT = 1
    MOVE_FORWARD = 2
    MOVE_RIGHT = 3
    STOP_MOVING = 4
    REVERSE = 5
    ACCELERATE = 6
    DECELERATE = 7

    @staticmethod
    def to_char(a: int):
        return {
            0: 'B',
            1: 'L',
            2: 'F',
            3: 'R',
            4: 'S',
            5: 'I',
            6: 'A',
            7: 'D',
        }[a]

    @classmethod
    def is_action_valid(cls, action):
        return action in cls._value2member_map_

    def is_moving_action(self):
        return self.value in [self.MOVE_RIGHT, self.MOVE_LEFT, self.MOVE_FORWARD, self.REVERSE]
    
    def is_action_speed(self):
        return self.value in [self.ACCELERATE, self.DECELERATE]


RailEnvGridPos = NamedTuple('RailEnvGridPos', [('r', int), ('c', int)])
RailEnvNextAction = NamedTuple('RailEnvNextAction', [('action', RailEnvActions), ('next_position', RailEnvGridPos),
                                                     ('next_direction', Grid4TransitionsEnum)])
