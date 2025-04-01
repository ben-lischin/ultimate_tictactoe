import random
import time
import collections
from uttt import UTTT

REWARDS = { 
    "X": 1,
    "O": -1,
    "T": 0,
    None: 0,
}

def rollout(moves): return random.choice(moves)

def simulate(state: UTTT):
    """
    Returns the winner of the given game played to completion.
    Moves are made based on the rollout function
    """
    while not state.game_winner():
        move = rollout(state.get_valid_moves())
        state = state.make_move(*move)
    return state.game_winner()
