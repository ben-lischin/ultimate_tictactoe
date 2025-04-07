import random
import time
import math
from uttt import UTTT

EXPLORATION = 400
ITERATIONS = 1000

def reward(winner, player):
    if winner == "T": return 0
    if winner == player: return 1
    return -1

#------------------------ Tree nodes --------------------------#

class Node:
    def __init__(self, state: UTTT, move: tuple[tuple, tuple] | None) -> None:
        self.state = state
        self.visits = 0
        self.val = 0
        self.move_to_here = move
        self.children = []

    def expand_children(self):
        moves = self.state.get_valid_moves()
        self.children = [Node(self.state.make_move(*move), move) for move in moves]

    def update_result(self, reward):
        self.visits += 1
        self.val += reward

def ucb(node: Node, parent_visits):
    if parent_visits == 0 or node.visits == 0: return float('inf')
    exploitation = node.val / node.visits
    exploration = EXPLORATION * math.sqrt(math.log(parent_visits) / node.visits)
    return exploitation + exploration

#------------------------ Phases --------------------------#

def select(node: Node) -> list[Node]:
    path = [node]
    while node.children:
        node = max(node.children, key=lambda n: ucb(n, node.visits))
        path.append(node)
    return path

def expand(node: Node):
    if node.state.game_winner(): return None
    node.expand_children()
    return random.choice(node.children)

def simulate(node: Node, player: str, rollout = random.choice):
    state = node.state
    while not state.game_winner():
        move = rollout(state.get_valid_moves())
        state = state.make_move(*move)
    return reward(state.game_winner(), player)

def back_propogation(path: list[Node], reward):
    for n in path: n.update_result(reward)

#------------------------ Move Prediction --------------------------#

def update_tree(root: Node, player: str):
    path = select(root)
    expanded = expand(path[-1])
    if expanded: path.append(expanded)
    reward = simulate(path[-1], player)
    back_propogation(path, reward)

def predict(state: UTTT):
    root = Node(state, None)
    for _ in range(ITERATIONS): update_tree(root, state.current_player)
    best_node = max(root.children, key=lambda node: node.val)
    return best_node.move_to_here
