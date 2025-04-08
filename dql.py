import math
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
from uttt import UTTT


def reward(winner, player):
    if winner == "T": return 0
    if winner == player: return 1
    return -1

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class Memory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) 
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)               
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) 
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 2 * 2, 81) 

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

# set hyperparameters
BATCH = 128
GAMMA = 0.99
INIT_EPS = 0.95
FIN_EPS = 0.05
DECAY_RATE = 0.995
DECAY = 1000
LR = 0.0001

class Agent:
    def __init__(self):
        self.policy_net = Network()
        self.target_net = Network()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = Memory(10000)
        self.steps_done = 0
    
    def get_best_action(self, state, valid_moves):
        """
        Selects an action using an epsilon-greedy policy.
        """
        sample = random.random()
        eps_threshold = FIN_EPS + (INIT_EPS - FIN_EPS) * \
                        math.exp(-1. * self.steps_done / DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                q_vals = self.policy_net(state)
                # map valid moves to unique indices
                uni_ind = [3 * board[0] + board[1] + 9 * (3 * move[0] + move[1]) for board, move in valid_moves]
                # pick valid move with the highest Q-value
                best_index = max(uni_ind, key=lambda i: q_vals[0, i].item())
                best_move = valid_moves[uni_ind.index(best_index)]
                return best_move
        else:
            # randomly move
            return random.choice(valid_moves)
    
    def opt_model(self):
        """
        Samples batch of transitions and performs an optimization step.
        Use Huber loss for stability.
        """
        if len(self.memory) < BATCH:
            return
        trans = self.memory.sample(BATCH)
        # transpose the batch into a Transition of batch-arrays
        batch = Transition(*zip(*trans))
        
        # create a mask for non-final states
        mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        # convert list of integerrs into tensor, need extra dim for gather
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1)
        # convery list of floats to tensor
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
        
        # get Q(s, a)
        state_action_vals = self.policy_net(state_batch).gather(1, action_batch)
        
        # get V(s')
        next_state_vals = torch.zeros(BATCH)
        with torch.no_grad():
            next_state_vals[mask] = self.target_net(next_states).max(1).values
        
        # get expected Q values
        expected_state_action_vals = (next_state_vals * GAMMA) + reward_batch
        
        # get Huber loss
        loss = nn.SmoothL1Loss()(state_action_vals, expected_state_action_vals.unsqueeze(1))
        
        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def new_target_net(self):
        """Updates target net with policy weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decays the exploration rate."""
        self.epsilon = max(FIN_EPS, INIT_EPS * DECAY_RATE)
    
    def store_transition(self, state, action, next_state, reward, done):
        """Store transtion on memory"""
        self.memory.push(state, action, next_state, reward, done)

def get_state(game: UTTT):
    state = []
    for row in range(3):
        for col in range(3):
            subboard = game.subboards[row][col]
            for r in range(3):
                for c in range(3):
                    cell = subboard._query((r, c))
                    state.append(1 if cell == 1 else -1 if cell == 0 else 0)
    state_tensor = torch.tensor(state, dtype=torch.float32)
    return state_tensor.view(1, 1, 9, 9)  # batch_size=1, channels=1, height=9, width=9

def train_dql(num_episodes=1000):
    agent = Agent()
    for episode in range(num_episodes):
        game = UTTT()
        state = get_state(game)
        done = False
        
        while not done:
            valid_moves = game.get_valid_moves()
            # pick action
            action_move = agent.get_best_action(state, valid_moves)
            # convert to unique index
            action_index = 3 * action_move[0][0] + action_move[0][1] + 9 * (3 * action_move[1][0] + action_move[1][1])
            
            next_game = game.make_move(*action_move)
            r = reward(next_game.game_winner(), game.current_player)
            next_state = get_state(next_game)
            done = next_game.game_winner() is not None
            
            agent.store_transition(state, action_index, next_state, r, done)
            agent.opt_model()
            state = next_state
            game = next_game
        
        agent.decay_epsilon()
        # update target net every 10 episodes
        if episode % 10 == 0:
            agent.new_target_net()
        print(f"{episode} completed")
    return agent

def predict(game: UTTT):
    agent = Agent()
    state_tensor = get_state(game)
    valid_moves = game.get_valid_moves()
    return agent.get_best_action(state_tensor, valid_moves)

if __name__ == "__main__":
    trained_dql = train_dql(num_episodes=1000)
    torch.save(trained_dql.policy_net.state_dict(), "dqn.pth")
