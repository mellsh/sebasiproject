import torch
import random
import numpy as np
from collections import deque
from model import LinearQNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # 탐험률
        self.gamma = 0.9  # 할인률
        self.memory = deque(maxlen=MAX_MEMORY)  # 경험 저장소
        self.model = LinearQNet(input_size=11, hidden_size=64, output_size=3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, env):
        return env._get_state()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        print("📥 상태 벡터:", state)

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            print("🎲 랜덤 행동:", move)
        else:
            state0 = torch.tensor(state, dtype=torch.float32)
            if len(state0.shape) == 1:
                state0 = state0.unsqueeze(0)  # (4,) -> (1, 4)
            prediction = self.model(state0)
            print("🤖 모델 예측:", prediction)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return move