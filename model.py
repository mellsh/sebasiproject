import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearQNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)

        # batch 여부 확인
        if len(state.shape) == 1:
            # 1차원이면 (1, n) 형태로 reshape
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done, )

        # Q-value 예측
        pred = self.model(state)

        # target 계산
        target = pred.clone()
        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_new += self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][action[idx]] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()