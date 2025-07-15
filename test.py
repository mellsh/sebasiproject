import torch
from env import SnakeEnv
from agent import Agent
import time

def test():
    env = SnakeEnv()
    agent = Agent()

    # 모델 불러오기 (파일명은 학습 시 저장한 파일명과 동일해야 함)
    agent.model.load_state_dict(torch.load('model.pth'))
    agent.model.eval()

    state = env.reset()
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        env.render(fps=10)
        state = next_state
        time.sleep(0.05)

if __name__ == '__main__':
    test()