from env import SnakeEnv
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np

def plot(scores, mean_scores):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.legend(['Score', 'Mean Score'])
    plt.pause(0.1)

def train():
    print("train start")
    try:
        env = SnakeEnv()
        agent = Agent()
        
        scores = []
        mean_scores = []
        total_score = 0
        print("✅ 환경과 에이전트 생성 완료")
    except Exception as e:
        print("❌ 초기화 중 오류 발생:", e)
        return


    for game in range(1, 1001):  # 에피소드 수
        print(f"game {game}시작")
        state = env.reset()
        print("🔍 state shape:", np.shape(state))
        done = False
        score = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            # 단기 학습
            agent.train_short_memory(state, action, reward, next_state, done)

            # 경험 저장
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            env.render(fps=30)
            score += reward

            # 시각화 (선택)
            # env.render()

        # 게임 끝나면 장기 학습
        agent.train_long_memory()
        agent.n_games += 1

        scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        mean_scores.append(mean_score)

        print(f"Game {agent.n_games} → Score: {score:.1f} | Mean: {mean_score:.2f}")

        if agent.n_games % 20 == 0:
            plot(scores, mean_scores)

if __name__ == '__main__':
    train()