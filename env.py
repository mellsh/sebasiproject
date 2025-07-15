import pygame
import random
import numpy as np
from snake import snake, cube, drawGrid

ROWS = cube.rows
WIDTH = cube.w


class SnakeEnv:
    def __init__(self):
        pygame.init()
        self.win = pygame.display.set_mode((WIDTH, WIDTH))
        pygame.display.set_caption("RL Snake")
        self.clock = pygame.time.Clock()

        self.snake = None
        self.snack = None

    def reset(self):
        self.snake = snake((255, 0, 0), (10, 10))  # 시작 위치
        self.snack = self._place_snack()
        return self._get_state()

    def step(self, action):
        """
        action: 0 = left, 1 = straight, 2 = right
        """
        self._apply_action(action)
        self.snake.move()

        reward = -0.1
        done = False
        head = self.snake.head.pos

        # 충돌 체크
        if head in list(map(lambda x: x.pos, self.snake.body[1:])):
            reward = -10
            done = True
            return self._get_state(), reward, done

        # 먹이 먹음
        if head == self.snack.pos:
            self.snake.addCube()
            self.snack = self._place_snack()
            reward = 10

        return self._get_state(), reward, done

    def render(self, fps=10):
        self.clock.tick(fps)
        self.win.fill((0, 0, 0))
        self.snake.draw(self.win)
        self.snack.draw(self.win)
        drawGrid(WIDTH, ROWS, self.win)
        pygame.display.update()

    def _place_snack(self):
        positions = list(map(lambda z: z.pos, self.snake.body))
        while True:
            x, y = random.randrange(ROWS), random.randrange(ROWS)
            if (x, y) not in positions:
                return cube((x, y), color=(0, 255, 0))

    def _apply_action(self, action):
        # 방향 전환 처리 (상대적 회전)
        dirs = {
            (1, 0): {"left": (0, -1), "right": (0, 1)},
            (-1, 0): {"left": (0, 1), "right": (0, -1)},
            (0, 1): {"left": (1, 0), "right": (-1, 0)},
            (0, -1): {"left": (-1, 0), "right": (1, 0)},
        }

        dirnx, dirny = self.snake.dirnx, self.snake.dirny
        if action == 0:
            nx, ny = dirs[(dirnx, dirny)]["left"]
        elif action == 2:
            nx, ny = dirs[(dirnx, dirny)]["right"]
        else:
            nx, ny = dirnx, dirny

        self.snake.dirnx, self.snake.dirny = nx, ny
        self.snake.turns[self.snake.head.pos[:]] = [nx, ny]

    def _get_state(self):
        head = self.snake.head.pos
        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)

        dir_l = self.snake.dirnx == -1
        dir_r = self.snake.dirnx == 1
        dir_u = self.snake.dirny == -1
        dir_d = self.snake.dirny == 1

        danger_straight = (
            (dir_r and point_r in list(map(lambda x: x.pos, self.snake.body))) or
            (dir_l and point_l in list(map(lambda x: x.pos, self.snake.body))) or
            (dir_u and point_u in list(map(lambda x: x.pos, self.snake.body))) or
            (dir_d and point_d in list(map(lambda x: x.pos, self.snake.body)))
        )

        danger_right = (
            (dir_u and point_r in list(map(lambda x: x.pos, self.snake.body))) or
            (dir_d and point_l in list(map(lambda x: x.pos, self.snake.body))) or
            (dir_l and point_u in list(map(lambda x: x.pos, self.snake.body))) or
            (dir_r and point_d in list(map(lambda x: x.pos, self.snake.body)))
        )

        danger_left = (
            (dir_d and point_r in list(map(lambda x: x.pos, self.snake.body))) or
            (dir_u and point_l in list(map(lambda x: x.pos, self.snake.body))) or
            (dir_r and point_u in list(map(lambda x: x.pos, self.snake.body))) or
            (dir_l and point_d in list(map(lambda x: x.pos, self.snake.body)))
        )

        fx, fy = self.snack.pos

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),

            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),

            int(fx < head[0]),  # food left
            int(fx > head[0]),  # food right
            int(fy < head[1]),  # food up
            int(fy > head[1])   # food down
        ]

        return np.array(state, dtype=np.float32)