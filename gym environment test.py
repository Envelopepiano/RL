import gym
from gym import spaces
import pygame
import numpy as np

class CustomEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.states = ['A', 'B', 'C', 'D', 'E', 'F']
        self.size = len(self.states)  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.observation_space = spaces.Discrete(self.size)
        self.action_space = spaces.Discrete(6)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([0, 0]),  # No movement for state F
            5: np.array([0, 0])   # No movement for state F
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"state": self.states[self._agent_location]}

    def _get_info(self):
        return {"distance": 0}  # You can modify this based on your requirements

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 设置初始状态和目标状态
        self._agent_location = self.states.index('C')  # 将代理位置设置为C
        self._target_location = self.states.index('F')  # 将目标位置设置为F

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        terminated = (self.states[self._agent_location] == 'F')
        reward = 50 if terminated else -1
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        for i, state in enumerate(self.states):
            pygame.draw.rect(
                canvas,
                (255, 0, 0) if state == 'F' else (255, 255, 255),
                pygame.Rect(
                    pix_square_size * i,
                    0,
                    pix_square_size,
                    pix_square_size
                ),
            )

        pygame.draw.circle(
        canvas,
        (0, 0, 255),
        (int((self._agent_location + 0.5) * pix_square_size),
        int((self._agent_location + 0.5) * pix_square_size)),
        int(pix_square_size / 3),
        )



        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

# Create an instance of the environment with human rendering
env = CustomEnv(render_mode='human')

# Reset the environment to get the initial observation and info
observation, info = env.reset()

# Optionally, you can run a loop to see the agent taking random steps
for _ in range(30):
    action = env.action_space.sample()  # Select a random action
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated:
        break

# Close the environment to clean up any resources
env.close()

from gymnasium.envs.registration import register

register(
    id='CustomEnv-v0',
    entry_point="C:/Users/user/Desktop/碩一上/RL/gym environment test.py",  
)

