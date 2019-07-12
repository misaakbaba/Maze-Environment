import environment as env
import numpy as np

# directions
left = 0
down = 1
right = 2
up = 3

# hyperparameters
lr = 0.01
y = 0.99
exploration_rate = 0.2

maze = [[0, 0, 0, 0],
        [0, -1, 0, -1],
        [0, 0, -1, 0],
        [-1, 0, 0, 1]]

np_maze = np.array(maze)
n_row, n_col = np_maze.shape
q_table = np.zeros((n_row * n_col, 4), dtype=float)

maze_env = env.Env(maze)
# state, reward, done = maze_env.step(right)
# state, reward, done = maze_env.step(down)
# env.show(maze_env)
# print(state)
# print(reward)
# print(done)
env.q_learn(maze_env, q_table, lr, y, exploration_rate, 1000)
