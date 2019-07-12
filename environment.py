import random
import numpy as np
import matplotlib.pyplot as plt


class Env():
    def __init__(self, maze, player=(0, 0)):
        self.maze2 = np.array(maze)
        self.state = self.reset(player)
        nrows, ncol = self.maze2.shape
        self.target = (nrows - 1, ncol - 1)
        self.done = False
        self.visited = []

    def reset(self, player):
        self.state = (0, 0)
        self.done = False
        return self.state

    def update_state(self, action):
        # current_row, current_col = self.state
        state_row, state_col = self.state
        maze = self.maze2
        nrow, ncol = self.maze2.shape
        if action == 0 and state_col != 0:  # left
            state_col -= 1
        elif action == 1 and state_row != nrow - 1:  # down
            state_row += 1
        elif action == 2 and state_col != ncol - 1:  # right
            state_col += 1
        elif action == 3 and state_row != 0:  # up
            state_row -= 1

        if maze[state_row][state_col] == -1:
            return self.state
        self.visited.append([state_row, state_col])
        new_state = (state_row, state_col)
        return new_state

    def get_reward(self):
        # state_row, state_col = self.state
        if self.state == self.target:
            self.done = True
            return 1
        else:
            return 0

    def step(self, action):
        self.state = self.update_state(action)
        reward = self.get_reward()
        done = self.done
        return self.state, reward, done


def show(qmaze):
    plt.grid('on')
    nrows, ncols = qmaze.maze2.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze2)
    for row, col in qmaze.visited:
        canvas[row, col] = 0.6
    rat_row, rat_col = qmaze.state
    canvas[rat_row, rat_col] = 0.3  # rat cell
    canvas[nrows - 1, ncols - 1] = 0.9  # cheese cell
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    plt.show()
    return img


def q_learn(env, q_table, lr, y, exploration_rate, max_move):
    for i in range(max_move):
        state = env.state
        maze_row, maze_col = env.maze2.shape
        while not env.done:
            cur_row, cur_col = state
            index = cur_row * maze_col + cur_col
            if random.uniform(0, 1) > exploration_rate:
                action = np.argmax(q_table[index, :])
            else:
                action = random.randint(0, 3)

            new_state, reward, done = env.step(action)
            cur_row, cur_col = new_state
            new_index = cur_row * maze_col + cur_col
            q_table[index, action] = q_table[index, action] * (1 - lr) + \
                                     lr * (reward + y * np.max(q_table[new_index, :]))
            state = new_state
    print(q_table)

# def play_game(env, q_table):
#     while not env.done:
#         action = np.argmax(q_table[index, :])
#
