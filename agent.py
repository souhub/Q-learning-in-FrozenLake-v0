import numpy as np


class QLearningAgent:
    def __init__(self, n_states, n_actions, gamma) -> None:
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions))
        self.gamma = gamma

    # epsilon-greedy
    def take_action(self, state, epsilon):
        if np.random.uniform() < epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        return action

    # update Q-table
    def learn(self, state, action, reward, next_state, alpha):
        gain = reward+self.gamma*max(self.Q[next_state])
        estimated = self.Q[state][action]
        self.Q[state][action] += alpha*(gain-estimated)
