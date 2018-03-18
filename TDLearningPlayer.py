import numpy as np
import random
from BlackJack import BlackJack
from Player import Player, HIT, STICK
from copy import deepcopy


class TDLearningPlayer(Player):
    def __init__(self, lmbda=0.0):
        Player.__init__(self)
        self.Q = np.random.randn(11, 21, 5, 2)
        self.N = np.zeros([11, 21, 5, 2])
        self.epsilon = 1.
        self.lmbda = lmbda
        self.gamma = 0.7
        self.last_reward = 0

    def choose_action(self, state):
        if state is None:
            raise StandardError("No game associated to player")

        dealers_first_card = state[0]
        random.seed(dealers_first_card)
        random_action = random.randint(0, 1)

        action_choice = np.random.choice(['RAND', 'GREEDY'], p=[self.epsilon, 1 - self.epsilon])

        if action_choice == 'RAND':
            return random_action

        if self.Q[dealers_first_card - 1][self.current_total - 1][self.number_of_aces_used][HIT] == \
                self.Q[dealers_first_card - 1][self.current_total - 1][self.number_of_aces_used][STICK]:
            return random_action

        if self.Q[dealers_first_card - 1][self.current_total - 1][self.number_of_aces_used][HIT] > \
                self.Q[dealers_first_card - 1][self.current_total - 1][self.number_of_aces_used][STICK]:
            return HIT
        else:
            return STICK

    def receive_reward(self, reward):
        self.last_reward = reward

    def run_episode(self, index):
        game = BlackJack([self], index)
        self.current_total = 0
        self.number_of_aces_used = 0
        action = self.choose_action(game.get_current_state())
        old_state = deepcopy(game.get_current_state())
        E = np.zeros([11, 21, 5, 2])
        reward = 0
        new_q = deepcopy(self.Q)

        while not game.game_over:
            old_total = self.current_total - 1
            old_naces = self.number_of_aces_used

            game.step([action])
            if game.game_over:
                break

            new_action = self.choose_action(old_state)
            reward = self.last_reward

            new_total = self.current_total - 1
            new_naces = self.number_of_aces_used

            dealers_first_card = game.get_current_state()[0]

            delta = reward + \
                self.gamma * self.Q[dealers_first_card - 1][new_total][new_naces][new_action] - \
                self.Q[dealers_first_card - 1][old_total][old_naces][action]

            E[dealers_first_card - 1][old_total][old_naces][action] += 1

            self.N[dealers_first_card - 1][old_total][old_naces][action] += 1
            alpha = 1. / self.N[dealers_first_card - 1][old_total][old_naces][action]

            self.Q = self.Q + alpha * delta * E
            E = self.gamma * self.lmbda * E

            self.epsilon = 100. / (100 + self.N[dealers_first_card - 1][self.current_total - 1][self.number_of_aces_used][action])
            action = new_action
            old_state = deepcopy(game.get_current_state())

        self.Q = deepcopy(new_q)
        return reward

    def run_episodes(self, n):
        for k in range(1, n+1):
            self.run_episode(k)

        return self.Q, self.N
