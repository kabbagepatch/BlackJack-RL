import numpy as np
import random
from BlackJack import BlackJack
from Player import Player, HIT, STICK
from copy import deepcopy


class LinFuncApproxPlayer(Player):
    def __init__(self, lmbda=0.0):
        Player.__init__(self)
        self.N = np.zeros([11, 21, 5, 2])
        self.W = np.zeros([96, ])
        self.epsilon = 0.05
        self.lmbda = lmbda
        self.gamma = 0.7
        self.last_reward = 0

    @staticmethod
    def generate_features(game_state, current_total, number_of_aces_used, action):
        features = np.zeros([4, 6, 2, 2])

        dealers_first_card = game_state[0]
        number_of_aces_used = min(number_of_aces_used, 1)

        if dealers_first_card <= 10 and current_total <= 18:
            features[(dealers_first_card - 1) / 3][(current_total - 1) / 3][number_of_aces_used][action] = 1

        if dealers_first_card <= 11 and current_total <= 18:
            features[(dealers_first_card - 2) / 3][(current_total - 1) / 3][number_of_aces_used][action] = 1

        if dealers_first_card <= 10 and current_total <= 21:
            features[(dealers_first_card - 1) / 3][(current_total - 4) / 3][number_of_aces_used][action] = 1

        if dealers_first_card <= 11 and current_total <= 21:
            features[(dealers_first_card - 2) / 3][(current_total - 4) / 3][number_of_aces_used][action] = 1

        return features.reshape((96, ))

    def choose_action(self, state):
        if state is None:
            raise StandardError("No game associated to player")

        dealers_first_card = state[0]
        random.seed(dealers_first_card)
        random_action = random.randint(0, 1)

        action_choice = np.random.choice(['RAND', 'GREEDY'], p=[self.epsilon, 1 - self.epsilon])

        if action_choice == 'RAND':
            return random_action

        if np.dot(self.generate_features(state, self.current_total, self.number_of_aces_used, HIT), self.W) == \
                np.dot(self.generate_features(state, self.current_total, self.number_of_aces_used, STICK), self.W):
            return random_action

        if np.dot(self.generate_features(state, self.current_total, self.number_of_aces_used, HIT), self.W) > \
                np.dot(self.generate_features(state, self.current_total, self.number_of_aces_used, STICK), self.W):
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
        eligibility_trace = np.zeros([96, ])
        reward = 0
        alpha = 0.01

        while not game.game_over:
            old_total = self.current_total - 1
            old_n_aces = self.number_of_aces_used

            game.step([action])
            reward = self.last_reward
            new_action = self.choose_action(old_state)
            new_total = self.current_total - 1
            new_n_aces = self.number_of_aces_used

            old_q = np.dot(self.generate_features(old_state, old_total, old_n_aces, action), self.W)
            new_q = np.dot(self.generate_features(game.get_current_state(), new_total, new_n_aces, new_action), self.W) if not game.game_over else 0

            delta = reward + self.gamma * new_q - old_q
            eligibility_trace = self.gamma * self.lmbda * eligibility_trace + self.generate_features(old_state, old_total, old_n_aces, action)
            self.W = self.W + alpha * delta * eligibility_trace

            old_state = deepcopy(game.get_current_state())
            action = new_action
        return reward

    def run_episodes(self, n):
        for k in range(1, n+1):
            self.run_episode(k)
