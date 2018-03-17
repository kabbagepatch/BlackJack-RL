import random
from Player import Player, HIT, STICK


class RandomPlayer(Player):
    def __init__(self, seed, description = "Randomly sticks or hits."):
        Player.__init__(self)
        random.seed(seed)
        self.description = description

    def choose_action(self, state):
        if state is None:
            raise StandardError("No game associated to player")

        return random.randint(HIT, STICK)

    def receive_reward(self, reward):
        if reward == -1:
            print "Randomly lost"
        if reward == 1:
            print "Randomly won"
