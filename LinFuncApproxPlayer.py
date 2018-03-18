import numpy as np
import random
from BlackJack import BlackJack
from copy import deepcopy


class LinFuncApproxPlayer:
    def __init__(self, lmbda=0.0):
        self.N = np.zeros([10, 22, 2])
        self.W = np.zeros([36, ])
        self.epsilon = 0.05
        self.gamma = 0.7
        self.lmbda = lmbda
