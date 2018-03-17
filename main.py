from BlackJack import BlackJack
from HumanPlayer import HumanPlayer
from RandomPlayer import RandomPlayer

me = HumanPlayer()
stupid_ai1 = RandomPlayer(28)
stupid_ai2 = RandomPlayer(33)
stupid_ai3 = RandomPlayer(98)
new_game = BlackJack([me, stupid_ai1, stupid_ai2, stupid_ai3], 332)

while not new_game.game_over:
    new_game.step()