from BlackJack import BlackJack
from HumanPlayer import HumanPlayer
from RandomPlayer import RandomPlayer
from MonteCarloPlayer import MonteCarloPlayer
from TDLearningPlayer import TDLearningPlayer
from LinFuncApproxPlayer import LinFuncApproxPlayer
from plotQValues import plotQValues

'''Human vs three Random AIs'''
# me = HumanPlayer()
# stupid_ai1 = RandomPlayer(28)
# stupid_ai2 = RandomPlayer(33)
# stupid_ai3 = RandomPlayer(98)
# new_game = BlackJack([me, stupid_ai1, stupid_ai2, stupid_ai3], 332)
#
# while not new_game.game_over:
#     actions = [me.choose_action(new_game.get_current_state()),
#                stupid_ai1.choose_action(new_game.get_current_state()),
#                stupid_ai1.choose_action(new_game.get_current_state()),
#                stupid_ai1.choose_action(new_game.get_current_state())]
#     new_game.step(actions)

# '''Human vs Monte Carlo Player'''
# me = HumanPlayer()
# mc = MonteCarloPlayer()
# q_mc, n_mc = mc.run_episodes(10000)
#
# new_game = BlackJack([me, mc], 332)
# while not new_game.game_over:
#     if new_game.player_done[0] == 0:
#         player1_action = 1
#     else:
#         player1_action = me.choose_action(new_game.get_current_state())
#
#     if new_game.player_done[1] == 0:
#         player2_action = 1
#     else:
#         player2_action = mc.choose_action(new_game.get_current_state())
#
#     actions = [player1_action, player2_action]
#     new_game.step(actions, True)

# '''Random AI vs Monte Carlo Player'''
# mc = MonteCarloPlayer()
# q_mc, n_mc = mc.run_episodes(10000)
# plotQValues(q_mc)
#
# mc.epsilon = 0
# rand_rewards = []
# mc_rewards = []
# for i in range(0, 100):
#     stupid_ai = RandomPlayer(i * 10)
#     mc.current_total = 0
#     mc.number_of_aces_used = 0
#     new_game = BlackJack([stupid_ai, mc], i * 15)
#
#     while not new_game.game_over:
#         if new_game.player_done[0] != 0:
#             player1_action = 1
#         else:
#             player1_action = stupid_ai.choose_action(new_game.get_current_state())
#
#         if new_game.player_done[1] != 0:
#             player2_action = 1
#         else:
#             player2_action = mc.choose_action(new_game.get_current_state())
#
#         actions = [player1_action, player2_action]
#         new_game.step(actions)
#
#     rand_rewards.append(stupid_ai.last_reward > 0)
#     mc_rewards.append(mc.last_reward > 0)
#
# print "Randomly won", sum(rand_rewards)
# print "Monte Carlo won", sum(mc_rewards)

'''Monte Carlo vs TD Learning Player vs Linear Func Approximator vs Random AI'''
mc = MonteCarloPlayer()
q_mc, n_mc = mc.run_episodes(10000)
td = TDLearningPlayer()
q_td, n_td = td.run_episodes(10000)
lin = LinFuncApproxPlayer()
lin.run_episodes(10000)

mc.epsilon = 0
td.epsilon = 0
lin.epsilon = 0
mc_rewards = []
td_rewards = []
lin_rewards = []
rand_rewards = []
for i in range(0, 100):
    rand = RandomPlayer(i * 10)
    mc.current_total = 0
    mc.number_of_aces_used = 0
    td.current_total = 0
    td.number_of_aces_used = 0
    lin.current_total = 0
    lin.number_of_aces_used = 0
    new_game = BlackJack([mc, td, lin, rand], i * 15, 3)

    while not new_game.game_over:
        if new_game.player_done[0] != 0:
            player1_action = 1
        else:
            player1_action = mc.choose_action(new_game.get_current_state())

        if new_game.player_done[1] != 0:
            player2_action = 1
        else:
            player2_action = td.choose_action(new_game.get_current_state())

        if new_game.player_done[2] != 0:
            player3_action = 1
        else:
            player3_action = lin.choose_action(new_game.get_current_state())

        if new_game.player_done[2] != 0:
            player4_action = 1
        else:
            player4_action = rand.choose_action(new_game.get_current_state())

        actions = [player1_action, player2_action, player3_action, player4_action]
        new_game.step(actions)

    rand_rewards.append(rand.last_reward > 0)
    lin_rewards.append(lin.last_reward > 0)
    td_rewards.append(td.last_reward > 0)
    mc_rewards.append(mc.last_reward > 0)

print "Lin won", sum(lin_rewards)
print "TD won", sum(td_rewards)
print "MC won", sum(mc_rewards)
print "Random won", sum(rand_rewards)
