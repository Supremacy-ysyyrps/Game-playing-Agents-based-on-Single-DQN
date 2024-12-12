from agent import DQNAgent
from board import TicTacToeBoard

env = TicTacToeBoard()
agent = DQNAgent(env)
# agent.train()
# agent.show_trend()
agent.test_with_("ai")
agent.test_with_("human")
