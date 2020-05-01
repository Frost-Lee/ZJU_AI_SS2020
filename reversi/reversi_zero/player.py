import numpy as np
from tensorflow import keras

from . import mct
from . import nn_model
from .board import Board
from . import utils
from . import config


class ReversiPlayer(object):
    
    def __init__(self, color):
        self.color = color
        assert color == -1 or color == 1
    
    def play(self, board):
        return 'A1', None, None
    
    def notify(self, position):
        print(position)


class HumanPlayer(ReversiPlayer):

    def __init__(self, color):
        super().__init__(color)
    
    def play(self, board):
        position = input('Give a position to place your piece.')
        return utils.flat_coordinate(position), None, None
    
    def notify(self, position):
        super().notify(position)


class ReversiZeroPlayer(ReversiPlayer):

    def __init__(self, color, model):
        super().__init__(color)
        self.model = model
        self.root = mct.MCTNode(None, 1, -self.color)
    
    def play(self, board, strategy='determinestic'):
        policy, value = mct.MCTSearch.evaluate(self.root, board, self.model, with_noise=strategy=='probabilistic')
        if strategy == 'determinestic':
            move = np.argmax(policy)
        elif strategy == 'probabilistic':
            move = np.random.choice(policy.shape[0], p=policy)
        self.root = self.root.children[move]
        self.root.parent = None
        return move, policy, value
    
    def notify(self, position):
        if self.root.children is not None and position in self.root.children:
            self.root = self.root.children[position]
            self.root.parent = None
        else:
            self.root = mct.MCTNode(None, 1, -self.color)


def play(player_1, player_2, verbose=0):
    assert player_1.color != player_2.color
    board = Board()
    player_dict = {player_1.color: player_1, player_2.color: player_2}
    next_player = -1
    while next_player != 0:
        move, policy, value = player_dict[next_player].play(board)
        move_result = board._move(utils.grid_coordinate(move), utils.player_char_identifier(next_player))
        assert move_result != False
        player_dict[-next_player].notify(move)
        if verbose == 1:
            print(utils.player_char_identifier(next_player), ' moved at ', utils.grid_coordinate(move))
        elif verbose == 2:
            board.display()
        next_player, _ = utils.next_player(board, next_player)
    winner, difference = board.get_winner()
    winner = utils.winner_mapping(winner)
    return winner, difference


def self_play(model, verbose=0):
    states, policies, player_queue = [], [], []
    board = Board()
    player_1, player_2 = ReversiZeroPlayer(-1, model), ReversiZeroPlayer(1, model)
    player_dict = {-1: player_1, 1: player_2}
    next_player = -1
    while next_player != 0:
        move, policy, value = player_dict[next_player].play(board, 'probabilistic')
        states.append(utils.board_state(board, next_player))
        policies.append(policy)
        move_result = board._move(utils.grid_coordinate(move), utils.player_char_identifier(next_player))
        assert move_result != False
        player_dict[-next_player].notify(move)
        if verbose == 1:
            print(utils.player_char_identifier(next_player), ' moved at ', utils.grid_coordinate(move))
        elif verbose == 2:
            board.display()
        player_queue.append(next_player)
        next_player, _ = utils.next_player(board, next_player)
    winner = utils.winner_mapping(board.get_winner()[0])
    values = [1 if player == winner else -1 if player == -winner else 0 for player in player_queue]
    return states, policies, values
