import numpy as np
import copy

from . import utils
from . import config

class MCTNode(object):
    def __init__(self, parent, prior_probability, player):
        self.parent = parent
        self.children = None
        self.prior_probability = prior_probability
        self.player = player
        self.visit_count = 0
        self.action_value_sum = 0
    
    @property
    def action_value(self):
        if self.visit_count == 0:
            return 0
        return self.action_value_sum / self.visit_count

    def upper_confidence_bound(self, cpuct):
        sum_visit_count = self.visit_count if self.parent is None else self.parent.visit_count
        sum_visit_count = sum_visit_count if sum_visit_count != 0 else 1
        return self.action_value + cpuct * self.prior_probability * np.sqrt(sum_visit_count) / (1 + self.visit_count)

    def expand(self, player, actions, policy):
        self.children = dict(zip(
            actions,
            [MCTNode(self, prior_probability, player) for prior_probability in policy]
        ))
    
    def record_simulation(self, value):
        self.visit_count += 1
        self.action_value_sum += value


class MCTSearch(object):
    def __init__(self):
        pass
    
    @staticmethod
    def _expand_node(root, board, model):
        player, legal_moves = utils.next_player(board, root.player)
        if player == 0:
            winner = utils.winner_mapping(board.get_winner()[0])
            if winner == player:
                return 1
            elif winner == -player:
                return -1
            else:
                return 0
        legal_moves = [*map(utils.flat_coordinate, legal_moves)]
        policy, value = model.predict(utils.board_state(board, player))
        policy = [policy[move] for move in legal_moves]
        policy /= np.sum(policy)
        root.expand(player, legal_moves, policy)
        return value

    @staticmethod
    def evaluate(root, board, model):
        cpuct = np.sqrt(config.MCT_SIMULATION_COUNT) / 8
        for _ in range(config.MCT_SIMULATION_COUNT):
            sub_board = copy.deepcopy(board)
            sub_root = root
            while sub_root.children is not None and len(sub_root.children) > 0:
                move, child = max(sub_root.children.items(), key=lambda x: x[1].upper_confidence_bound(cpuct))
                move_result = sub_board._move(utils.grid_coordinate(move), utils.player_char_identifier(child.player))
                assert move_result != False
                sub_root = child
            value = MCTSearch._expand_node(sub_root, sub_board, model)
            end_player = sub_root.player
            while sub_root is not None:
                sub_root.record_simulation(value if sub_root.player == end_player else -value)
                sub_root = sub_root.parent
        policy = np.zeros(8 * 8, dtype=np.float32)
        for key, value in root.children.items():
            policy[key] = value.visit_count
        policy, value = policy / np.sum(policy), root.action_value
        return policy, value
