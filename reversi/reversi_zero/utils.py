import numpy as np

def flat_coordinate(coordinate):
    row, col = None, None
    if isinstance(coordinate, str):
        row, col = '12345678'.index(coordinate[1].upper()), 'ABCDEFGH'.index(coordinate[0].upper())
    elif isinstance(coordinate, tuple):
        row, col = coordinate
    else:
        return None
    return row * 8 + col

def grid_coordinate(coordinate, format='board'):
    row, col = coordinate // 8, coordinate % 8
    if format == 'board':
        return chr(ord('A') + col) + str(row + 1)
    elif format == 'number':
        return (row, col)
    else:
        return None

def player_char_identifier(player):
    if player == -1:
        return 'X'
    elif player == 1:
        return 'O'
    else:
        return '.'

def player_num_identifier(player):
    if player == 'X':
        return -1
    elif player == 'O':
        return 1
    else:
        return 0

def board_state(board, player):
    state = np.zeros((8, 8, 2))
    board_grid = np.array([[*map(player_num_identifier, row)] for row in board._board])
    state[:, :, 0] = -board_grid * (board_grid < 0)
    state[:, :, 1] = board_grid * (board_grid > 0)
    if player == 1:
        state = np.flip(state)
    return state

def next_player(board, previous_player):
    player = -previous_player
    legal_moves = [*board.get_legal_actions(player_char_identifier(player))]
    if len(legal_moves) > 0:
        return player, legal_moves
    player = -player
    legal_moves = [*board.get_legal_actions(player_char_identifier(player))]
    if len(legal_moves) > 0:
        return player, legal_moves
    return 0, None

def winner_mapping(winner):
    if winner == 0:
        return -1
    elif winner == 1:
        return 1
    else:
        return 0

def choose_move(policy, temperature):
    policy = np.log(policy + 1e-10) / temperature
    policy = np.exp(policy - np.max(policy))
    policy = policy / np.sum(policy)
    return np.random.choice(policy.size, p=policy)
