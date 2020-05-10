import numpy as np
import os

from reversi_zero import board
from reversi_zero import nn_model
from reversi_zero import player
from reversi_zero import utils

EVALUATION_ROUNDS = 16

def evaluate(model_1, model_2):
    results = []
    for index in range(EVALUATION_ROUNDS):
        root_player = 1 if index % 2 == 0 else -1
        player_1 = player.ReversiZeroPlayer(root_player, model_1)
        player_2 = player.ReversiZeroPlayer(-root_player, model_2)
        winner, difference = player.play(player_1, player_2)
        results.append((1 if winner == root_player else -1 if winner == -root_player else 0, difference))
    return results

model_name_path_dict = {
    'raw': None,
    'serial_1': '/Volumes/ccschunk2/project_chunks/artificial_intelligence/reversi/res_8/training_4/best_model.hdf5',
    'serial_2': '/Volumes/ccschunk2/project_chunks/artificial_intelligence/reversi/res_8/training_5/best_model.hdf5',
    'full': '/Volumes/ccschunk2/project_chunks/artificial_intelligence/reversi/full_data_training/models/model_checkpoint_64.hdf5'
}

model_dict = {}
print('loading models.')
for name, path in model_name_path_dict.items():
    model_dict[name] = nn_model.NNModel()
    if path is not None:
        model_dict[name].load(path)
print('models loaded.')

for index, (name_1, model_1) in enumerate([*model_dict.items()]):
    for name_2, model_2 in [*model_dict.items()][index:]:
        evaluation_result = evaluate(model_1, model_2)
        print(name_1, 'vs', name_2)
        print(name_1, 'win:', [*map(lambda x: x[0], evaluation_result)].count(1))
        print('tie:', [*map(lambda x: x[0], evaluation_result)].count(0))
        print(name_2, 'win:', [*map(lambda x: x[0], evaluation_result)].count(-1))
