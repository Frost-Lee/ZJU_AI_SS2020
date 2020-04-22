import numpy as np
import os

from reversi_zero import board
from reversi_zero import nn_model
from reversi_zero import player

model_path = '/Volumes/ccschunk2/reversi_zero_models'
model_1 = nn_model.NNModel()
model_2 = nn_model.NNModel()
model_2.load('/Users/Frost/Desktop/170.hdf5')
# player.self_play(model, verbose=2)
player_1 = player.ReversiZeroPlayer(-1, model_2)
player_2 = player.ReversiZeroPlayer(1, model_1)

player.play(player_1, player_2, verbose=2)

