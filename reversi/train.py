import numpy as np
import os
import h5py

from reversi_zero import board
from reversi_zero import nn_model
from reversi_zero import player

TRAINING_DATA_ARCHIVE_PATH = '/Users/Frost/Desktop/training.hdf5'
MODEL_ARCHIVE_PATH = '/Users/Frost/Desktop/dump_model.hdf5'
BEST_MODEL_ARCHIVE_PATH = '/Users/Frost/Desktop/best_model.hdf5'
TRAINING_QUEUE_LENGTH = 4
TRAINING_BATCHES = 128
PLAYS_PER_BATCH = 32
EPOCHS_PER_BATCH = 128
EVALUATE_COUNT = 5
EVALUATE_SUCCESS_COUNT = 3

# model_path = '/Volumes/ccschunk2/reversi_zero_models'
# model_1 = nn_model.NNModel()
# model_2 = nn_model.NNModel()
# model_2.load('/Users/Frost/Desktop/170.hdf5')
# player.self_play(model, verbose=2)
# player_1 = player.ReversiZeroPlayer(-1, model_2)
# player_2 = player.ReversiZeroPlayer(1, model_1)

# player.play(player_1, player_2, verbose=2)

def evaluate(model_1, model_2):
    player_1 = player.ReversiZeroPlayer(-1, model_1)
    player_2 = player.ReversiZeroPlayer(1, model_2)
    results = []
    for _ in range(EVALUATE_COUNT):
        results.append(player.play(player_1, player_2))
    return results

best_model = nn_model.NNModel()
states_queue, policies_queue, values_queue = [], [], []
for batch_index in range(TRAINING_BATCHES):
    state_batches, policy_batches, value_batches = [], [], []
    for play_index in range(PLAYS_PER_BATCH):
        states, policies, values = player.self_play(best_model, verbose=1)
        state_batches += states
        policy_batches += policies
        value_batches += values
        print(play_index, ' play finished.')
    states_queue += state_batches
    policies_queue += policy_batches
    values_queue += value_batches
    if len(states_queue) > PLAYS_PER_BATCH * TRAINING_QUEUE_LENGTH:
        states_queue = states_queue[PLAYS_PER_BATCH:]
    if len(policies_queue) > PLAYS_PER_BATCH * TRAINING_QUEUE_LENGTH:
        policies_queue = policies_queue[PLAYS_PER_BATCH:]
    if len(values_queue) > PLAYS_PER_BATCH * TRAINING_QUEUE_LENGTH:
        values_queue = values_queue[PLAYS_PER_BATCH:]
    state_batches, policy_batches, value_batches = np.array(state_batches), np.array(policy_batches), np.array(value_batches)
    with h5py.File(TRAINING_DATA_ARCHIVE_PATH, 'w') as out_file:
        out_file['batch_{}/states'.format(batch_index)] = state_batches
        out_file['batch_{}/policies'.format(batch_index)] = policy_batches
        out_file['batch_{}/values'.format(batch_index)] = value_batches
    new_model = best_model.clone()
    # Augment the data & Train the model
    new_model.fit(np.array(states_queue), np.array(policies_queue), np.array(values_queue))
    evaluate_result = evaluate(best_model, new_model)
    if [*map(lambda x: x[0], evaluate_result)].count(1) >= EVALUATE_SUCCESS_COUNT:
        best_model = new_model
        print('new model won, use new model for generation.')
    else:
        print('new model lose, still use previous model for generation.')
    if batch_index % 8 == 0 or batch_index == TRAINING_BATCHES - 1:
        new_model.save(MODEL_ARCHIVE_PATH)



with h5py.File(TRAINING_DATA_ARCHIVE_PATH, 'w') as out_file:
    states, policies, values = player.self_play(best_model)
