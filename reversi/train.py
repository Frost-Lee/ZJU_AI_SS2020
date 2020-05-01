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
BATCH_SIZE = 64
EPOCHS = 32
EVALUATE_COUNT = 5
EVALUATE_SUCCESS_COUNT = 3


class TrainingDataFeed(object):

    def __init__(self):
        self.states = []
        self.policies = []
        self.values = []
        self.batch_play_count = 0
        self.batch_count = 0

    def collect(self, states, policies, values):
        assert len(self.states) == len(self.policies) == len(self.values)
        self.batch_play_count += 1
        self.states += states
        self.policies += policies
        self.values += values
        max_sample_len = TRAINING_QUEUE_LENGTH * PLAYS_PER_BATCH * 8 * 8 - 4
        if len(self.states) > max_sample_len:
            self.states = self.states[-max_sample_len:]
            self.policies = self.policies[-max_sample_len:]
            self.values = self.values[-max_sample_len:]
        if self.batch_play_count == PLAYS_PER_BATCH:
            self.dump()
            self.batch_play_count = 0
    
    def fetch(self):
        states, policies, values = np.array(self.states), np.array(self.policies), np.array(self.values)
        states = np.concatenate([
            states, 
            states[:, ::-1, ::-1, :], 
            states[:, ::-1, :, :], 
            states[:, :, ::-1, :],
            np.rot90(states, k=1, axes=(1, 2)),
            np.rot90(states, k=2, axes=(1, 2)), 
            np.rot90(states, k=3, axes=(1, 2))
        ])
        policies = np.reshape(policies, (policies.shape[0], 8, 8))
        policies = np.concatenate([
            policies,
            policies[:, ::-1, ::-1],
            policies[:, ::-1, :],
            policies[:, :, ::-1],
            np.rot90(policies, k=1, axes=(1, 2)),
            np.rot90(policies, k=2, axes=(1, 2)), 
            np.rot90(policies, k=3, axes=(1, 2))
        ])
        policies = np.reshape(policies, (policies.shape[0], 8 * 8))
        values = np.concatenate([values] * 7)
        return states, policies, values

    def dump(self):
        batch_sample_len = PLAYS_PER_BATCH * 8 * 8 - 4
        with h5py.File(TRAINING_DATA_ARCHIVE_PATH, 'a') as out_file:
            out_file['batch_{}/states'.format(self.batch_count)] = np.array(self.states[-batch_sample_len:])
            out_file['batch_{}/policies'.format(self.batch_count)] = np.array(self.policies[-batch_sample_len:])
            out_file['batch_{}/values'.format(self.batch_count)] = np.array(self.values[-batch_sample_len:])
            self.batch_count += 1


def evaluate(model_1, model_2):
    results = []
    for index in range(EVALUATE_COUNT):
        root_player = 1 if index % 2 == 0 else -1
        player_1 = player.ReversiZeroPlayer(root_player, model_1)
        player_2 = player.ReversiZeroPlayer(-root_player, model_2)
        results.append(player.play(player_1, player_2))
    return results


best_model = nn_model.NNModel()
data_feed = TrainingDataFeed()
new_model_count = 0
for batch_index in range(TRAINING_BATCHES):
    print('begin training batch', batch_index, '.')
    for play_index in range(PLAYS_PER_BATCH):
        data_feed.collect(*player.self_play(best_model, verbose=0))
        print('\r', play_index, ' play finished.', end='')
    new_model = best_model.clone()
    new_model.fit(*data_feed.fetch(), batch_size=BATCH_SIZE, epochs=EPOCHS)
    new_model.save(MODEL_ARCHIVE_PATH)
    print('evaluating models.')
    evaluate_result = evaluate(best_model, new_model)
    print('new model win:', evaluate_result.count(1))
    if [*map(lambda x: x[0], evaluate_result)].count(1) >= EVALUATE_SUCCESS_COUNT:
        best_model = new_model
        new_model_count += 1
        best_model.save(BEST_MODEL_ARCHIVE_PATH)
        print('new model adopted.')
    else:
        print('new model not adopted.')
    print('model updated for', new_model_count, 'times.')
