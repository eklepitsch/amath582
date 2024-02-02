import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

JUMPING = 'jumping'
RUNNING = 'running'
WALKING = 'walking'

# Specify data directories
data_dir = os.path.join(os.curdir, 'hw2data')
test_data_dir = os.path.join(data_dir, 'test')
train_data_dir = os.path.join(data_dir, 'train')

# Define containers for the data
training_data = {JUMPING: [], RUNNING: [], WALKING: []}
testing_data = {JUMPING: [], RUNNING: [], WALKING: []}

# Load the data
n_samples = 5
for i in range(1, n_samples + 1):
    for action in [JUMPING, RUNNING, WALKING]:
        training_data[action].append(
            np.load(os.path.join(train_data_dir, f'{action}_{i}.npy')))
        if i == 1:
            testing_data[action].append(
                np.load(os.path.join(test_data_dir, f'{action}_{i}t.npy')))

n_movements = 3
n_joints = 38
n_axes = 3
n_timesteps = 100

# Construct the matrix X_train
m = n_samples * n_movements * n_joints * n_axes
n = n_timesteps
X_train = np.empty((m, n))
rows_per_sample = n_joints * n_axes
rows_per_movement = n_samples * rows_per_sample
for i, action in enumerate([JUMPING, RUNNING, WALKING]):
    for j in range(n_samples):
        m_start = (i * rows_per_movement) + (j * rows_per_sample)
        m_end = (i * rows_per_movement) + ((j + 1) * rows_per_sample)
        X_train[m_start:m_end, :] = training_data[action][i]


pass

