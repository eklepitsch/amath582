import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from enum import Enum


JUMPING = 'jumping'
RUNNING = 'running'
WALKING = 'walking'
movements = [JUMPING, RUNNING, WALKING]


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
    for movement in movements:
        training_data[movement].append(
            np.load(os.path.join(train_data_dir, f'{movement}_{i}.npy')))
        if i == 1:
            testing_data[movement].append(
                np.load(os.path.join(test_data_dir, f'{movement}_{i}t.npy')))

n_movements = 3
n_joints = 38
n_axes = 3
n_timesteps = 100

# Construct the matrix X_train
# For a 1710 x 100 matrix:
# m = n_samples * n_movements * n_joints * n_axes
# n = n_timesteps
# X_train = np.empty((m, n))
# rows_per_sample = n_joints * n_axes
# rows_per_movement = n_samples * rows_per_sample

# def get_sample_row_range(movement, index):
#     i = movements.index(movement)
#     j = index
#     row_start = (i * rows_per_movement) + (j * rows_per_sample)
#     row_end = (i * rows_per_movement) + ((j + 1) * rows_per_sample)
#     return row_start, row_end
#
#
# for movement in movements:
#     for j in range(n_samples):
#         m_start, m_end = get_sample_row_range(movement, j)
#         X_train[m_start:m_end, :] = training_data[movement][j]
#
#
# def get_sample_data_xyz(movement, index):
#     m_start, m_end = get_sample_row_range(movement, index)
#     return X_train[m_start:m_end, :].reshape(n_joints, n_axes, -1)

# For a 114 x 1500 matrix:
m = n_joints * n_axes
n = n_movements * n_samples * n_timesteps
X_train = np.empty((m, n))

columns_per_sample = n_timesteps
columns_per_movement = n_samples * columns_per_sample


def get_sample_column_range(movement, index):
    i = movements.index(movement)
    j = index
    col_start = (i * columns_per_movement) + (j * columns_per_sample)
    col_end = (i * columns_per_movement) + ((j + 1) * columns_per_sample)
    return col_start, col_end


for movement in movements:
    for j in range(n_samples):
        n_start, n_end = get_sample_column_range(movement, j)
        X_train[:, n_start:n_end] = training_data[movement][j]


def get_sample_data_xyz(movement, index):
    n_start, n_end = get_sample_column_range(movement, index)
    return X_train[:, n_start:n_end].reshape(n_joints, n_axes, -1)


xyz_test = get_sample_data_xyz(RUNNING, 3)
xyz_real = training_data[RUNNING][3].reshape(n_joints, n_axes, -1)

pca = PCA(n_components=5)
pca.fit_transform(X_train.transpose())
mode_xyz = np.reshape(pca.components_, (5, n_joints, n_axes))
fig_pca_modes, ax_pca_modes = plt.subplots(subplot_kw=dict(projection='3d'))
for mode in range(5):
    ax_pca_modes.scatter3D(mode_xyz[mode, :, 0], mode_xyz[mode, :, 1],
                           mode_xyz[mode, :, 2], label=f'Mode {mode}')
ax_pca_modes.legend()
ax_pca_modes.set_xlabel('x')
ax_pca_modes.set_ylabel('y')
ax_pca_modes.set_zlabel('z')
fig_pca_modes.suptitle('First 5 PCA modes of $X_{train}$ in xyz-space')
fig_pca_modes.show()