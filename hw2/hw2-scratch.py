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

pca = PCA(n_components=5, svd_solver='full')
pca.fit(X_train.transpose())
mode_xyz = np.reshape(pca.components_, (5, n_joints, n_axes))
fig_pca_modes, ax_pca_modes = plt.subplots(1, 5, subplot_kw=dict(projection='3d'))
for mode in range(5):
    ax_pca_modes[mode].scatter3D(mode_xyz[mode, :, 0], mode_xyz[mode, :, 1],
                                 mode_xyz[mode, :, 2], label=f'Mode {mode + 1}')
    ax_pca_modes[mode].legend()
    ax_pca_modes[mode].set_xlabel('x')
    ax_pca_modes[mode].set_ylabel('y')
    ax_pca_modes[mode].set_zlabel('z')

fig_pca_modes.suptitle('First 5 PCA modes of $X_{train}$ in xyz-space - using sklearn.PCA')
fig_pca_modes.set_figwidth(15)
fig_pca_modes.tight_layout(pad=3)
fig_pca_modes.show()


# SVD approach
X_train_centered = X_train - np.mean(X_train, axis=1)[:, None]
dU, ds, dVt = np.linalg.svd(X_train_centered)
print(dU.shape, ds.shape, dVt.shape)

fig_pca_modes_svd, ax_pca_modes_svd = plt.subplots(1, 5, subplot_kw=dict(projection='3d'))
for mode in range(5):
    mode_xyz = dU[:, mode].reshape(n_joints, n_axes)
    ax_pca_modes_svd[mode].scatter3D(mode_xyz[:, 0], mode_xyz[:, 1],
                                     mode_xyz[:, 2], label=f'Mode {mode + 1}')
    ax_pca_modes_svd[mode].legend()
    ax_pca_modes_svd[mode].set_xlabel('x')
    ax_pca_modes_svd[mode].set_ylabel('y')
    ax_pca_modes_svd[mode].set_zlabel('z')

fig_pca_modes_svd.suptitle('First 5 PCA modes of $X_{train}$ in xyz-space - using SVD')
fig_pca_modes_svd.set_figwidth(15)
fig_pca_modes_svd.tight_layout(pad=3)
fig_pca_modes_svd.show()

# First 5 PCA modes in 1-D space
fig_pca_modes_svd_1d, ax_pca_modes_svd_1d = plt.subplots()
for mode in range(5):
    ax_pca_modes_svd_1d.plot(dU[:, mode], label=f'Mode {mode + 1}')
ax_pca_modes_svd_1d.legend()
fig_pca_modes_svd_1d.suptitle('First 5 PCA modes using SVD, in 1D')
fig_pca_modes_svd_1d.show()

# Plot singular values
fig_singular_values, ax_singular_values = plt.subplots()
E = np.power(ds, 2)/np.sum(np.power(ds, 2))
E_cumsum = np.cumsum(E)
ax_singular_values.plot(E_cumsum[:15], label='Cumulative energy')
ax_singular_values.hlines([0.7, 0.8, 0.9, 0.95], 0, 15, linestyles='dashed', colors='r',
                          label='Energy level threshold')

ax_singular_values.set_xlabel('index $j$')
ax_singular_values.set_ylabel('$\Sigma E_j$')
ax_singular_values.legend()
fig_singular_values.suptitle('Cumulative energy - Using SVD')
fig_singular_values.show()

# print(pca.singular_values_)
# print(ds)

# Reconstruct X_train using 2 modes
ds_2modes = np.copy(ds)
ds_2modes[2:] = 0
X_approx_2modes = np.mean(X_train, axis=1)[:, None] + np.dot(dU, np.dot(np.diag(ds_2modes), dVt))