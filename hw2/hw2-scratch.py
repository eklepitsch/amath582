import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


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


def get_sample_data(movement, index):
    n_start, n_end = get_sample_column_range(movement, index)
    return X_train[:, n_start:n_end]


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
ds_2modes_diag = np.diag(ds_2modes)
ds_2modes_append = np.zeros((114, 1386))
ds_2modes = np.hstack((ds_2modes_diag, ds_2modes_append))
X_approx_2modes = np.mean(X_train, axis=1)[:, None] + \
                  np.dot(dU, np.dot(ds_2modes, dVt))

du_2 = dU[:, 0:2]
du_2_T = du_2.transpose()
pca_components_2 = np.dot(du_2_T, X_approx_2modes)

pca_2d_fig, pca_2d_ax = plt.subplots()
pca_2d_ax.plot(pca_components_2[0, 0:500], pca_components_2[1, 0:500],
               color='r', label=JUMPING)
pca_2d_ax.plot(pca_components_2[0, 500:1000], pca_components_2[1, 500:1000],
               color='g', label=RUNNING)
pca_2d_ax.plot(pca_components_2[0, 1000:1500], pca_components_2[1, 1000:1500],
               color='b', label=WALKING)
pca_2d_ax.legend()
pca_2d_ax.set_xlabel('PCA1')
pca_2d_ax.set_ylabel('PCA2')
pca_2d_fig.suptitle('PCA1, PCA2 space')
pca_2d_fig.show()

# Reconstruct X_train using 3 modes
ds_3modes = np.copy(ds)
ds_3modes[3:] = 0
ds_3modes_diag = np.diag(ds_3modes)
ds_3modes_append = np.zeros((114, 1386))
ds_3modes = np.hstack((ds_3modes_diag, ds_3modes_append))
X_approx_3modes = np.mean(X_train, axis=1)[:, None] + \
                  np.dot(dU, np.dot(ds_3modes, dVt))

du_3 = dU[:, 0:3]
du_3_T = du_3.transpose()
pca_components_3 = np.dot(du_3_T, X_approx_3modes)

pca_3d_fig, pca_3d_ax = plt.subplots(subplot_kw=dict(projection='3d'))
pca_3d_ax.plot(pca_components_3[0, 0:500], pca_components_3[1, 0:500],
               pca_components_3[2, 0:500],
               color='r', label=JUMPING)
pca_3d_ax.plot(pca_components_3[0, 500:1000], pca_components_3[1, 500:1000],
               pca_components_3[2, 500:1000],
               color='g', label=RUNNING)
pca_3d_ax.plot(pca_components_3[0, 1000:1500], pca_components_3[1, 1000:1500],
               pca_components_3[2, 1000:1500],
               color='b', label=WALKING)
pca_3d_ax.legend()
pca_3d_ax.set_xlabel('PCA1')
pca_3d_ax.set_ylabel('PCA2')
pca_3d_ax.set_zlabel('PCA3')
pca_3d_fig.suptitle('PCA1, PCA2, PCA3 space')
pca_3d_fig.show()

# Compute centroids for each movement in each space
pca_2d_centroids = {
    JUMPING: (np.mean(pca_components_2[0, 0:500]),
              np.mean(pca_components_2[1, 0:500])),
    RUNNING: (np.mean(pca_components_2[0, 500:1000]),
              np.mean(pca_components_2[1, 500:1000])),
    WALKING: (np.mean(pca_components_2[0, 1000:1500]),
              np.mean(pca_components_2[1, 1000:1500]))
}
pca_2d_ax.plot(pca_2d_centroids[JUMPING][0],
               pca_2d_centroids[JUMPING][1],
               label=JUMPING, color='r', marker='o', markersize=12)
pca_2d_ax.plot(pca_2d_centroids[RUNNING][0],
               pca_2d_centroids[RUNNING][1],
               label=RUNNING, color='g', marker='o', markersize=12)
pca_2d_ax.plot(pca_2d_centroids[WALKING][0],
               pca_2d_centroids[WALKING][1],
               label=WALKING, color='b', marker='o', markersize=12)
pca_2d_fig.show()

pca_3d_centroids = {
    JUMPING: (np.mean(pca_components_3[0, 0:500]),
              np.mean(pca_components_3[1, 0:500]),
              np.mean(pca_components_3[2, 0:500])),
    RUNNING: (np.mean(pca_components_3[0, 500:1000]),
              np.mean(pca_components_3[1, 500:1000]),
              np.mean(pca_components_3[2, 500:1000])),
    WALKING: (np.mean(pca_components_3[0, 1000:1500]),
              np.mean(pca_components_3[1, 1000:1500]),
              np.mean(pca_components_3[2, 1000:1500]))
}
pca_3d_ax.plot(pca_3d_centroids[JUMPING][0],
               pca_3d_centroids[JUMPING][1],
               pca_3d_centroids[JUMPING][2],
               label=JUMPING, color='r', marker='o', markersize=12)
pca_3d_ax.plot(pca_3d_centroids[RUNNING][0],
               pca_3d_centroids[RUNNING][1],
               pca_3d_centroids[RUNNING][2],
               label=RUNNING, color='g', marker='o', markersize=12)
pca_3d_ax.plot(pca_3d_centroids[WALKING][0],
               pca_3d_centroids[WALKING][1],
               pca_3d_centroids[WALKING][2],
               label=WALKING, color='b', marker='o', markersize=12)
pca_3d_fig.show()

ground_truth = [JUMPING, JUMPING, JUMPING, JUMPING, JUMPING,
                RUNNING, RUNNING, RUNNING, RUNNING, RUNNING,
                WALKING, WALKING, WALKING, WALKING, WALKING]
predicted_labels_2PCA = []
predicted_labels_3PCA = []

# Loop through each sample, project to PCA-2 space, and classify
for m in movements:
    for i in range(n_samples):
        sample = get_sample_data(m, i)
        pca_point = np.dot(du_2_T, sample)
        pca_point_centroid = (np.mean(pca_point[0, :]), np.mean(pca_point[1, :]))
        distances = {}
        for mvmt, centroid in pca_2d_centroids.items():
            distances[mvmt] = math.dist(pca_point_centroid, centroid)
        predicted_labels_2PCA.append(min(distances, key=distances.get))

accuracy_2pca = accuracy_score(ground_truth, predicted_labels_2PCA)
print(f'Accuracy of classifier in 2-PCA space: {accuracy_2pca}')

# Loop through each sample, project to PCA-3 space, and classify
for m in movements:
    for i in range(n_samples):
        sample = get_sample_data(m, i)
        pca_point = np.dot(du_3_T, sample)
        pca_point_centroid = (np.mean(pca_point[0, :]),
                              np.mean(pca_point[1, :]),
                              np.mean(pca_point[2, :]))
        distances = {}
        for mvmt, centroid in pca_3d_centroids.items():
            distances[mvmt] = math.dist(pca_point_centroid, centroid)
        predicted_labels_3PCA.append(min(distances, key=distances.get))

accuracy_3pca = accuracy_score(ground_truth, predicted_labels_3PCA)
print(f'Accuracy of classifier in 3-PCA space: {accuracy_3pca}')

ground_truth = []
for i in range(500):
    ground_truth.append(JUMPING)
for i in range(500):
    ground_truth.append(RUNNING)
for i in range(500):
    ground_truth.append(WALKING)

# Build a generic classifier for any k-PCA space
def classifier(k):
    if k > n_joints * n_axes:
        print(f'k cannot be larger than {n_joints * n_axes}')
        return

    # Reconstruct X_train using k modes
    ds_kmodes = np.copy(ds)
    ds_kmodes[k:] = 0
    ds_kmodes_diag = np.diag(ds_kmodes)
    ds_kmodes_append = np.zeros((114, 1386))
    ds_kmodes = np.hstack((ds_kmodes_diag, ds_kmodes_append))
    X_approx_kmodes = np.mean(X_train, axis=1)[:, None] + \
                      np.dot(dU, np.dot(ds_kmodes, dVt))

    # Project the reconstructed X_train into k-PCA space
    du_k = dU[:, 0:k]
    du_k_T = du_k.transpose()
    pca_components_k = np.dot(du_k_T, X_approx_kmodes)
    pass

    # Compute the k-centroid for each movement
    pca_kd_centroids = {}
    pca_kd_centroids[JUMPING] = \
        [np.mean(pca_components_k[i, 0:500]) for i in range(k)]
    pca_kd_centroids[RUNNING] = \
        [np.mean(pca_components_k[i, 500:1000]) for i in range(k)]
    pca_kd_centroids[WALKING] = \
        [np.mean(pca_components_k[i, 1000:1500]) for i in range(k)]

    # Loop through each sample, project to PCA-k space, and classify
    predicted_labels_kPCA = []
    for m in movements:
        for i in range(n_samples):
            sample = get_sample_data(m, i)
            for j in range(n_timesteps):
                pca_point = np.dot(du_k_T, sample[:, j])
                # pca_point_centroid = [np.mean(pca_point[i, :]) for i in range(k)]
                distances = {}
                for mvmt, centroid in pca_kd_centroids.items():
                    distances[mvmt] = math.dist(pca_point, centroid)
                predicted_labels_kPCA.append(min(distances, key=distances.get))

    accuracy_kpca = accuracy_score(ground_truth, predicted_labels_kPCA)

    return accuracy_kpca, predicted_labels_kPCA

accuracies = []
for k in range(1, 41):
    accuracy, _ = classifier(k)
    accuracies.append(accuracy)

print(accuracies)

fig_accuracy, ax_accuracy = plt.subplots()
k = np.arange(1, 41)
ax_accuracy.plot(k, accuracies, label='Training')
ax_accuracy.legend()
ax_accuracy.set_xlabel('k')
ax_accuracy.set_ylabel('accuracy')
fig_accuracy.suptitle('Classifier accuracy for various $k$')
fig_accuracy.show()
