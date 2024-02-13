# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: .venv
# ---

# %%
import numpy as np
import struct
import matplotlib.pyplot as plt

with open('data/train-images.idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    Xtraindata = np.transpose(data.reshape((size, nrows*ncols)))

with open('data/train-labels.idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    ytrainlabels = data.reshape((size,)) # (Optional)

with open('data/t10k-images.idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    Xtestdata = np.transpose(data.reshape((size, nrows*ncols)))

with open('data/t10k-labels.idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    ytestlabels = data.reshape((size,)) # (Optional)
        

    
traindata_imgs =  np.transpose(Xtraindata).reshape((60000,28,28))    
print(Xtraindata.shape)
print(ytrainlabels.shape)
print(Xtestdata.shape)
print(ytestlabels.shape)


# %%
def plot_digits(XX, N, title):
    fig, ax = plt.subplots(N, N, figsize=(N, N))
    
    for i in range(N):
      for j in range(N):
        ax[i,j].imshow(XX[:,(N)*i+j].reshape((28, 28)), cmap="Greys")
        ax[i,j].axis("off")
    fig.suptitle(title, fontsize=24)

plot_digits(Xtraindata, 8, "First 64 Training Images" )

# %% [markdown]
# Find the first 16 PC modes and plot them

# %%
from sklearn.decomposition import PCA

# Compute only the first 16 PCA components and ignore the rest
pca = PCA(n_components=16)
pca.fit(Xtraindata.transpose())
print(pca.components_.shape)

plot_digits(pca.components_.transpose(), 4, "First 16 PCA modes")


# %% [markdown]
# Cumulative energy analysis

# %%
# This time, compute all of the PCA components (ie. 784 of them)
pca = PCA()
pca.fit(Xtraindata.transpose())
print(pca.components_.shape)


# %%
# Compute the cumulative energy captured by the PCA modes
E = np.power(pca.singular_values_, 2)/np.sum(np.power(pca.singular_values_, 2))
E_cumsum = np.pad(np.cumsum(E), (1, 0))  # Pad with one zero for the sake of the plot

fig_singular_values, ax_singular_values = plt.subplots(1, 1)
ax_singular_values.plot(E_cumsum, label='Cumulative energy')
ax_singular_values.set_xlabel('k')
ax_singular_values.set_ylabel('$\Sigma E_k$')
fig_singular_values.suptitle('Energy captured in the first k modes')


# Compute the number of modes required to capture 85% of the energy
def find_energy_threshold(thresh):
    # Plot the threhold line
    ax_singular_values.hlines(thresh, 0, len(E_cumsum) - 1,
                              linestyles='dashed', colors='r',
                              label=f'{int(thresh*100)}% threshold')
    n_modes = next(i for i, k in enumerate(E_cumsum) if k >= thresh)
    ax_singular_values.vlines(n_modes, 0, 1, color='r')
    return n_modes
    

thresh = 0.85
n_modes = find_energy_threshold(thresh)
print(f'Number of PCA modes to capture {int(thresh*100)}% of energy in'
      f' training data: {n_modes}')

ax_singular_values.legend()


# %% [markdown]
# Reconstruct some images using the first 59 PCA modes

# %%
# Reconstruct training data using only the first n modes
print(f'Reconstructing training data using {n_modes} modes')

pca = PCA(n_components=n_modes)
X_train_reconstructed = pca.inverse_transform(
    pca.fit_transform(Xtraindata.transpose())).transpose()
print(X_train_reconstructed.shape)

# Plot the reconstructed digits
plot_digits(X_train_reconstructed, 8, "First 64 reconstructed training images, using {n_modes} modes")

# %%

# %%
