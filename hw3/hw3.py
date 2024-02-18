#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib as mpl

# Set to True to save the figures to .png files
save_figures = True
image_dir_name = 'images'
image_dir = None
if save_figures:
    # Bump up the resolution (adds processing time)
    mpl.rcParams['figure.dpi'] = 900
    image_dir = os.path.join(os.curdir, image_dir_name)
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

# Import the MNIST dataset.

# In[1]:


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


# A function to plot the 28x28 images in the MNIST dataset:

# In[2]:


def plot_digits(XX, N, title):
    fig, ax = plt.subplots(N, N, figsize=(N, N))
    
    for i in range(N):
      for j in range(N):
        ax[i,j].imshow(XX[:,(N)*i+j].reshape((28, 28)), cmap="Greys")
        ax[i,j].axis("off")
    if title:
        fig.suptitle(title, fontsize=24)

    return fig, ax


# In[3]:


# Plot the first 64 images in the dataset
plot_digits(Xtraindata, 8, "First 64 Training Images" )


# Find the first 16 PC modes and plot them.

# In[4]:


from sklearn.decomposition import PCA

# Compute only the first 16 PCA components and ignore the rest
pca = PCA(n_components=16)
pca.fit(Xtraindata.transpose())
print(pca.components_.shape)

fig, ax = plot_digits(pca.components_.transpose(), 4, None)
if save_figures:
    fig.savefig(os.path.join(image_dir, 'first-16-pca-modes.png'))


# Cumulative energy analysis:

# In[5]:


# This time, compute all of the PCA components (ie. 784 of them)
pca = PCA()
pca.fit(Xtraindata.transpose())
print(pca.components_.shape)


# In[6]:


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
    ax_singular_values.text(85, 0.3, f'k = {n_modes}')
    return n_modes
    

thresh = 0.85
n_modes = find_energy_threshold(thresh)
print(f'Number of PCA modes to capture {int(thresh*100)}% of energy in'
      f' training data: {n_modes}')

ax_singular_values.legend()
if save_figures:
    fig_singular_values.savefig(os.path.join(image_dir, 'energy-analysis.png'), bbox_inches='tight')


# Reconstruct some images using the first 59 PCA modes:

# In[7]:


# Reconstruct training data using only the first n modes
print(f'Reconstructing training data using {n_modes} modes')

pca = PCA(n_components=n_modes)
X_train_reconstructed = pca.inverse_transform(
    pca.fit_transform(Xtraindata.transpose())).transpose()
print(X_train_reconstructed.shape)

# Plot the reconstructed digits
fig, ax = plot_digits(X_train_reconstructed, 8, None)

if save_figures:
    fig.savefig(os.path.join(image_dir, 'reconstructed-images.png'))


# Write a function that selects a subset of particular digits:

# In[8]:


def get_all_samples_of_digit(d):
    # d is one of the digits 0-9
    training_indices = [i for i, k in enumerate(ytrainlabels) if k == d]
    training_samples = np.empty((Xtraindata.shape[0], len(training_indices)))
    #print(training_samples.shape)
    for out_idx, in_idx in enumerate(training_indices):
        training_samples[:, out_idx] = Xtraindata[:, in_idx]

    training_labels = [k for k in ytrainlabels if k == d]

    testing_indices = [i for i, k in enumerate(ytestlabels) if k == d]
    testing_samples = np.empty((Xtestdata.shape[0], len(testing_indices)))
    #print(testing_samples.shape)
    for out_idx, in_idx in enumerate(testing_indices):
        testing_samples[:, out_idx] = Xtestdata[:, in_idx]

    testing_labels = [k for k in ytestlabels if k == d]

    return training_samples, training_labels, testing_samples, testing_labels



def get_all_samples_of_digits(digits):
    # digits is an array of one or more digits
    X_subtrain = np.empty((Xtraindata.shape[0], 0))
    y_subtrain = np.empty((0))
    X_subtest = np.empty((Xtraindata.shape[0], 0))
    y_subtest = np.empty((0))

    # Populate the arrays
    for d in digits:
        xx_subtrain, yy_subtrain, xx_subtest, yy_subtest = \
            get_all_samples_of_digit(d)

        X_subtrain = np.hstack([X_subtrain, xx_subtrain])
        y_subtrain = np.hstack([y_subtrain, yy_subtrain])
        X_subtest = np.hstack([X_subtest, xx_subtest])
        y_subtest = np.hstack([y_subtest, yy_subtest])

    return X_subtrain, y_subtrain, X_subtest, y_subtest
        

# Test to make sure it works
# for d in range(0, 10):
#     X_subtrain, y_subtrain, X_subtest, y_subtest = get_all_samples_of_digit(d)
#     plot_digits(X_subtrain, 8, f'First 64 samples of digit {d}')    


# Select digits 1,8 and project into 59-PCA space:

# In[32]:


X_subtrain, y_subtrain, X_subtest, y_subtest = get_all_samples_of_digits([1, 8])

print(n_modes)
pca = PCA(n_components=n_modes)
X_subtrain_projected = pca.fit_transform(X_subtrain.transpose())
X_subtest_projected = pca.transform(X_subtest.transpose())
print(X_subtrain_projected.shape)
print(X_subtest_projected.shape)
print(y_subtrain.shape)
print(y_subtest.shape)


# Apply Ridge classifier and do cross validation:

# In[33]:


from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

clf = RidgeClassifier()
estimator = clf.fit(X_subtrain_projected, y_subtrain)
pred = clf.predict(X_subtest_projected)

accuracy = accuracy_score(y_subtest, pred)
print(f'Test accuracy: {accuracy}')

score = cross_val_score(estimator, X_subtrain_projected, y_subtrain)
print(f'Training accuracies, cross validation: {score}')
print(f'Training accuracy, mean: {np.mean(score)}')
print(f'Training accuracy, std dev: {np.std(score)}')

cm = confusion_matrix(y_subtrain, clf.predict(X_subtrain_projected))
ConfusionMatrixDisplay(cm, display_labels=[1, 8]).plot()


# Create a function to apply a classifier more generally:

# In[34]:


from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate, cross_val_score

def do_classification(digits, classifier, k_modes, plot_cm=False):
    """ Perform Ridge classification on the given digits.
    Args:
        digits: list of digits to classify
        classifier: a classifier object (RidgeClassifier,
                    KNeighborsClassifier, etc)
        k_modes: the number of modes in PCA space to use
                 for the classifier
        plot_cm: Whether to plot the confusion matrix
    """
    X_subtrain, y_subtrain, X_subtest, y_subtest = get_all_samples_of_digits(digits)

    pca = PCA(n_components=k_modes)
    X_subtrain_projected = pca.fit_transform(X_subtrain.transpose())
    X_subtest_projected = pca.transform(X_subtest.transpose())

    estimator = classifier.fit(X_subtrain_projected, y_subtrain)
    pred = classifier.predict(X_subtest_projected)
    testing_accuracy = accuracy_score(y_subtest, pred)

    training_accuracies = cross_val_score(estimator, X_subtrain_projected, y_subtrain)
    training_mean = np.mean(training_accuracies)
    training_std = np.std(training_accuracies)

    cm_train = confusion_matrix(y_subtrain, classifier.predict(X_subtrain_projected))
    cm_train_display = ConfusionMatrixDisplay(cm_train, display_labels=digits)
    cm_test = confusion_matrix(y_subtest, classifier.predict(X_subtest_projected))
    cm_test_display = ConfusionMatrixDisplay(cm_test, display_labels=digits)
    if plot_cm:
        cm_train_display.plot()
        cm_test_display.plot()
    
    return training_accuracies, testing_accuracy, training_mean, training_std, cm_train_display, cm_test_display


# In[23]:


from sklearn.linear_model import RidgeClassifier

digits = ['1, 8', '3, 8', '2, 7']
test_accuracies = []
training_accuracies = []
training_stds = []

_, test_accuracy, mean, std_dev, _, _  = do_classification([1, 8], RidgeClassifier(), n_modes)
print(f'Test accuracy for classication of [1, 8] ({n_modes}-PCA): {test_accuracy}')
print(f'Training accuracy and std dev: {mean}, {std_dev}')

test_accuracies.append(test_accuracy)
training_accuracies.append(mean)
training_stds.append(std_dev)

_, test_accuracy, mean, std_dev, _, _ = do_classification([3, 8], RidgeClassifier(), n_modes)
print(f'Test accuracy for classication of [3, 8] ({n_modes}-PCA): {accuracy}')
print(f'Training accuracy and std dev: {mean}, {std_dev}')

test_accuracies.append(test_accuracy)
training_accuracies.append(mean)
training_stds.append(std_dev)

_, test_accuracy, mean, std_dev, _, _ = do_classification([2, 7], RidgeClassifier(), n_modes)
print(f'Test accuracy for classication of [2, 7] ({n_modes}-PCA): {accuracy}')
print(f'Training accuracy and std dev: {mean}, {std_dev}')

test_accuracies.append(test_accuracy)
training_accuracies.append(mean)
training_stds.append(std_dev)


# In[13]:


# Define a function to round floating point results to a specified precision
def float_formatter(x):
    """
    Specify the precision and notation (positional vs. scientific)
    for floating point values.
    """
    p = 6
    if abs(x) < 1e-4 or abs(x) > 1e4:
        return np.format_float_scientific(x, precision=p)
    else:
        return np.format_float_positional(x, precision=p)


# In[35]:


# Create a table with the binary classification results
results_table = []
for digits, train_mean, train_std, test_accuracy in zip(digits,
                                                        training_accuracies,
                                                        training_stds,
                                                        test_accuracies):
    results_table.append([digits, float_formatter(train_mean),
                          float_formatter(train_std), float_formatter(test_accuracy)])

fig_results, ax_results = plt.subplots()
table = ax_results.table(cellText=results_table, colLabels=['Digits to classify', 
                                                            'Training accuracy (mean)', 
                                                            'Training std dev', 
                                                            'Test accuracy'],
                        loc='center')
ax_results.axis('off')
table.scale(1, 3)
fig_results.set_figheight(2)

if save_figures:
    fig_results.savefig(os.path.join(image_dir, 'binary-classification-results.png'), bbox_inches='tight')

# Use all the digits (muticlass classification)

# In[36]:


from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

classifiers = ['Ridge', 'KNN', 'LDA']
test_accuracies = []
training_accuracies = []
training_stds = []

fig_cm, ax_cm = plt.subplots(2, 3)

_, test_accuracy, mean, std_dev, cm_train, cm_test  = do_classification([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                          RidgeClassifier(),
                                                          n_modes)
print(f'Test accuracy for Ridge classication of all digits ({n_modes}-PCA): {test_accuracy}')
print(f'Training accuracy and std dev: {mean}, {std_dev}')
print('\n')

test_accuracies.append(test_accuracy)
training_accuracies.append(mean)
training_stds.append(std_dev)

ax_cm[0][0].set_title(f'Ridge')
cm_train.plot(ax=ax_cm[0][0], colorbar=False)
cm_test.plot(ax=ax_cm[1][0], colorbar=False)

_, test_accuracy, mean, std_dev, cm_train, cm_test  = do_classification([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                          KNeighborsClassifier(),
                                                          n_modes)
print(f'Test accuracy for KNN classication of all digits ({n_modes}-PCA): {test_accuracy}')
print(f'Training accuracy and std dev: {mean}, {std_dev}')
print('\n')

test_accuracies.append(test_accuracy)
training_accuracies.append(mean)
training_stds.append(std_dev)

ax_cm[0][1].set_title(f'KNN')
cm_train.plot(ax=ax_cm[0][1], colorbar=False)
cm_test.plot(ax=ax_cm[1][1], colorbar=False)

_, test_accuracy, mean, std_dev, cm_train, cm_test  = do_classification([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                          LinearDiscriminantAnalysis(),
                                                          n_modes)
print(f'Test accuracy for LDA classication of all digits ({n_modes}-PCA): {test_accuracy}')
print(f'Training accuracy and std dev: {mean}, {std_dev}')

test_accuracies.append(test_accuracy)
training_accuracies.append(mean)
training_stds.append(std_dev)

ax_cm[0][2].set_title(f'LDA')
cm_train.plot(ax=ax_cm[0][2], colorbar=False)
cm_test.plot(ax=ax_cm[1][2], colorbar=False)

ax_cm[0][0].set_ylabel(f'Train\n\nTrue label')
ax_cm[1][0].set_ylabel(f'Test\n\nTrue label')

fig_cm.suptitle('Confusion matrices for various classifiers')
fig_cm.set_figwidth(15)
fig_cm.set_figheight(10)

if save_figures:
    fig_cm.savefig(os.path.join(image_dir, 'multiclass-confusion-matrices.png'))


# In[37]:


# Create a table with the multiclass classification results
results_table = []
for classifier, train_mean, std_dev, test_accuracy in zip(classifiers,
                                                            training_accuracies,
                                                            training_stds,
                                                            test_accuracies):
    results_table.append([classifier, float_formatter(train_mean),
                          float_formatter(std_dev), float_formatter(test_accuracy)])

fig_results, ax_results = plt.subplots()
table = ax_results.table(cellText=results_table, colLabels=['Classifier', 
                                                            'Training accuracy (mean)', 
                                                            'Training std dev', 
                                                            'Test accuracy'],
                        loc='center')
ax_results.axis('off')
table.scale(1, 3)
fig_results.set_figheight(2)

if save_figures:
    fig_results.savefig(os.path.join(image_dir, 'multiclass-classification-results.png'), bbox_inches='tight')

# In[ ]:




