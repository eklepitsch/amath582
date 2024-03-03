#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
import random


# In[2]:


import torchvision
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from enum import Enum, auto


# In[3]:


# Use the following code to load and normalize the dataset for training and testing
# It will downlad the dataset into data subfolder (change to your data folder name)
train_dataset = torchvision.datasets.FashionMNIST('data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

test_dataset = torchvision.datasets.FashionMNIST('data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))


# Use the following code to create a validation set of 10%
train_indices, val_indices, _, _ = train_test_split(
    range(len(train_dataset)),
    train_dataset.targets,
    stratify=train_dataset.targets,
    test_size=0.1,
)

# Generate training and validation subsets based on indices
train_split = Subset(train_dataset, train_indices)
val_split = Subset(train_dataset, val_indices)


# set batches sizes
train_batch_size = 128 #Define train batch size
test_batch_size  = 256 #Define test batch size (can be larger than train batch size)


# Define dataloader objects that help to iterate over batches and samples for
# training, validation and testing
train_batches = DataLoader(train_split, batch_size=train_batch_size, shuffle=True)
val_batches = DataLoader(val_split, batch_size=train_batch_size, shuffle=True)
test_batches = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
                                           
num_train_batches=len(train_batches)
num_val_batches=len(val_batches)
num_test_batches=len(test_batches)


print(f"# Training batches: {num_train_batches}")
print(f"# Validation batches: {num_val_batches}")
print(f"# Test batches: {num_test_batches}")


#Sample code to visulaize the first sample in first 16 batches 

# batch_num = 0
# for train_features, train_labels in train_batches:
    
#     if batch_num == 16:
#         break    # break here
    
#     batch_num = batch_num +1
#     print(f"Feature batch shape: {train_features.size()}")
#     print(f"Labels batch shape: {train_labels.size()}")
    
#     img = train_features[0].squeeze()
#     label = train_labels[0]
#     plt.imshow(img, cmap="gray")
#     plt.show()
#     print(f"Label: {label}")



# # Sample code to plot N^2 images from the dataset
# def plot_images(XX, N, title):
#     fig, ax = plt.subplots(N, N, figsize=(8, 8))
    
#     for i in range(N):
#       for j in range(N):
#         ax[i,j].imshow(XX[(N)*i+j], cmap="Greys")
#         ax[i,j].axis("off")
#     fig.suptitle(title, fontsize=24)

# plot_images(train_dataset.data[:64], 8, "First 64 Training Images" )

    


# In[4]:


#Define your (As Cool As It Gets) Fully Connected Neural Network 
class ACAIGFCN(nn.Module):
    #Initialize model layers, add additional arguments to adjust
    def __init__(self, input_dim, output_dim, n_layers, layers_size): 
        super(ACAIGFCN, self).__init__()
        #Define the network layer(s) and activation function(s)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(input_dim, layers_size)])
        self.linears.extend([torch.nn.Linear(layers_size, layers_size) for i in range(1, n_layers-1)]) 
        self.linears.extend([torch.nn.Linear(layers_size, output_dim)])
 
    def forward(self, input):
        #Define how your model propagates the input through the network
        x = input
        for l in self.linears:
            x = F.relu(l(x))
        return x


# In[5]:


# Add dropout
class ACAIGFCN_dropout(nn.Module):
    #Initialize model layers, add additional arguments to adjust
    def __init__(self, input_dim, output_dim, n_layers, layers_size): 
        super(ACAIGFCN_dropout, self).__init__()
        #Define the network layer(s) and activation function(s)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(input_dim, layers_size)])
        for i in range(1, n_layers-1):
            # Add dropout in the hidden layers
            self.linears.extend([torch.nn.Dropout(0.2)])
            self.linears.extend([torch.nn.Linear(layers_size, layers_size)])
        self.linears.extend([torch.nn.Linear(layers_size, output_dim)])
 
    def forward(self, input):
        #Define how your model propagates the input through the network
        x = input
        for l in self.linears:
            x = F.relu(l(x))
        return x


# In[6]:


# Add initialization
class ACAIGFCN_init(nn.Module):
    #Initialize model layers, add additional arguments to adjust
    def __init__(self, input_dim, output_dim, n_layers, layers_size, initializer=torch.nn.init.normal_): 
        super(ACAIGFCN_init, self).__init__()
        #Define the network layer(s) and activation function(s)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(input_dim, layers_size)])
        self.linears.extend([torch.nn.Linear(layers_size, layers_size) for i in range(1, n_layers-1)]) 
        self.linears.extend([torch.nn.Linear(layers_size, output_dim)])

        # Add initialization
        for l in self.linears:
            initializer(l.weight)
 
    def forward(self, input):
        #Define how your model propagates the input through the network
        x = input
        for l in self.linears:
            x = F.relu(l(x))
        return x


# In[7]:


# Add batch normalization
class ACAIGFCN_batch_norm(nn.Module):
    #Initialize model layers, add additional arguments to adjust
    def __init__(self, input_dim, output_dim, n_layers, layers_size): 
        super(ACAIGFCN_batch_norm, self).__init__()
        #Define the network layer(s) and activation function(s)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(input_dim, layers_size)])
        for i in range(1, n_layers-1):
            # Add batch normalization after each layer
            self.linears.extend([torch.nn.BatchNorm1d(layers_size)])
            self.linears.extend([torch.nn.Linear(layers_size, layers_size)])
        self.linears.extend([torch.nn.BatchNorm1d(layers_size)])
        self.linears.extend([torch.nn.Linear(layers_size, output_dim)])
 
    def forward(self, input):
        #Define how your model propagates the input through the network
        x = input
        for l in self.linears:
            x = F.relu(l(x))
        return x


# In[8]:


# Set the random seed to ensure reproducible results
torch.manual_seed(0)
random.seed(0)

# Initialize neural network model with input, output and hidden layer dimensions
model = ACAIGFCN(input_dim = 784, output_dim = 10, n_layers=2, layers_size=50) #... add more parameters
                
# Define the learning rate and epochs number
learning_rate = 0.005
epochs = 30

train_loss_list = np.zeros((epochs,))
validation_accuracy_list = np.zeros((epochs,))
validation_std_list = np.zeros((epochs,))

# Define loss function  and optimizer
loss_func = torch.nn.CrossEntropyLoss() # Use Cross Entropy loss from torch.nn 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Use optimizers from torch.optim


# Iterate over epochs, batches with progress bar and train+ validate the ACAIGFCN
# Track the loss and validation accuracy
start_time = time.time()
for epoch in tqdm.trange(epochs):

    # ACAIGFCN Training 
    batch_loss_list = np.zeros((num_train_batches,))
    i = 0
    for train_features, train_labels in train_batches:
        # Set model into training mode
        model.train()
        
        # Reshape images into a vector
        train_features = train_features.reshape(-1, 28*28)

        # Reset gradients, Calculate training loss on model 
        # Perfrom optimization, back propagation
        train_outputs = model(train_features)
        loss = loss_func(train_outputs, train_labels)
        batch_loss_list[i] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i = i + 1
 
    # Record loss for the epoch
    train_loss_list[epoch] = np.mean(batch_loss_list)
    
    # ACAIGFCN Validation
    batch_acc_list = np.zeros((num_val_batches,))
    i = 0
    for val_features, val_labels in val_batches:
        
        # Telling PyTorch we aren't passing inputs to network for training purpose
        with torch.no_grad(): 
            model.eval()
            
             # Reshape validation images into a vector
            val_features = val_features.reshape(-1, 28*28)
          
            # Compute validation outputs (targets) 
            # and compute accuracy 
            validation_outputs = model(val_features)
            correct = (torch.argmax(validation_outputs, dim=1) == 
                       val_labels).type(torch.FloatTensor)

            val_acc = correct.mean()
            batch_acc_list[i] = val_acc
            i = i + 1

    validation_accuracy_list[epoch] = np.mean(batch_acc_list)
    validation_std_list[epoch] = np.std(batch_acc_list)
            
            
    # Record accuracy for the epoch; print training loss, validation accuracy
    # Record standard deviation too
    print(f"Epoch: {epoch}; Training loss: {train_loss_list[epoch]}")
    print(f"Epoch: {epoch}; Validation Accuracy: {validation_accuracy_list[epoch]*100}%")
    print(f"Epoch: {epoch}; Validation Std Dev: {validation_std_list[epoch]}")
    print(f"Elapsed training time: {(time.time() - start_time)/60} minutes")



# In[9]:


# Plot training loss and validation accuracy throughout the training epochs
fig, ax = plt.subplots(2, 1)

def plot_loss_and_accuracy(loss, accuracy, label):
    ax[0].plot(np.arange(1, len(loss) + 1), loss, label=label)
    ax[0].set_ylabel('Training Loss')
    ax[0].legend()
    
    ax[1].plot(np.arange(1, len(accuracy) + 1), accuracy * 100, label=label)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Validation Accuracy (%)')
    #ax[1].legend()
    
    fig.set_figwidth(12)
    fig.set_figheight(6)


# In[10]:


plot_loss_and_accuracy(train_loss_list, validation_accuracy_list, 'Baseline (SGD)')
fig


# In[11]:


import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 900
fig.savefig('images/baseline-learning-curves.png')


# In[12]:


#Calculate accuracy on test set

# Telling PyTorch we aren't passing inputs to network for training purpose
start_time = time.time()
with torch.no_grad():
    
    batch_acc_list = np.zeros((num_test_batches,))
    i = 0
    for test_features, test_labels in test_batches:

        model.eval()
        # Reshape test images into a vector
        test_features = test_features.reshape(-1, 28*28)

         # Compute validation outputs (targets) 
         # and compute accuracy 
        test_outputs = model(test_features)
        correct = (torch.argmax(test_outputs, dim=1) == test_labels).type(torch.FloatTensor)
        batch_acc_list[i] = correct.mean()
        i = i + 1
    
    # Compute total (mean) accuracy
    # Report total (mean) accuracy, can also compute std based on batches
    print(f"Test Accuracy: {np.mean(batch_acc_list)*100}%")
    print(f"Test Std Dev: {np.std(batch_acc_list)}")
    print(f"Elapsed test time: {(time.time() - start_time)/60} minutes")


# In[13]:


# Create a function to do the above more generally

# Capture the baseline parameters in a class
class BaselineParams:
    input_dim = 784
    output_dim = 10
    n_layers = 2
    layers_size = 100
    train_batch_size = 128
    test_batch_size = 256
    learning_rate = 0.005
    epochs = 30
    

class LossFn(Enum):
    CROSS_ENTROPY = auto()


class Optimizer(str, Enum):
    SGD = 'SGD'
    RMSPROP = 'RMSProp'
    ADAM = 'Adam'


def train_and_test_model(epochs = BaselineParams.epochs,
                         learning_rate=BaselineParams.learning_rate,
                         loss_function=LossFn.CROSS_ENTROPY,
                         opt=Optimizer.SGD,
                         dropout=False,
                         init=False,
                         initializer=None,
                         batch_norm=False):
    # Set the random seed to ensure reproducible results
    torch.manual_seed(0)
    random.seed(0)

    # Initialize the baseline neural network model
    if not dropout and not init and not batch_norm:
        model = ACAIGFCN(input_dim = BaselineParams.input_dim, output_dim = BaselineParams.output_dim, n_layers = BaselineParams.n_layers, layers_size = BaselineParams.layers_size)
    elif dropout and not init and not batch_norm:
        model = ACAIGFCN_dropout(input_dim = BaselineParams.input_dim, output_dim = BaselineParams.output_dim, n_layers = BaselineParams.n_layers, layers_size = BaselineParams.layers_size)
    elif not dropout and init and initializer is not None and not batch_norm:
        model = ACAIGFCN_init(input_dim = BaselineParams.input_dim, output_dim = BaselineParams.output_dim, n_layers = BaselineParams.n_layers, layers_size = BaselineParams.layers_size, initializer=initializer)
    elif not dropout and not init and batch_norm:
        model = ACAIGFCN_batch_norm(input_dim = BaselineParams.input_dim, output_dim = BaselineParams.output_dim, n_layers = BaselineParams.n_layers, layers_size = BaselineParams.layers_size)
        

    train_batches = DataLoader(train_split, batch_size=BaselineParams.train_batch_size, shuffle=True)
    val_batches = DataLoader(val_split, batch_size=BaselineParams.train_batch_size, shuffle=True)
    test_batches = DataLoader(test_dataset, batch_size=BaselineParams.test_batch_size, shuffle=True)
                                           
    num_train_batches=len(train_batches)
    num_val_batches=len(val_batches)
    num_test_batches=len(test_batches)

    train_loss_list = np.zeros((epochs,))
    validation_accuracy_list = np.zeros((epochs,))
    validation_std_list = np.zeros((epochs,))
    
    # Define loss function  and optimizer
    loss_func = None
    optimizer = None
    if loss_function == LossFn.CROSS_ENTROPY:
        loss_func = torch.nn.CrossEntropyLoss() 

    if opt == Optimizer.SGD: 
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif opt == Optimizer.RMSPROP:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif opt == Optimizer.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Iterate over epochs, batches with progress bar and train+ validate the ACAIGFCN
    # Track the loss and validation accuracy
    print(f"Reporting statistics for optimizer: {opt.value}, lr = {learning_rate}, dropout={dropout}, initializer={initializer}, batch_norm={batch_norm}")
    start_time = time.time()
    for epoch in tqdm.trange(epochs):
    
        # ACAIGFCN Training 
        batch_loss_list = np.zeros((num_train_batches,))
        i = 0
        for train_features, train_labels in train_batches:
            # Set model into training mode
            model.train()
            
            # Reshape images into a vector
            train_features = train_features.reshape(-1, 28*28)
    
            # Reset gradients, Calculate training loss on model 
            # Perfrom optimization, back propagation
            train_outputs = model(train_features)
            loss = loss_func(train_outputs, train_labels)
            batch_loss_list[i] = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i = i + 1
     
        # Record loss for the epoch
        train_loss_list[epoch] = np.mean(batch_loss_list)
        
        # ACAIGFCN Validation
        batch_acc_list = np.zeros((num_val_batches,))
        i = 0
        for val_features, val_labels in val_batches:
            
            # Telling PyTorch we aren't passing inputs to network for training purpose
            with torch.no_grad(): 
                model.eval()
                
                 # Reshape validation images into a vector
                val_features = val_features.reshape(-1, 28*28)
              
                # Compute validation outputs (targets) 
                # and compute accuracy 
                validation_outputs = model(val_features)
                correct = (torch.argmax(validation_outputs, dim=1) == 
                           val_labels).type(torch.FloatTensor)
    
                val_acc = correct.mean()
                batch_acc_list[i] = val_acc
                i = i + 1
    
        validation_accuracy_list[epoch] = np.mean(batch_acc_list)
        validation_std_list[epoch] = np.std(batch_acc_list)
                
        # Record accuracy for the epoch; print training loss, validation accuracy
        # Record standard deviation too
        if epoch == epochs-1:
            print(f"Epoch: {epoch}; Training loss: {train_loss_list[epoch]}")
            print(f"Epoch: {epoch}; Validation Accuracy: {validation_accuracy_list[epoch]*100}%")
            print(f"Epoch: {epoch}; Validation Std Dev: {validation_std_list[epoch]}")
            print(f"Elapsed training time: {(time.time() - start_time)/60} minutes")

    if init:
        plot_loss_and_accuracy(train_loss_list, validation_accuracy_list, label=f"initializer = {initializer}")
    else:
        plot_loss_and_accuracy(train_loss_list, validation_accuracy_list, label=f"{opt.value}, lr = {learning_rate}")

    # Compute test accuracy
    with torch.no_grad(): 
        batch_acc_list = np.zeros((num_test_batches,))
        i = 0
        for test_features, test_labels in test_batches:
    
            model.eval()
            # Reshape test images into a vector
            test_features = test_features.reshape(-1, 28*28)
    
             # Compute validation outputs (targets) 
             # and compute accuracy 
            test_outputs = model(test_features)
            correct = (torch.argmax(test_outputs, dim=1) == test_labels).type(torch.FloatTensor)
            batch_acc_list[i] = correct.mean()
            i = i + 1
        
        # Compute total (mean) accuracy
        # Report total (mean) accuracy, can also compute std based on batches
        test_accuracy = np.mean(batch_acc_list)
        test_std = np.std(batch_acc_list)
        print(f"Test Accuracy: {test_accuracy*100}%")
        print(f"Test Std Dev: {test_std}")
    
    return train_loss_list, validation_accuracy_list, validation_std_list, test_accuracy, test_std


# In[14]:


# Hyperparameter tuning
fig, ax = plt.subplots(2, 1)
SGD_loss_1, SGD_train_acc_1, SGD_train_std_1, SGD_test_acc_1, SGD_test_std_1 = train_and_test_model(opt=Optimizer.SGD, learning_rate=0.001)
SGD_loss_5, SGD_train_acc_5, SGD_train_std_5, SGD_test_acc_5, SGD_test_std_5 = train_and_test_model(opt=Optimizer.SGD, learning_rate=0.005)
SGD_loss_10, SGD_train_acc_10, SGD_train_std_10, SGD_test_acc_10, SGD_test_std_10 = train_and_test_model(opt=Optimizer.SGD, learning_rate=0.01)
RMS_loss_1, RMS_train_acc_1, RMS_train_std_1, RMS_test_acc_1, RMS_test_std_1 = train_and_test_model(opt=Optimizer.RMSPROP, learning_rate=0.001)
RMS_loss_5, RMS_train_acc_5, RMS_train_std_5, RMS_test_acc_5, RMS_test_std_5 = train_and_test_model(opt=Optimizer.RMSPROP, learning_rate=0.005)
RMS_loss_10, RMS_train_acc_10, RMS_train_std_10, RMS_test_acc_10, RMS_test_std_10 = train_and_test_model(opt=Optimizer.RMSPROP, learning_rate=0.01)
ADAM_loss_1, ADAM_train_acc_1, ADAM_train_std_1, ADAM_test_acc_1, ADAM_test_std_1 = train_and_test_model(opt=Optimizer.ADAM, learning_rate=0.001)
ADAM_loss_5, ADAM_train_acc_5, ADAM_train_std_5, ADAM_test_acc_5, ADAM_test_std_5 = train_and_test_model(opt=Optimizer.ADAM, learning_rate=0.005)
ADAM_loss_10, ADAM_train_acc_10, ADAM_train_std_10, ADAM_test_acc_10, ADAM_test_std_10 = train_and_test_model(opt=Optimizer.ADAM, learning_rate=0.01)
fig


# In[15]:


import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 900
fig.savefig('images/optimizer-learning-curves.png')


# In[16]:


# Rerun the baseline with dropout
fig, ax = plt.subplots(2, 1)
_, _, _, _, _ = train_and_test_model(opt=Optimizer.SGD, learning_rate=0.005, dropout=True)


# In[17]:


# Rerun the baseline with different initializers
fig, ax = plt.subplots(2, 1)
_, _, _, _, _ = train_and_test_model(opt=Optimizer.SGD, learning_rate=0.005, init=True, initializer=torch.nn.init.normal_)
_, _, _, _, _ = train_and_test_model(opt=Optimizer.SGD, learning_rate=0.005, init=True, initializer=torch.nn.init.xavier_normal_)
_, _, _, _, _ = train_and_test_model(opt=Optimizer.SGD, learning_rate=0.005, init=True, initializer=torch.nn.init.kaiming_uniform_)


# In[18]:


# Run the baseline with batch normalization
fig, ax = plt.subplots(2, 1)
_, _, _, _, _ = train_and_test_model(opt=Optimizer.SGD, learning_rate=0.005, batch_norm=True)
_, _, _, _, _ = train_and_test_model(opt=Optimizer.ADAM, learning_rate=0.001, batch_norm=True)


# In[ ]:




