#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
import random


# In[4]:


import torchvision
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from enum import Enum, auto


# In[5]:


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

    


# In[104]:


# Define baseline FCN
class FCN(nn.Module):
    #Initialize model layers, add additional arguments to adjust
    def __init__(self, input_dim, output_dim, n_layers, layers_size): 
        super(FCN, self).__init__()
        #Define the network layer(s) and activation function(s)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(input_dim, layers_size)])
        self.linears.extend([torch.nn.Linear(layers_size, layers_size) for i in range(0, n_layers-1)]) 
        self.linears.extend([torch.nn.Linear(layers_size, output_dim)])
 
    def forward(self, input):
        #Define how your model propagates the input through the network
        x = input
        for l in self.linears[:-1]:
            x = F.relu(l(x))
        x = self.linears[-1](x)
        return x


# In[111]:


# Find the size of the model
model = FCN(input_dim = 784, output_dim = 10, n_layers=2, layers_size=200)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {total_params}")


# In[29]:


from torch.nn import Conv2d, ReLU, MaxPool2d, Linear, LogSoftmax
from torch import flatten

# Define the baseline CNN
# Adapted from https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
class CNN_100K(nn.Module):
    def __init__(self, num_channels, classes): 
        super(CNN_100K, self).__init__()
        # Convolution + pooling layer 1
        self.conv1 = Conv2d(in_channels=num_channels, out_channels=20,
            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Convolution + pooling layer 2
        self.conv2 = Conv2d(in_channels=20, out_channels=40,
            kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Fully connected layer 1
        self.fc1 = Linear(in_features=640, out_features=120)
        self.relu3 = ReLU()
        # Fully connected layer 2
        self.fc2 = Linear(in_features=120, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)
 
    def forward(self, input):
        # Convolution + pooling layer 1
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # Convolution + pooling layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # Fully connected layer 1
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # Fully connected layer 2
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return output


# In[19]:


# Find the size of the model
model = CNN_100K(1, 10)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {total_params}")


# In[56]:


class CNN_50K(nn.Module):
    def __init__(self, num_channels, classes): 
        super(CNN_50K, self).__init__()
        # Convolution + pooling layer 1
        self.conv1 = Conv2d(in_channels=num_channels, out_channels=10,
            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Convolution + pooling layer 2
        self.conv2 = Conv2d(in_channels=10, out_channels=20,
            kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Fully connected layer 1
        self.fc1 = Linear(in_features=320, out_features=130)
        self.relu3 = ReLU()
        # Fully connected layer 2
        self.fc2 = Linear(in_features=130, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)
 
    def forward(self, input):
        # Convolution + pooling layer 1
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # Convolution + pooling layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # Fully connected layer 1
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # Fully connected layer 2
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return output


# In[25]:


# Find the size of the model
model = CNN_50K(1, 10)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {total_params}")


# In[71]:


class CNN_20K(nn.Module):
    def __init__(self, num_channels, classes): 
        super(CNN_20K, self).__init__()
        # Convolution + pooling layer 1
        self.conv1 = Conv2d(in_channels=num_channels, out_channels=10,
            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Convolution + pooling layer 2
        self.conv2 = Conv2d(in_channels=10, out_channels=20,
            kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Fully connected layer 1
        self.fc1 = Linear(in_features=320, out_features=40)
        self.relu3 = ReLU()
        # Fully connected layer 2
        self.fc2 = Linear(in_features=40, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)
 
    def forward(self, input):
        # Convolution + pooling layer 1
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # Convolution + pooling layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # Fully connected layer 1
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # Fully connected layer 2
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return output


# In[72]:


# Find the size of the model
model = CNN_20K(1, 10)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {total_params}")


# In[75]:


class CNN_10K(nn.Module):
    def __init__(self, num_channels, classes): 
        super(CNN_10K, self).__init__()
        # Convolution + pooling layer 1
        self.conv1 = Conv2d(in_channels=num_channels, out_channels=10,
            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Convolution + pooling layer 2
        self.conv2 = Conv2d(in_channels=10, out_channels=20,
            kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # Fully connected layer 1
        self.fc1 = Linear(in_features=320, out_features=15)
        self.relu3 = ReLU()
        # Fully connected layer 2
        self.fc2 = Linear(in_features=15, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)
 
    def forward(self, input):
        # Convolution + pooling layer 1
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # Convolution + pooling layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # Fully connected layer 1
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # Fully connected layer 2
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return output


# In[76]:


# Find the size of the model
model = CNN_10K(1, 10)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {total_params}")


# In[109]:


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


# In[112]:


# Create a function to train the NN generally

# Capture the baseline parameters in a class
class BaselineParams:
    input_dim = 784
    output_dim = 10
    n_layers = 2
    layers_size = 100
    train_batch_size = 64
    test_batch_size = 128
    learning_rate = 0.005
    epochs = 15
    

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
                         n_layers=BaselineParams.n_layers,
                         layers_size=BaselineParams.layers_size,
                         plot_label='None',
                         dropout=False,
                         init=False,
                         initializer=None,
                         batch_norm=False):
    # Set the random seed to ensure reproducible results
    torch.manual_seed(0)
    random.seed(0)

    # Initialize the baseline neural network model
    if not dropout and not init and not batch_norm:
        model = FCN(input_dim = BaselineParams.input_dim, output_dim = BaselineParams.output_dim, n_layers = n_layers, layers_size = layers_size)
    elif dropout and not init and not batch_norm:
        model = FCN_dropout(input_dim = BaselineParams.input_dim, output_dim = BaselineParams.output_dim, n_layers = n_layers, layers_size = layers_size)
    elif not dropout and init and initializer is not None and not batch_norm:
        model = FCN_init(input_dim = BaselineParams.input_dim, output_dim = BaselineParams.output_dim, n_layers = n_layers, layers_size = layers_size, initializer=initializer)
    elif not dropout and not init and batch_norm:
        model = FCN_batch_norm(input_dim = BaselineParams.input_dim, output_dim = BaselineParams.output_dim, n_layers = n_layers, layers_size = layers_size)
        

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
    print(f"Reporting statistics for optimizer: {opt.value}, lr = {learning_rate}, n_layers={n_layers}, layers_size={layers_size}, dropout={dropout}, initializer={initializer}, batch_norm={batch_norm}")
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
        plot_loss_and_accuracy(train_loss_list, validation_accuracy_list, label=plot_label)

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


# In[113]:


_,_,_,_,_ = train_and_test_model(opt=Optimizer.ADAM, learning_rate=0.001, n_layers=2, layers_size=60, plot_label='FCN 50K')
_,_,_,_,_ = train_and_test_model(opt=Optimizer.ADAM, learning_rate=0.001, n_layers=2, layers_size=110, plot_label='FCN 100K')
_,_,_,_,_ = train_and_test_model(opt=Optimizer.ADAM, learning_rate=0.001, n_layers=2, layers_size=200, plot_label='FCN 200K')


# In[114]:


fig


# In[115]:


import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 900
fig.savefig('images/FCN-learning-curves.png')


# In[90]:


# Create a function to train the CNN generally

# Capture the baseline parameters in a class
class BaselineParams:
    num_channels = 1
    classes = 10
    train_batch_size = 64
    test_batch_size = 128
    learning_rate = 0.005
    epochs = 15
    

class LossFn(Enum):
    CROSS_ENTROPY = auto()


class Optimizer(str, Enum):
    SGD = 'SGD'
    RMSPROP = 'RMSProp'
    ADAM = 'Adam'


def train_and_test_model_CNN(epochs = BaselineParams.epochs,
                             learning_rate=BaselineParams.learning_rate,
                             loss_function=LossFn.CROSS_ENTROPY,
                             opt=Optimizer.SGD,
                             plot_label='None',
                             network='100K',
                             dropout=False,
                             init=False,
                             initializer=None,
                             batch_norm=False):
    # Set the random seed to ensure reproducible results
    torch.manual_seed(0)
    random.seed(0)

    # Initialize the baseline neural network model
    if network == '10K':
        model = CNN_10K(num_channels = BaselineParams.num_channels, classes = BaselineParams.classes)
    elif network == '20K':
        model = CNN_20K(num_channels = BaselineParams.num_channels, classes = BaselineParams.classes)
    elif network == '50K':
        model = CNN_50K(num_channels = BaselineParams.num_channels, classes = BaselineParams.classes)
    elif network == '100K':
        model = CNN_100K(num_channels = BaselineParams.num_channels, classes = BaselineParams.classes)
    else:
        raise AssertionError('Invalid network name')

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
        #loss_func = torch.nn.CrossEntropyLoss() 
        loss_func = torch.nn.NLLLoss()

    if opt == Optimizer.SGD: 
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif opt == Optimizer.RMSPROP:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif opt == Optimizer.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Iterate over epochs, batches with progress bar and train+ validate the ACAIGFCN
    # Track the loss and validation accuracy
    print(f"Reporting statistics for network: {network}, optimizer: {opt.value}, lr = {learning_rate}, dropout={dropout}, initializer={initializer}, batch_norm={batch_norm}")
    start_time = time.time()
    for epoch in tqdm.trange(epochs):
    
        # ACAIGFCN Training 
        batch_loss_list = np.zeros((num_train_batches,))
        i = 0
        for train_features, train_labels in train_batches:
            # Set model into training mode
            model.train()
            
            # Reshape images into a vector
            #train_features = train_features.reshape(-1, 28*28)
    
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
                 #val_features = val_features.reshape(-1, 28*28)
              
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
        plot_loss_and_accuracy(train_loss_list, validation_accuracy_list, label=plot_label)

    # Compute test accuracy
    with torch.no_grad(): 
        batch_acc_list = np.zeros((num_test_batches,))
        i = 0
        for test_features, test_labels in test_batches:
    
            model.eval()
            # Reshape test images into a vector
            #test_features = test_features.reshape(-1, 28*28)
    
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


# In[92]:


_,_,_,_,_ = train_and_test_model_CNN(network='10K', opt=Optimizer.ADAM, learning_rate=0.001, plot_label='CNN 10K')
_,_,_,_,_ = train_and_test_model_CNN(network='20K', opt=Optimizer.ADAM, learning_rate=0.001, plot_label='CNN 20K')
_,_,_,_,_ = train_and_test_model_CNN(network='50K', opt=Optimizer.ADAM, learning_rate=0.001, plot_label='CNN 50K')
_,_,_,_,_ = train_and_test_model_CNN(network='100K', opt=Optimizer.ADAM, learning_rate=0.001,plot_label='CNN 100K')


# In[93]:


fig


# In[94]:


import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 900
fig.savefig('images/CNN-learning-curves.png')


# In[99]:


# Plot the training time or test accuracy vs the number of weights in the CNN
n_weights = [10, 20, 50, 100]
train_time = [4.53, 4.64, 4.75, 6.01]
test_accuracy = [89.34, 89.74, 89.78, 91.01]

fig_cnn_training, ax_cnn_training = plt.subplots(1, 1)
ax_cnn_training.plot(n_weights, test_accuracy)


# In[ ]:




