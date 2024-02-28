#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import tqdm


# In[55]:


import torchvision
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


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
train_batch_size = 512 #Define train batch size
test_batch_size  = 256 #Define test batch size (can be larger than train batch size)


# Define dataloader objects that help to iterate over batches and samples for
# training, validation and testing
train_batches = DataLoader(train_split, batch_size=train_batch_size, shuffle=True)
val_batches = DataLoader(val_split, batch_size=train_batch_size, shuffle=True)
test_batches = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
                                           
num_train_batches=len(train_batches)
num_val_batches=len(val_batches)
num_test_batches=len(test_batches)


print(num_train_batches)
print(num_val_batches)
print(num_test_batches)


#Sample code to visulaize the first sample in first 16 batches 

batch_num = 0
for train_features, train_labels in train_batches:
    
    if batch_num == 16:
        break    # break here
    
    batch_num = batch_num +1
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")



# Sample code to plot N^2 images from the dataset
def plot_images(XX, N, title):
    fig, ax = plt.subplots(N, N, figsize=(8, 8))
    
    for i in range(N):
      for j in range(N):
        ax[i,j].imshow(XX[(N)*i+j], cmap="Greys")
        ax[i,j].axis("off")
    fig.suptitle(title, fontsize=24)

plot_images(train_dataset.data[:64], 8, "First 64 Training Images" )

    


# In[52]:


#Define your (As Cool As It Gets) Fully Connected Neural Network 
class ACAIGFCN(nn.Module):
    #Initialize model layers, add additional arguments to adjust
    def __init__(self, input_dim, output_dim): 
        super(ACAIGFCN, self).__init__()
        #Define the network layer(s) and activation function(s)
        self.layer1 = torch.nn.Linear(input_dim, 1000)
        self.layer2 = torch.nn.Linear(1000, 1000)
        self.layer3 = torch.nn.Linear(1000, 2000)
        self.layer4 = torch.nn.Linear(2000, 1000)
        self.layer5 = torch.nn.Linear(1000, output_dim)

        #self.layer1 = torch.nn.Linear(input_dim, output_dim)
 
    def forward(self, input):
        #Define how your model propagates the input through the network
        x = F.relu(self.layer1(input))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.layer5(x)

        #x = self.layer1(input)
        return x


# In[ ]:


# Initialize neural network model with input, output and hidden layer dimensions
model = ACAIGFCN(input_dim = 784, output_dim = 10) #... add more parameters
                
# Define the learning rate and epochs number
learning_rate = 0.01
epochs = 50


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
    print(f"Elapsed training time: {time.time() - start_time} seconds")



# In[51]:


# Plot training loss and validation accuracy throughout the training epochs
fig, ax = plt.subplots(1, 2)
ax[0].plot(np.arange(1, len(train_loss_list) + 1), train_loss_list)
ax[0].set_title('Training loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')

ax[1].plot(np.arange(1, len(train_loss_list) + 1), validation_accuracy_list * 100)
ax[1].set_title('Validation Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy (%)')

fig.set_figwidth(10)


# In[7]:


#Calculate accuracy on test set

# Telling PyTorch we aren't passing inputs to network for training purpose
with torch.no_grad():
    
    for test_features, test_labels in test_batches:

        model.eval()
        # Reshape test images into a vector
        test_features = test_features.reshape(-1, 28*28)

         # Compute validation outputs (targets) 
         # and compute accuracy 
    
    # Compute total (mean) accuracy
    # Report total (mean) accuracy, can also compute std based on batches


# In[ ]:




