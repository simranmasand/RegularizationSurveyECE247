#import libraries
from utils import *

from time import sleep
#!pip install progress
from progress.bar import Bar
import progressbar
import torch
import torch.nn as nn
import torch.nn.functional as F #activation functions
from torch.optim import SGD
import seaborn as sns
import matplotlib.pyplot as plt
# Importing Libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import matplotlib.pyplot as plt
#%matplotlib inline
import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


device = "cuda" if torch.cuda.is_available else "cpu"
print(device)
import matplotlib.pyplot as plt

label_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

trainloader,validloader = get_train_valid_loader(data_dir='./data',
                           batch_size=4,
                           augment=False,
                           random_seed=0,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=True,
                           num_workers=0,
                           pin_memory=True)

##### download test data
testloader = get_test_loader(data_dir='./data',
                    batch_size=4,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True)


###### load model RESNET
model=torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False, trust_repo=True)
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_values = []
accuracies = []
val_accuracy = []
best_val_acc = 0
for epoch in range(10):  # loop over the dataset multiple times
    total = 0
    correct = 0

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        loss_values.append(loss.item())
        if i % 200 == 199:  # print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    # caculate accuracy
    train_correct, train_total, train_acc = predict_accuracy(model, trainloader)
    ## calculate validation accuracy
    val_correct, val_total, val_acc = predict_accuracy(model, validloader)
    accuracies.append(train_acc)
    val_accuracy.append(val_acc)

    print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (epoch+1, 10, train_acc, val_acc))

    # retain best parameters
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = model.state_dict()
        #save_checkpoint(model, 'resnet_adam_plain.pt', loss_values, 'loss_res_adam_plain.txt')

print('Finished Training')


####### save the best params

#model.load_state_dict('resnet_adam_plain.pt')


####### plot loss surface
plt.plot(loss_values,label='Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend();
plt.title('Resnet-20 Loss with Adam ')


###### plot accuracies
plt.plot(accuracies,label='Training');
plt.plot(val_accuracy,label='Validation');
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend();
plt.title('Resnet-20 Accuracy with Adam')


###### Calculate Testing accuracy

predict_accuracy(model, testloader)

