import numpy as np 
import time

import torch
import torch.nn as nn
import torch.utils.data as D

# my imports
from torch.optim import lr_scheduler
import copy

import tqdm

# my classes
from ImagesDS import ImagesDS
from trainTestSplit import trainTestSplit
from DensNet import DensNet

import warnings
warnings.filterwarnings('ignore')

import sys

groupCode = sys.argv[1]
modelFile = sys.argv[2]
trainFile = sys.argv[3]
Nepochs = sys.argv[4]

path_data = '../../input'
device = 'cuda'
batch_size = 32   # was 52 for densenet121

ds = ImagesDS(trainFile, path_data, useBothSites=True)#, useOnly=500)
ds_train, ds_val = trainTestSplit(ds, val_share=0.01)
#ds_train, ds_val = trainTestSplit(ds, val_share=0.146829)

classes = 1108 # 30 - HUVEC30 # controls 31 # 61 - HUVEC+CONTROLS # 1108

model = DensNet(num_classes=classes, pretrained=True)
if modelFile != 'none':
    model.load_state_dict(torch.load(f'{modelFile}'))
    model.eval()
model.to(device)

def worker_init_fn(worker_id):                                                          
     np.random.seed(np.random.get_state()[1][0] + worker_id)

loader = D.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
vloader = D.DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)

dataLoaders = {'train': loader, 'val':vloader}
dataset_sizes = {'train':len(ds_train), 'val':len(ds_val)}
del loader, vloader


#criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

lr=0.001#0.00015
#optimizer = torch.optim.Adam(model.parameters(), lr=1.5625e-05, weight_decay=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01, momentum=0)

# Decay LR by a factor of 0.75 every 5 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    USEBOTHSITES = 0

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10000000.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode

            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            batch = 0

            # Iterate over data.
            for inputs, labels in dataLoaders[phase]:

                inputs[0] = inputs[0].to(device)
                inputs[1] = inputs[1].to(device)
                if USEBOTHSITES==1:
                    inputs[2] = inputs[2].to(device)
                #print(labels)
                labels = labels.to(device)

                if batch%1000==0:
                    print(f"Batch: {batch} {time.ctime()} {inputs[0].shape} number of samples:{dataset_sizes[phase]} lr:{optimizer.param_groups[0]['lr']}")
                batch += 1

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)
                    #target = torch.zeros_like(outputs, device=device)
                    #target[np.arange(inputs[0].size(0)), labels] = 1
                    #loss = criterion(outputs, target)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs[0].size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                #best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f"best_model_regular_loss{groupCode}.bin")
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), f"best_model_regular_acc{groupCode}.bin")

            if phase == 'val':
                torch.save(model.state_dict(), f'final_model_{groupCode}.bin')
                scheduler.step()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model


model = train_model(model, criterion, optimizer, scheduler, num_epochs=int(Nepochs))
#torch.save(model.state_dict(), f'final_model_{groupCode}.bin')
