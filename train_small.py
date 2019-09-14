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
batch_size = 11   # was 16

ds = ImagesDS(trainFile, path_data, useBothSites=True)#, useOnly=500)
#ds_train, ds_val = trainTestSplit(ds, val_share=0.1468024)
ds_train, ds_val = trainTestSplit(ds, val_share=0.125)
#ds_train, ds_val = trainTestSplit(ds, val_share=0.02436053)

#ds2 = ImagesDS(trainFile, path_data, useBothSites=False,mode='val')
#_, ds_val = trainTestSplit(ds2, val_share=0.125)

classes = 61 # 30 - HUVEC30 # controls 31 # 61 - HUVEC+CONTROLS # 1108

model = DensNet(num_classes=classes, pretrained=True)
if modelFile != 'none':
    model.load_state_dict(torch.load(f'{modelFile}'))
    model.eval()

# set drop rate = 0.5
if 1==0:
    DROP_RATE = 0.3
    for idx1, m in enumerate(model.named_children()):
        if m[0]=='features':
            #print(idx1, '->', m[1])
            for idx2, n in enumerate(m[1].named_children()):
                if 'denseblock' in n[0]:
                    #print(idx2, '->', n[0])
                    for idx3, o in enumerate(n[1].named_children()):
                        #print(idx3, '->', o[0])
                        o[1].drop_rate=DROP_RATE


model.to(device)

loader = D.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
vloader = D.DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=4)
#tloader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)

dataLoaders = {'train': loader, 'val':vloader}
dataset_sizes = {'train':len(ds_train), 'val':len(ds_val)}
del loader, vloader


criterion = nn.BCEWithLogitsLoss()
#criterion = nn.CrossEntropyLoss()

#optimizer = torch.optim.Adam(model.parameters(), lr=1.5625e-05, weight_decay=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01, momentum=0)

# Decay LR by a factor of 0.5 every 5 epochs
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

                    #loss = criterion(outputs, labels)
                    target = torch.zeros_like(outputs, device=device)
                    target[np.arange(inputs[0].size(0)), labels] = 1
                    loss = criterion(outputs, target)

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
torch.save(model.state_dict(), f'final_model_{groupCode}.bin')
