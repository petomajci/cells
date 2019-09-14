import numpy as np 
import pandas as pd
import time

from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F

import torchvision
from torchvision import transforms as T

# my classes
from cosface import LMCL_loss
from ImagesDS import ImagesDS
from trainTestSplit import trainTestSplit
from DensNet_forCosFace import DensNet

# my imports
from torch.optim import lr_scheduler
import copy

import tqdm

import warnings
warnings.filterwarnings('ignore')

import sys

groupCode = sys.argv[1]
modelFile = sys.argv[2]
trainFile = sys.argv[3]
Nepochs = int(sys.argv[4])

####

path_data = '../../input'
device = 'cuda'
batch_size = 22

###

ds = ImagesDS(f'{trainFile}', path_data, useBothSites=True)#, useOnly=5536)
#ds_train, ds_val = trainTestSplit(ds, val_share=0.1512762)
ds_train, ds_val = trainTestSplit(ds, val_share=0.1)

###

classes = 31 #1108
model = DensNet(num_classes=classes, pretrained=False)
if modelFile != 'none':
    model.load_state_dict(torch.load(f'{modelFile}'))
    model.eval();  # important to set dropout and normalization layers

model.to(device);

###

loader = D.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4)
vloader = D.DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=4)

dataLoaders = {'train': loader, 'val':vloader}
dataset_sizes = {'train':len(ds_train), 'val':len(ds_val)}
del loader, vloader

###

# NLLLoss
nllloss = nn.CrossEntropyLoss()
# CenterLoss
lmcl_loss = LMCL_loss(num_classes=classes, feat_dim=1024, s=32, m=0.2)
print(lmcl_loss.centers.shape)
#lmcl_loss.load_state_dict(torch.load('../input/model-512-14epochs/final_centers_tmp.bin'))

#use_cuda:
nllloss = nllloss.cuda()
lmcl_loss = lmcl_loss.cuda()

criterion = [nllloss, lmcl_loss]

#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

# optimzer4nn
optimizer4nn = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
sheduler_4nn = lr_scheduler.StepLR(optimizer4nn, 5, gamma=0.75)

# optimzer4center
optimzer4center = torch.optim.AdamW(lmcl_loss.parameters(), lr=0.001, weight_decay=0.01)
sheduler_4center = lr_scheduler.StepLR(optimizer4nn, 5, gamma=0.75)
optimizer = [optimizer4nn, optimzer4center]


# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=1.0)
scheduler = [sheduler_4nn, sheduler_4center]

###

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    USEBOTHSITES = 0
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                criterion[1].train()  # Set loss-model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                criterion[1].eval()   # Set loss-model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            batch = 0

            # Iterate over data.
            for inputs, labels in dataLoaders[phase]:
                #print(len(inputs))
                #print(inputs[0].shape)
                #print(inputs[1].shape)
                inputs[0] = inputs[0].to(device)
                inputs[1] = inputs[1].to(device)
                if USEBOTHSITES==1:
                    inputs[2] = inputs[2].to(device)
                labels = labels.to(device)

                if batch%1000==0:
                     print(f"Batch: {batch} {time.ctime()} {inputs[0].shape} number of samples:{dataset_sizes[phase]} lr:{optimizer[0].param_groups[0]['lr']}")
                batch += 1

                # zero the parameter gradients
                optimizer[0].zero_grad()
                optimizer[1].zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    features = model(inputs)
                    #outputs = model(inputs)
                    #print(inputs[1].shape)

                    logits, mlogits = criterion[1](features, labels)  # 1D cosFace
		    #logits, mlogits = criterion[1](features, labels, inputs[1])  # 2D cosFace
                    loss = criterion[0](mlogits, labels)

                    _, preds = torch.max(logits.data, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer[0].step()
                        optimizer[1].step()

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
                torch.save(model.state_dict(), f"best_model_cosFace_loss{groupCode}.bin")
                torch.save(lmcl_loss.state_dict(), f"best_centers_cosFace_loss{groupCode}.bin")
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), f"best_model_cosFace_acc{groupCode}.bin")
                torch.save(lmcl_loss.state_dict(), f"best_centers_cosFace_acc{groupCode}.bin")


            if phase == 'val':
                scheduler[0].step()
                scheduler[1].step()


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model


model = train_model(model, criterion, optimizer, scheduler, num_epochs=Nepochs)

torch.save(model.state_dict(), "final_model2.bin")
torch.save(lmcl_loss.state_dict(), "final_centers.bin")
