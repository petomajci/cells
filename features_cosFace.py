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
from cosface2D import LMCL_loss2D
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
centersFile = sys.argv[3]
trainFile = sys.argv[4]
mode = sys.argv[5]

####

path_data = '../../input'
device = 'cuda'
batch_size = 22

###

ds = ImagesDS(trainFile, path_data, mode=mode, useBothSites=True)#, useOnly=5536)

###

classes = 31 #1108
model = DensNet(num_classes=classes, pretrained=False)
if modelFile != 'none':
    model.load_state_dict(torch.load(modelFile))
    model.eval();  # important to set dropout and normalization layers

model.to(device);

###

# CenterLoss
lmcl_loss = LMCL_loss2D(num_classes=classes, feat_dim=1024, s=32, m=0.2)
print(lmcl_loss.centers.shape)
if centersFile != 'none':
     lmcl_loss.load_state_dict(torch.load(centersFile))

#use_cuda:
lmcl_loss = lmcl_loss.cuda()


###

loader = D.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

@torch.no_grad()
def prediction(model, loader):
    #preds = np.empty(0)
    preds = np.empty((0,classes))
    featuresA = np.empty((0,1024))
    xx=0
    
    for x, labels in loader: 
        x[0] = x[0].to(device) # image
        x[1] = x[1].to(device) # cellLine
        labels = labels * 0
        labels = labels.to(device)
        #output = model(x)
        
        features = model(x)
        #print(x[0][0,0,0,1:10])
        #print(features[0,:])
        #print(labels)
        logits, _ = lmcl_loss(features, labels, x[1])
        _, predicted = torch.max(logits.data, 1)
        
        #print(output.cpu())
        #idx = output.max(dim=-1)[1].cpu().numpy()
        #preds = np.append(preds, idx, axis=0)
        
        preds = np.append(preds,logits.cpu(), axis=0)
        featuresA = np.append(featuresA,features.cpu(), axis=0)
        xx += 1
        if xx%50==0:
            print(xx)
            #break
    return preds, featuresA

model.eval()
preds1a, features1 = prediction(model, loader)
#preds1a.to_csv('perds.csv', index=False)
np.savetxt(f"preds_{groupCode}.csv", preds1a, delimiter=",")
np.savetxt(f"features_{groupCode}.csv", features1, delimiter=",",fmt='%.3f')
#features1.tofile('features.csv',sep=',',format='%.3f')
