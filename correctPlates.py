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

# my imports
from torch.optim import lr_scheduler
import copy

import tqdm

import warnings
warnings.filterwarnings('ignore')


train_csv = pd.read_csv("../input/train.csv")
test_csv = pd.read_csv("../input/test.csv")
sub = pd.read_csv("submission_20epochs.csv")
submission = pd.read_csv("submission_20epochs.csv")

plate_groups = np.zeros((1108,4), int)
for sirna in range(1108):
    grp = train_csv.loc[train_csv.sirna==sirna,:].plate.value_counts().index.values
    #print(grp)
    assert len(grp) == 3
    plate_groups[sirna,0:3] = grp
    plate_groups[sirna,3] = 10 - grp.sum()
    
plate_groups[:10,:]


# TEST DATA:

all_test_exp = test_csv.experiment.unique()

group_plate_probs = np.zeros((len(all_test_exp),4))
for idx in range(len(all_test_exp)):
    preds = sub.loc[test_csv.experiment == all_test_exp[idx],'sirna'].values
    pp_mult = np.zeros((len(preds),1108))
    pp_mult[range(len(preds)),preds] = 1
    
    sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
    assert len(pp_mult) == len(sub_test)
    
    for j in range(4):
        mask = np.repeat(plate_groups[np.newaxis, :, j], len(pp_mult), axis=0) == \
               np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
        
        group_plate_probs[idx,j] = np.array(pp_mult)[mask].sum()/len(pp_mult)
        

#PRINT assignement probabilities
pd.DataFrame(group_plate_probs, index = all_test_exp)

# assign group
exp_to_group = group_plate_probs.argmax(1)
print(exp_to_group)

preds1a = pd.read_csv("preds.csv.zip",header= None)
#print(preds1a.shape)
#print(preds1a.loc[0,0:5])


import math
classes = 1108

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
vsigmoid = np.vectorize(sigmoid)


preds = np.empty(0)
predProbabs = np.empty((preds1a.shape[0]//2,classes))

for k in range(preds1a.shape[0]//2):
    xx = (vsigmoid(preds1a.loc[2*k,]) + vsigmoid(preds1a.loc[2*k+1,])) / 2
    predProbabs[k,] = np.transpose(xx)
    #xx = (vsigmoid(preds1a[k,]) + vsigmoid(preds2a[k,]) + vsigmoid(preds1b[k,]) + vsigmoid(preds2b[k,]) + vsigmoid(preds1c[k,]) + vsigmoid(preds2c[k,]) + vsigmoid(preds1d[k,]) + vsigmoid(preds2d[k,])) / 8
    m=max(xx)
    X = [i for i, j in enumerate(xx) if j == m][0]
    preds = np.append(preds,X)
    #print(X, xx[X])
#print(preds)

print(preds.shape)
print(predProbabs.shape)


#function to set 75% sirna probabilities to 0...
def select_plate_group(pp_mult, idx):
    sub_test = test_csv.loc[test_csv.experiment == all_test_exp[idx],:]
    assert len(pp_mult) == len(sub_test)
    mask = np.repeat(plate_groups[np.newaxis, :, exp_to_group[idx]], len(pp_mult), axis=0) != \
           np.repeat(sub_test.plate.values[:, np.newaxis], 1108, axis=1)
    pp_mult[mask] = 0
    return pp_mult


#correct predictions according to the group:
for idx in range(len(all_test_exp)):
    #print('Experiment', idx)
    indices = (test_csv.experiment == all_test_exp[idx])

    preds = predProbabs[indices,:].copy()

    preds = select_plate_group(preds, idx)
    sub.loc[indices,'sirna'] = preds.argmax(1)

# check how much is the new and old submission identical... was only 35% in the original work
print((sub.sirna == submission['sirna']).mean())

sub.to_csv('submission_20_corrected.csv', index=False, columns=['id_code','sirna'])
