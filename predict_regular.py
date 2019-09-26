import numpy as np 
import pandas as pd
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
testFile = sys.argv[3]
sirnaFile = sys.argv[4]
outputFile = sys.argv[5]
layers = int(sys.argv[6])
Navg = int(sys.argv[7])

path_data = '../../input'
device = 'cuda'
batch_size = 52   # was 16

ds = ImagesDS(testFile, path_data, useBothSites=True, mode='test')#, useOnly=500)

classes = 1108 # 30 - HUVEC30 # controls 31 # 61 - HUVEC+CONTROLS # 1108

#model = DensNet_controls(num_classes=classes, Ncontrols = 31, pretrained=True)
print(f"Layers: {layers}")
model = DensNet(num_classes=classes, pretrained=True, layers = layers)
model.load_state_dict(torch.load(f'{modelFile}'))

model.to(device)

loader = D.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

mySirnas = np.genfromtxt(sirnaFile, delimiter=',', dtype=int).reshape((277,1))
mask = np.zeros((1,1108))
mask[0,mySirnas] = 1

@torch.no_grad()
def prediction(model, loader):
    #preds = np.empty(0)
    print(len(ds))
    FinalPreds = np.zeros((len(ds)//2,classes))
   
    for it in range(Navg):
      preds = np.empty((0,classes))
      xx = 0
      for x, _ in loader: 
        for i in range(2):
           x[i] = x[i].to(device) # img1, img2, controls1, controls2, celline
	
        output = model(x)
        preds = np.append(preds,output.cpu(), axis=0)
        
        xx += 1
        if xx%50==0:
            print(f'{xx} time:{time.ctime()}')
            #break

      # AVERAGE over sites
      if (1==1):
        predsM = np.empty((0,classes))
        for k in range(preds.shape[0]//2):
            xx = (preds[2*k,:] + preds[2*k+1,:]) / 2
            xx.reshape(1,1108)
            #print(predsM.shape)
            #print(xx.shape)
            predsM = np.append(predsM,xx.reshape(1,1108), axis=0)
      FinalPreds = FinalPreds + predsM

    return FinalPreds

model.eval()
preds1a = prediction(model, loader) / Navg
#print(preds1a.shape)
#print(preds1a[0,:])

testData = pd.read_csv(testFile)
predsAll = np.zeros(testData.shape[0])
experiments = list(set(testData['experiment']))
for exp in experiments: #['HEPG2-08']:
   #print(exp)
   A = preds1a[np.where(testData['experiment']==exp)[0],:]
   
   maxV = np.ndarray.max(A,axis=1,keepdims=True)
   A1 = A - maxV
   A1 = np.exp(A1)
   A1 = A1 * mask   # filter out predictions outside of this group

   A1 = A1/(A1.sum(axis=0) + 1e-100)

   maxV = np.ndarray.max(A,axis=0,keepdims=True)
   A2 = A - maxV
   A2 = np.exp(A2)
   A2 = A2 * mask
   A2 = np.transpose(np.transpose(A2)/(A2.sum(axis=1) + 1e-100 ))

   A = np.log(A1) + np.log(A2)


   preds =np.zeros(A.shape[0])
   # greedily assign the most confident prediction
   for i in range(A.shape[0]):
        #x= np.max(A)
        #print(x)
        best = np.where(A==np.max(A))
        row = best[0][0]
        column = best[1][0]
        preds[row] = column
        A[row,:] = -1e100
        A[:,column] = -1e100

   predsAll[np.where(testData['experiment']==exp)] = preds
  
testData['sirna'] = predsAll.astype(int)
testData.to_csv(outputFile, index=False, columns=['id_code','sirna'])
