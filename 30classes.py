
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

path_data = '../input/recursion-cellular-image-classification'

device = 'cuda'

batch_size = 16   # was 64

import random

import PIL.ImageOps as ImageOps

import PIL.ImageChops as ImageChops


class FullTrainingDataset(torch.utils.data.Dataset):

    def __init__(self, full_ds, offset, length):

        self.full_ds = full_ds

        self.offset = offset

        self.length = length

        #assert len(full_ds)<offset+length, Exception(“Parent Dataset not long enough”)

        #super(FullTrainingDataset, self).init()


    def __len__(self):

        return self.length


    def __getitem__(self, i):

        return self.full_ds[i+self.offset]

def trainTestSplit(dataset, val_share=0.2):

    #print(f'size: {len(dataset)}')

    val_offset = int(len(dataset)*(1-val_share))

    return FullTrainingDataset(dataset, 0, val_offset), FullTrainingDataset(dataset, val_offset, len(dataset)-val_offset)


class ImagesDS(D.Dataset):

    def __init__(self, csv_file, img_dir, mode='train', useBothSites=True, channels=[1,2,3,4,5,6], useOnly=0):

        

        df = pd.read_csv(csv_file)

        self.records = df.to_records(index=False)

        self.channels = channels

        self.mode = mode

        self.img_dir = img_dir

        self.useBothSites = useBothSites

        Nexamples = df.shape[0]

        if useOnly>0:

            Nexamples = min(Nexamples,useOnly)

        if useBothSites:

            self.len = 2 * Nexamples  # site1 and site2

        else:

            self.len = Nexamples


    @staticmethod

    def _remove_artifacts(img):            

        x1 = img[1-1, :, :]

        x2 = img[2-1, :, :]

        x3 = img[3-1, :, :]

        x4 = img[4-1, :, :]

        x5 = img[5-1, :, :]

        x6 = img[6-1, :, :]

        tt = 0.3

        boolVector = (x1>tt) & (x2>tt) & (x3>tt) & (x4>tt) & (x5>tt) & (x6>tt)

        y1 = np.where(boolVector, 0.3, x1)

        y2 = np.where(boolVector, 0.3, x2)

        y3 = np.where(boolVector, 0.3, x3)

        y4 = np.where(boolVector, 0.3, x4)

        y5 = np.where(boolVector, 0.3, x5)

        y6 = np.where(boolVector, 0.3, x6)

        # and x3>0.1 and x4>0.1 and x5>0.1 and x6>0.1

        #img[1-1, :, :] = torch.as_tensor(y1)

        img[2-1, :, :] = torch.as_tensor(y2)

        img[3-1, :, :] = torch.as_tensor(y3)

        #img[4-1, :, :] = torch.as_tensor(y4)

        img[5-1, :, :] = torch.as_tensor(y5)

        img[6-1, :, :] = torch.as_tensor(y6)

        

        return img

            

    @staticmethod

    def _correct_overlaping_channels(img):

        img = ImagesDS._remove_artifacts(img)

        #return img

        tt=0.1

        # CORRECT OVERLAPPING CHANNELS

        # correct nucleolus:

        x0 = img[1-1, :, :] # nucleus

        x = img[4-1, :, :] # nucleolus

        x = np.where(x>x0, x0, x)

        img[4-1, :, :] = torch.as_tensor(x)


        # correct golgi (remove nucleus signal and actin signal)

        x1 = img[1-1, :, :] # nucleus

        x2 = img[3-1, :, :].numpy() # actin

        x = img[6-1, :, :] # golgi

        # remove nucleus from golgi

        x = np.where(x1>tt, 0, x)

        # remove actin from golgi

        x = np.where(x2>x, 0, x)

        img[6-1, :, :] = torch.as_tensor(x)

        # remove necleus signal from endoplasmatic reticulum

        x0 = img[1-1, :, :] # nucleus

        x = img[2-1, :, :] # ER

        x = np.where(x0>0.3, 0, x)

        img[2-1, :, :] = torch.as_tensor(x)

        #correct actin and mitochondria (remove nucleolus signal)

        x0 = img[4-1, :, :] # nucleolus

        x1 = img[3-1, :, :] # actin

        x2 = img[5-1, :, :] # mitochondria

        x3 = img[6-1, :, :].numpy()  # golgi

        # remove nucleolus signal

        x1 = np.where(x0>tt, 0, x1)

        # remove golgi from actin

        x1 = np.where(x3>x1, 0, x1)

        img[3-1, :, :] = torch.as_tensor(x1)


        # remove nucleolus from mitochondria

        x2 = np.where(x0>tt, 0, x2)

        img[5-1, :, :] = torch.as_tensor(x2)

        

        return img

    

    @staticmethod

    def _load_img_as_tensor(file_name, r1, r2, r3, noiseLevel=0):

        output_size = 224

        

        with Image.open(file_name) as img:

            #minV, maxV = img.getextrema()

            #print (minV, maxV, file_name)

            #normalize = T.Normalize(mean=[minV,],std=[(maxV-minV/255),])

                        

            angle = r1 * 90

            

            img = T.functional.rotate(img, angle)

            if r2==1:

                img = T.functional.hflip(img)

            if r3==1:

                img = T.functional.vflip(img)

            

            # remove noise (all intensity lower than 10)

            if noiseLevel>0:

                img = ImageChops.subtract(img, ImageChops.constant(img, noiseLevel))

            # Contrast stretching

            img = ImageOps.autocontrast(img, cutoff=1, ignore=None)

            #i = np.random.uniform(0,199)

            #j = np.random.uniform(0,199)

            #img = T.functional.crop(img, i, j, 312, 312)

            

            #img = T.functional.resize(img,output_size)

            

            #transform = T.Compose([T.ToTensor(), normalize])

            #transform = T.Compose([T.RandomVerticalFlip(), T.RandomHorizontalFlip(), T.ToTensor(), normalize])

            transform = T.ToTensor()

            

            return transform(img)

            

    def _get_img_path(self, index, channel, site=1):

        if self.useBothSites:

            my_index = index//2

            site = (index%2) + 1

        else:

            my_index = index

            #use sie from the input parameter

        

        experiment, well, plate = self.records[my_index].experiment, self.records[my_index].well, self.records[my_index].plate

        

        return '/'.join([self.img_dir,self.mode,experiment,f'Plate{plate}',f'{well}_s{site}_w{channel}.png'])

        

    def __getitem__(self, index):

        GETBOTHSITES=0

        

        paths = [self._get_img_path(index, ch) for ch in self.channels]

        if GETBOTHSITES==1:

            paths2 = [self._get_img_path(index, ch, site=2) for ch in self.channels]

        

        if self.useBothSites:

            dd = 2

        else:

            dd = 1

            

        experiment = self.records[index//dd].experiment

        cellLine = torch.FloatTensor([0,0,0,0])

        if 'HEPG2' in experiment:

        #if 'HUVEC-0' in experiment:

            cellLine[0] = 1

        if 'HUVEC' in experiment:

            cellLine[1] = 1

        if 'RPE' in experiment:

            cellLine[2] = 1

        if 'U2OS' in experiment:

            cellLine[3] = 1

        #cellLine = cellLine.long()

        r1 = random.randint(0,4)

        r2 = random.randint(0,2)

        r3 = random.randint(0,2)

        img = torch.cat([self._load_img_as_tensor(img_path,r1,r2,r3) for img_path in paths])

        

        img = self._correct_overlaping_channels(img)

        

        normalize = T.Normalize(mean=[0.485,0.485,0.485,0.485,0.485,0.485,],

                                     std=[0.229,0.229,0.229,0.229,0.229,0.229,])

        img = normalize(img)

        

        if GETBOTHSITES==1:

            img2 = torch.cat([self._load_img_as_tensor(img_path,r1,r2,r3) for img_path in paths2])

            if self.mode == 'train':

                return [img, img2, cellLine], self.records[index//dd].sirna

            else:

                return [img, img2, cellLine], self.records[index//dd].id_code

        else:

            if self.mode == 'train':

                return [img, cellLine], self.records[index//dd].sirna

            else:

                return [img, cellLine], 0 # self.records[index//dd].id_code


    def __len__(self):

        """

        Total number of samples in the dataset

        """

        return self.len


###ds = ImagesDS('../input/my-train-set/my_train.set', path_data, useBothSites=False, useOnly=5536)

###ds_train, ds_val = trainTestSplit(ds, val_share=0.1512762)


ds = ImagesDS('../input/my-train-set30/my_train_30classesHUVEC.set', path_data, useBothSites=True)

ds_train, ds_val = trainTestSplit(ds, val_share=0.125)


ds_test = ImagesDS(path_data+'/test.csv', path_data, mode='test')

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import PIL.ImageChops as ImageChops

import cv2 as cv

​

def showhistogram(im):

    a = np.array(im.getdata())

    a2 = [x for x in a if x > 0]


    fig, ax = plt.subplots(figsize=(10,4))

    n,bins,patches = ax.hist(a, bins=range(256), edgecolor='none')

    ax.set_title("histogram")

    ax.set_xlim(0,255)

    #ax.set_xlim(0,50)


    cm = plt.cm.get_cmap('cool')

    norm = matplotlib.colors.Normalize(vmin=bins.min(), vmax=bins.max())

    for b,p in zip(bins,patches):

        p.set_facecolor(cm(norm(b)))

    plt.show()


#print((ds_train.__getitem__(3)[0][0]).shape)

#print(ds_train.__getitem__(4)[0][1])

for N in range(1,7):

    im = Image.open(f"../input/recursion-cellular-image-classification/train/HUVEC-01/Plate1/D14_s1_w{N}.png")

    img = cv.imread(f"../input/recursion-cellular-image-classification/train/HUVEC-01/Plate1/D14_s1_w{N}.png",0)

    #im = ImageChops.subtract(im, ImageChops.constant(im, 10))

    #im.show()

    #fig, ax = plt.subplots(figsize=(20,8))

    ret,thresh1 = cv.threshold(img,30,255,cv.THRESH_BINARY)

    #plt.imshow(np.asarray(thresh1))

    #plt.imshow(np.asarray(im))

    #plt.imshow(im,cmap="gray")

    #print(np.asarray(im).shape)

    a = np.array(im.getdata())

    #print(im.getdata().shape)

    a2 = [x for x in a if x > 10]

    #im = Image.fromarray(a2, 'RGB')

    fig, ax = plt.subplots(figsize=(20,8))

    im = ImageChops.subtract(im, ImageChops.constant(im, 5))

    im = ImageOps.autocontrast(im, cutoff=1, ignore=None)

    plt.imshow(np.asarray(im),cmap="gray")

    #print(np.median(a),np.median(a2),np.min(a),np.max(a),np.std(a))

    #showhistogram(im)


from torch.autograd import Variable


class LMCL_loss(nn.Module):

    """

        Refer to paper:

        Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou,Zhifeng Li, and Wei Liu

        CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018

        re-implement by yirong mao

        2018 07/02

        """

    def __init__(self, num_classes, feat_dim, s=7.00, m=0.2, NcellLines = 4):

        super(LMCL_loss, self).__init__()

        self.feat_dim = feat_dim

        self.num_classes = num_classes

        self.s = s

        self.m = m

        self.cellLines = NcellLines

        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, NcellLines))


    def forward(self, feat, label, cellLine):

        batch_size = feat.shape[0]

        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)

        nfeat = torch.div(feat, norms)


        batchCenters = torch.matmul(self.centers, torch.transpose(cellLine,0,1))

        #print(batchCenters.shape)

        

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)

        ncenters = torch.div(self.centers, norms_c)

        #print(ncenters.shape)

        #print(feat.shape)

        #print(torch.transpose(batchCenters, 0, 2).shape)

        #logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        logits = torch.matmul(nfeat, torch.transpose(batchCenters, 0, 2)) # Nbatch * Nbatch * Nclasses

        #print(logits)

        l2 = torch.diagonal(logits,0,dim1=0,dim2=1)

        logits = l2.transpose(dim0=0,dim1=1)

        #print(logits.shape)

        

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)

        y_onehot.zero_()

        y_onehot = y_onehot.cuda()

        

        #print(label.shape)

        #print(torch.unsqueeze(label, dim=-1).shape)

        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)

        #print(y_onehot.shape)

        

        margin_logits = self.s * (logits - y_onehot)


        return logits, margin_logits

class DensNet(nn.Module):

    def __init__(self, num_classes=1000, num_channels=6, pretrained=True):

        super().__init__()

        preloaded = torchvision.models.densenet121(pretrained=pretrained)

        

        # Freeze model parameters

        #for param in preloaded.parameters():

        #    param.requires_grad = False

        

        w = preloaded.features.conv0.weight.clone()

        #print(w.shape)

        

        self.features = preloaded.features

        self.features.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3, bias=True)

        

        if (pretrained):  

            print(self.features.conv0.weight.shape)

            self.features.conv0.weight = nn.Parameter(torch.cat((w,w),dim=1))

            #self.features.conv0.weight = nn.Parameter(torch.cat((w,

            #                        0.5*(w[:,:1,:,:]+w[:,2:,:,:])),dim=1))

        

        #self.normalizeInputLayer = nn.BatchNorm2d(6)

        

        #self.classifier = nn.Bilinear(1024,4, num_classes, bias=True)

        del preloaded

        

    def forward(self, x):

        USEBOTHSITES=0

        

        if USEBOTHSITES==0:

            #normalizedx = self.normalizeInputLayer(x[0])

            #features = self.features(normalizedx)

            features = self.features(x[0])

            features1 = F.relu(features, inplace=True)

            features1 = F.adaptive_avg_pool2d(features1, (1, 1)).view(features.size(0), -1)

            #out = self.classifier(features1,x[1])

            return features1#, out

            

        else:

            features = self.features(x[0])

            out = F.relu(features, inplace=True)

            out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)

            out = self.classifier(out,x[2])

            

            features1 = self.features(x[1])

            out1 = F.relu(features1, inplace=True)

            out1 = F.adaptive_avg_pool2d(out1, (1, 1)).view(features1.size(0), -1)

            out1 = self.classifier(out1,x[2])

            

            #return torch.max(torch.cat((out.unsqueeze(1), out1.unsqueeze(1)), dim=1),1)[0]

            return torch.sum(torch.cat((out.unsqueeze(1), out1.unsqueeze(1)), dim=1),1)


classes = 30 #1108

model = DensNet(num_classes=classes)

#model.load_state_dict(torch.load('../input/recursion-cellular-image-classification/recursion_dataset_license.pdf'))

#model.load_state_dict(torch.load('../input/my-model2/final_model2.bin'))

#model.load_state_dict(torch.load('../input/my-model3b/final_model60.bin'))

#model.load_state_dict(torch.load('../input/model14/final_model14.bin'))

#model.load_state_dict(torch.load('../input/model-512-8epochs/final_model_512_8.bin'))


# set drop rate = 0.3

if 1==0:

    DROP_RATE = 0.3

    for idx1, m in enumerate(model.named_children()):

        if m[0]=='features':

            #print(idx1, '->', m[1])

            for idx2, n in enumerate(m[1].named_children()):

                if 'denseblock' in n[0]:

                    print(idx2, '->', n[0])

                    for idx3, o in enumerate(n[1].named_children()):

                        #print(idx3, '->', o[0])

                        o[1].drop_rate=DROP_RATE


model.to(device);


loader = D.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2)

vloader = D.DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=2)

tloader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)


dataLoaders = {'train': loader, 'val':vloader}

dataset_sizes = {'train':len(ds_train), 'val':len(ds_val)}

#print(len(ds_train))

#print(len(ds_val))

del loader, vloader


###criterion = nn.BCEWithLogitsLoss()

#criterion = nn.CrossEntropyLoss()


# NLLLoss

nllloss = nn.CrossEntropyLoss()

# CenterLoss

loss_weight = 0.1

lmcl_loss = LMCL_loss(num_classes=classes, feat_dim=1024, s=4, m=0.3)

if 1==1:     #use_cuda:

    nllloss = nllloss.cuda()

    lmcl_loss = lmcl_loss.cuda()

    #model = model.cuda()


criterion = [nllloss, lmcl_loss]


#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)


# optimzer4nn

optimizer4nn = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

sheduler_4nn = lr_scheduler.StepLR(optimizer4nn, 20, gamma=0.5)


# optimzer4center

optimzer4center = torch.optim.SGD(lmcl_loss.parameters(), lr=0.01)

sheduler_4center = lr_scheduler.StepLR(optimizer4nn, 20, gamma=0.5)

optimizer = [optimizer4nn, optimzer4center]

# Decay LR by a factor of 0.1 every 7 epochs

#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=1.0)

scheduler = [sheduler_4nn, sheduler_4center]


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()

    USEBOTHSITES = 0

    /

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0


    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)


        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:

            if phase == 'train':

                scheduler[0].step()

                scheduler[1].step()

                model.train()  # Set model to training mode

            else:

                model.eval()   # Set model to evaluate mode


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

                    print(f"Batch: {batch} {time.ctime()} {inputs[0].shape} number of samples:{dataset_sizes[phase]}")

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

                    

                    logits, mlogits = criterion[1](features, labels, inputs[1])

                    loss = criterion[0](mlogits, labels)


                    _, preds = torch.max(logits.data, 1)

                    #_, preds = torch.max(outputs, 1)

                    

                    #loss = criterion(outputs, labels)

                    ###target = torch.zeros_like(outputs, device=device)

                    ###target[np.arange(inputs[0].size(0)), labels] = 1

                    ###loss = criterion(outputs, target)


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

            if phase == 'val' and epoch_acc > best_acc:

                best_acc = epoch_acc

                best_model_wts = copy.deepcopy(model.state_dict())

        print()


    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights

    model.load_state_dict(best_model_wts)

    return model

model = train_model(model, criterion, optimizer, scheduler, num_epochs=100)

torch.save(model.state_dict(), "final_model2.bin")

@torch.no_grad()

def prediction(model, loader):

    #preds = np.empty(0)

    preds = np.empty((0,classes))

    xx=0

    

    for x, labels in loader: 

        x[0] = x[0].to(device) # image

        x[1] = x[1].to(device) # cellLine

        labels = labels.to(device)

        #output = model(x)

        

        features = model(x)

        #print(labels)

        logits, mlogits = criterion[1](features, labels, x[1])

        _, predicted = torch.max(logits.data, 1)

        

        #print(output.cpu())

        #idx = output.max(dim=-1)[1].cpu().numpy()

        #preds = np.append(preds, idx, axis=0)

        

        preds = np.append(preds,logits.cpu(), axis=0)

        xx += 1

        if xx%10==0:

            print(xx)

            #break

    return preds

#preds1a = prediction(model, tloader)

#preds1b = prediction(model, tloader)

#preds1c = prediction(model, tloader)

#preds1d = prediction(model, tloader)

import math

def sigmoid(x):

  return 1 / (1 + math.exp(-x))

vsigmoid = np.vectorize(sigmoid)



preds = np.empty(0)


if 1==0:

    for k in range(preds1a.shape[0]//2):

        xx = (vsigmoid(preds1a[2*k,]) + vsigmoid(preds1a[2*k+1,])) / 2

        #xx = (vsigmoid(preds1a[k,]) + vsigmoid(preds2a[k,]) + vsigmoid(preds1b[k,]) + vsigmoid(preds2b[k,]) + vsigmoid(preds1c[k,]) + vsigmoid(preds2c[k,]) + vsigmoid(preds1d[k,]) + vsigmoid(preds2d[k,])) / 8

        m=max(xx)

        X = [i for i, j in enumerate(xx) if j == m][0]

        preds = np.append(preds,X)

        #print(X, xx[X])

    #print(preds)

print(preds.shape)

submission = pd.read_csv(path_data + '/test.csv')

#submission = submission[:400]

print(submission.shape)

print(preds.shape)


submission['sirna'] = preds.astype(int)

submission.to_csv('submission.csv', index=False, columns=['id_code','sirna'])


#print((preds1a[0,1:10]))

#print((preds2a[1,1:10]))

#preds1a.to_csv('perds.csv', index=False)

#np.savetxt("preds.csv", preds1a, delimiter=",")

Download submission File
Download model
Download predictions

