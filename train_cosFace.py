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

####

path_data = '../input'
device = 'cuda'
batch_size = 16

###

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

###

ds = ImagesDS(path_data + '/my_train.set', path_data, useBothSites=True, useOnly=5536)
ds_train, ds_val = trainTestSplit(ds, val_share=0.1512762)

ds_test = ImagesDS(path_data + '/test.csv', path_data, mode='test')

###

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

###

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

        self.classifier = nn.Bilinear(1024,4, num_classes, bias=True)
        del preloaded

    def forward(self, x):
        USEBOTHSITES=0

        if USEBOTHSITES==0:
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

###

classes = 1108 #1108
model = DensNet(num_classes=classes)
#model.load_state_dict(torch.load('./final_model_512_20.bin'))
#model.eval();  # important to set dropout and normalization layers
model.to(device);

###

loader = D.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2)
vloader = D.DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=2)
tloader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)

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
print(model.classifier.weight.shape)

#use_cuda:
nllloss = nllloss.cuda()
lmcl_loss = lmcl_loss.cuda()

criterion = [nllloss, lmcl_loss]

#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

# optimzer4nn
optimizer4nn = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
sheduler_4nn = lr_scheduler.StepLR(optimizer4nn, 20, gamma=0.5)

# optimzer4center
optimzer4center = torch.optim.SGD(lmcl_loss.parameters(), lr=0.001)
sheduler_4center = lr_scheduler.StepLR(optimizer4nn, 20, gamma=0.5)
optimizer = [optimizer4nn, optimzer4center]


# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=1.0)
scheduler = [sheduler_4nn, sheduler_4center]

###

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


model = train_model(model, criterion, optimizer, scheduler, num_epochs=7)

torch.save(model.state_dict(), "final_model2.bin")
torch.save(lmcl_loss.state_dict(), "final_centers.bin")
