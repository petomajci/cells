import numpy as np
import pandas as pd

import torch
import torch.utils.data as D

import random

from PIL import Image
import PIL.ImageChops as ImageChops
import PIL.ImageOps as ImageOps


import torchvision
from torchvision import transforms as T

class ImagesDS(D.Dataset):
    def __init__(self, csv_file, img_dir, mode='train', useBothSites=True, channels=[1, 2, 3, 4, 5, 6], useOnly=0):

        df = pd.read_csv(csv_file)
        self.records = df.to_records(index=False)
        self.channels = channels
        self.mode = mode
        self.img_dir = img_dir
        self.useBothSites = useBothSites
        Nexamples = df.shape[0]
        if useOnly > 0:
            Nexamples = min(Nexamples, useOnly)
        if useBothSites:
            self.len = 2 * Nexamples  # site1 and site2
        else:
            self.len = Nexamples

    @staticmethod
    def _remove_artifacts(img):
        x1 = img[1 - 1, :, :]
        x2 = img[2 - 1, :, :]
        x3 = img[3 - 1, :, :]
        x4 = img[4 - 1, :, :]
        x5 = img[5 - 1, :, :]
        x6 = img[6 - 1, :, :]
        tt = 0.3
        boolVector = (x1 > tt) & (x2 > tt) & (x3 > tt) & (x4 > tt) & (x5 > tt) & (x6 > tt)
        y1 = np.where(boolVector, 0.3, x1)
        y2 = np.where(boolVector, 0.3, x2)
        y3 = np.where(boolVector, 0.3, x3)
        y4 = np.where(boolVector, 0.3, x4)
        y5 = np.where(boolVector, 0.3, x5)
        y6 = np.where(boolVector, 0.3, x6)
        # and x3>0.1 and x4>0.1 and x5>0.1 and x6>0.1
        # img[1-1, :, :] = torch.as_tensor(y1)
        img[2 - 1, :, :] = torch.as_tensor(y2)
        img[3 - 1, :, :] = torch.as_tensor(y3)
        # img[4-1, :, :] = torch.as_tensor(y4)
        img[5 - 1, :, :] = torch.as_tensor(y5)
        img[6 - 1, :, :] = torch.as_tensor(y6)

        return img

    def _add_noise(img, mean=0, std=0.1):

      noise = img.new_tensor(img.data).normal_(mean=mean,std = std)
      #print(img)
      #print(img.shape)
      #print(noise.shape)
      out = torch.add(img, noise)

      return out

    @staticmethod
    def _correct_overlaping_channels(img):
        img = ImagesDS._remove_artifacts(img)
        # return img
        tt = 0.1
        # CORRECT OVERLAPPING CHANNELS
        # correct nucleolus:
        x0 = img[1 - 1, :, :]  # nucleus
        x = img[4 - 1, :, :]  # nucleolus
        x = np.where(x > x0, x0, x)
        img[4 - 1, :, :] = torch.as_tensor(x)

        # correct golgi (remove nucleus signal and actin signal)
        x1 = img[1 - 1, :, :]  # nucleus
        x2 = img[3 - 1, :, :].numpy()  # actin
        x = img[6 - 1, :, :]  # golgi
        # remove nucleus from golgi
        x = np.where(x1 > tt, 0, x)
        # remove actin from golgi
        x = np.where(x2 > x, 0, x)
        img[6 - 1, :, :] = torch.as_tensor(x)

        # remove necleus signal from endoplasmatic reticulum
        x0 = img[1 - 1, :, :]  # nucleus
        x = img[2 - 1, :, :]  # ER
        x = np.where(x0 > 0.3, 0, x)
        img[2 - 1, :, :] = torch.as_tensor(x)

        # correct actin and mitochondria (remove nucleolus signal)
        x0 = img[4 - 1, :, :]  # nucleolus
        x1 = img[3 - 1, :, :]  # actin
        x2 = img[5 - 1, :, :]  # mitochondria
        x3 = img[6 - 1, :, :].numpy()  # golgi

        # remove nucleolus signal
        x1 = np.where(x0 > tt, 0, x1)
        # remove golgi from actin
        x1 = np.where(x3 > x1, 0, x1)
        img[3 - 1, :, :] = torch.as_tensor(x1)

        # remove nucleolus from mitochondria
        x2 = np.where(x0 > tt, 0, x2)
        img[5 - 1, :, :] = torch.as_tensor(x2)

        return img

    @staticmethod
    def _load_img_as_tensor(file_name, r1, r2, r3, noiseLevel=0):
        output_size = 224

        with Image.open(file_name) as img:
            # minV, maxV = img.getextrema()
            # print (minV, maxV, file_name)
            # normalize = T.Normalize(mean=[minV,],std=[(maxV-minV/255),])

            angle = r1 * 90

            img = T.functional.rotate(img, angle)
            if r2 == 1:
                img = T.functional.hflip(img)
            if r3 == 1:
                img = T.functional.vflip(img)

            # remove noise (all intensity lower than 10)
            if noiseLevel > 0:
                img = ImageChops.subtract(img, ImageChops.constant(img, noiseLevel))
            # Contrast stretching
            img = ImageOps.autocontrast(img, cutoff=1, ignore=None)
            # i = np.random.uniform(0,199)
            # j = np.random.uniform(0,199)
            # img = T.functional.crop(img, i, j, 312, 312)

            # img = T.functional.resize(img,output_size)

            # transform = T.Compose([T.ToTensor(), normalize])
            transform = T.ToTensor()

            return transform(img)

    def _get_img_path(self, index, channel, site=1):
        if self.useBothSites:
            my_index = index // 2
            site = (index % 2) + 1
        else:
            my_index = index
            # use site from the input parameter

        experiment, well, plate = self.records[my_index].experiment, self.records[my_index].well, self.records[
            my_index].plate

        mode = self.mode
        if self.mode == 'val':
            mode = 'train'
        #return '/'.join([self.img_dir, mode, experiment, f'Plate{plate}', f'{well}_s{site}_w{channel}.png'])
        return '/'.join([self.img_dir, 'train', experiment, f'Plate{plate}', f'{well}_s{site}_w{channel}.png'])  # all files were moved to train

    def __getitem__(self, index):
        GETBOTHSITES = 0

        paths = [self._get_img_path(index, ch) for ch in self.channels]
        if GETBOTHSITES == 1:
            paths2 = [self._get_img_path(index, ch, site=2) for ch in self.channels]

        if self.useBothSites:
            dd = 2
        else:
            dd = 1

        experiment = self.records[index // dd].experiment
        cellLine = torch.FloatTensor([0, 0, 0, 0])
        if 'HEPG2' in experiment:
            cellLine[0] = 1
        if 'HUVEC' in experiment:
            cellLine[1] = 1
        if 'RPE' in experiment:
            cellLine[2] = 1
        if 'U2OS' in experiment:
            cellLine[3] = 1
        

        normalize = T.Normalize(mean=[0.485, 0.485, 0.485, 0.485, 0.485, 0.485, ],
                                std=[0.229, 0.229, 0.229, 0.229, 0.229, 0.229, ])

        r1 = random.randint(0, 3)
        r2 = random.randint(0, 1)
        r3 = random.randint(0, 1)
        img = torch.cat([self._load_img_as_tensor(img_path, r1, r2, r3) for img_path in paths])
        #if self.mode=='train':
        #img = ImagesDS._add_noise(img)
           #img2 = ImagesDS._add_noise(img)
           #print(np.corrcoef(img1[1,:,:].numpy().reshape((262144,)),img2[1,:,:].numpy().reshape((1,262144))))	   
        img = ImagesDS._correct_overlaping_channels(img)
        img = normalize(img)

        if GETBOTHSITES == 1:
            r1 = random.randint(0, 3)
            r2 = random.randint(0, 1)
            r3 = random.randint(0, 1)
            img2 = torch.cat([self._load_img_as_tensor(img_path, r1, r2, r3) for img_path in paths2])
            #if self.mode=='train':
            #img2 = ImagesDS._add_noise(img2)
            img2 = ImagesDS._correct_overlaping_channels(img2)
            img2 = normalize(img2)

            if self.mode == 'train':
                return [img, img2, cellLine], self.records[index // dd].sirna
            elif self.mode =='val':
                return [img, img2, cellLine], self.records[index // dd].sirna
            else:
                return [img, img2, cellLine], 0
        else:
            if self.mode == 'train':
                return [img, cellLine], self.records[index // dd].sirna
            else:
                return [img, cellLine], 0  # self.records[index//dd].id_code

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
