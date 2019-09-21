import os
import numpy as np
from skimage.io import imread
import pandas as pd
import PIL.ImageOps as ImageOps
import PIL.ImageChops as ImageChops
from PIL import Image
from PIL import ImageFilter

import time
import cv2
import tensorflow as tf

DEFAULT_BASE_PATH = '../input'
DEFAULT_CHANNELS = (1, 2, 3, 4, 5, 6)

def load_image(image_path, blur = False):
    #with tf.io.gfile.GFile(image_path, 'rb') as f:
    #    return imread(f, format='png')
    #print(image_path)
    with tf.io.gfile.GFile(image_path, 'rb') as f:
        im = Image.open(f)
        if blur:
            #print('warning: Loading with blur')
            im = im.filter(ImageFilter.BLUR)
        #im = ImageChops.subtract(im, ImageChops.constant(im, 10))
        #return ImageOps.autocontrast(im, cutoff=1, ignore=None)
        return im #ImageOps.equalize(im)


def load_images_as_tensor(image_paths, dtype=np.uint8, blur=False):
    n_channels = len(image_paths)

    data = np.ndarray(shape=(512, 512, n_channels), dtype=dtype)

    for ix, img_path in enumerate(image_paths):
        data[:, :, ix] = load_image(img_path, blur=blur)
        
    return data

def image_path(dataset,
               experiment,
               plate,
               address,
               site,
               channel,
               base_path=DEFAULT_BASE_PATH):
    
    return os.path.join(base_path, dataset, experiment, "Plate{}".format(plate),
                        "{}_s{}_w{}.png".format(address, site, channel))


def load_site(dataset,
              experiment,
              plate,
              well,
              site,
              channels=DEFAULT_CHANNELS,
              base_path=DEFAULT_BASE_PATH,
              blur = False):
    
    channel_paths = [
        image_path(
            dataset, experiment, plate, well, site, c, base_path=base_path)
        for c in channels
    ]
    #print(channel_paths[0])
    if os.path.isfile(channel_paths[0]):
        return load_images_as_tensor(channel_paths, blur=blur)
    else:
        return None

RGB_MAP = {
        1: {
            'rgb': np.array([255, 0, 0]),
            'range': [0, 255]#155]
        },
        2: {
            'rgb': np.array([0, 255, 0]),
            'range': [0, 255]#107]
        },
        3: {
            'rgb': np.array([0, 0, 255]),
            'range': [0, 255]#64]
        },
        4: {
            'rgb': np.array([0, 0, 255]),
            'range': [0, 255]#191]
        },
        5: {
            'rgb': np.array([0, 0, 255]),
            'range': [0, 255]#89]
        },
        6: {
            'rgb': np.array([0, 255, 0]),
            'range': [0, 255]#191]
        }
}


def convert_tensor_to_rgb(t, channels=range(1,7), vmax=255, rgb_map=RGB_MAP):
    """
    Converts and returns the image data as RGB image

    Parameters
    ----------
    t : np.ndarray
        original image data
    channels : list of int
        channels to include
    vmax : int
        the max value used for scaling
    rgb_map : dict
        the color mapping for each channel
        See rxrx.io.RGB_MAP to see what the defaults are.

    Returns
    -------
    np.ndarray the image data of the site as RGB channels
    """

    #noise = np.random.normal(0, 0.1*255, t.shape)
    #t = t + noise


    colored_channels = []
    for i, channel in enumerate(channels):
        #print(channel)
        mm = np.ndarray.max(t[:, :, channel-1])

        #if mm>0:
        #    rgb_map[channel]['range'][1] = mm
        x = (t[:, :, channel-1] / vmax) / \
            ((rgb_map[channel]['range'][1] - rgb_map[channel]['range'][0]) / 255) + \
            rgb_map[channel]['range'][0] / 255
        x = np.where(x > 1, 1., x)
        if (channel==4 and False): # nucleolus (only valid within nucleus)
            x0 = (t[:, :, 1-1] / vmax) # nucleus
            x = np.where(x>x0, x0, x)
        if (channel==13 or channel==15): #actin or mitochondria (remove nucleolus signal)
            # nucleolus
            x0 = (t[:, :, 4-1] / vmax)
            # nucleus
            x1 = (t[:, :, 1-1] / vmax)
            # filter nucleolus only in nucleus
            x0 = np.where(x0>x1, x1, x0)
            # remove nucleolus from actin or mitochondria
            x = np.where(x0>0.1, 0, x)
        if (channel==13): #actin (remove golgi signal)
            # golgi
            x0 = (t[:, :, 6-1] / vmax)
            # remove golgi from actin
            x = np.where(x0>x, 0, x)

        if (channel==16): #golgi (remove nucleus signal and actin signal)
            # nucleus
            x1 = (t[:, :, 1-1] / vmax)
            # actin
            x2 = (t[:, :, 3-1] / vmax)
            # remove nucleus from golgi
            x = np.where(x1>0.1, 0, x)
            # remove actin from golgi
            x = np.where(x2>x, 0, x)
        #x = np.where(x < 10/255, 0, x-10/255)
        x_rgb = np.array(
            np.outer(x, rgb_map[channel]['rgb']).reshape(512, 1024, 3),
            dtype=int)
        colored_channels.append(x_rgb)
    im = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
    im = np.where(im > 255, 255, im)
    return im

# MAIN

channels = (1,2,4)
#channels = (4,5,6)
cell='U2OS'

import sys

inputFile = sys.argv[1]
location = sys.argv[2]
destination = sys.argv[3]

df = pd.read_csv(inputFile)
records = df.to_records(index=False)

from scipy import ndimage
import matplotlib.pyplot as plt


def getNumberOfCells(x, threshold, minSize, maxSize):
    x[np.where((threshold<=x) )]=255
    x[np.where(x<255)]=0
    blobs, number_of_blobs = ndimage.label(x)

    ss = 0
    for l in range(number_of_blobs):
        s = np.sum(blobs==(l+1))
        if s>minSize and s<maxSize:     # channel 0
            w = np.where(blobs==(l+1))
            xx = w[0]
            yy = w[1]
            if (max(xx)-min(xx))*(max(yy)-min(yy)) < s*2.5:
                ss+=1

    return ss


def MoveAllChannelsToZero(inputTensor):
    x = inputTensor[:,:,0]
    x = np.where(x<=7, 0, x-7)
    inputTensor[:,:,0] = x

    
    # move min value of each channel to 0
    for ch in range(6):
        inputTensor[:,:,ch] -= np.min(inputTensor[:,:,ch])
        while np.sum(t1[:,:,ch]==0)<2000:   # there shall be at least 2000 empty pixels in each channel
            inputTensor[:,:,ch] = np.where(inputTensor[:,:,ch]==0,0,inputTensor[:,:,ch]-1)
    return inputTensor

########################################
#   MAIN program
########################################


for i in range(df.shape[0]):

    correctionF = np.zeros((2,6))

    for site in (1,2):
        k = i#22
        t1 = load_site(location, records[k].experiment, records[k].plate, records[k].well, site, blur=True)
      
        # move min value of each channel to 0
        t1 = MoveAllChannelsToZero(t1)

        max_ss = 0
        max_thr = 0
        for thr in range(1,70): 
            x = np.copy(t1[:,:,1])
            x[np.where((thr<=x) )]=255
            x[np.where(x<255)]=0
            blobs, number_of_blobs = ndimage.label(x)

            ss = 0
            for l in range(number_of_blobs):
                s = np.sum(blobs==(l+1))
                if s>400 and s<4000:    # channel 1
                #if s>200 and s<2000:     # channel 0
                #if s>.200 and s<2000000:  # channel 3
                    w = np.where(blobs==(l+1))
                    xx = w[0]
                    yy = w[1]
                    if (max(xx)-min(xx))*(max(yy)-min(yy)) < s*2.5:
                        ss+=1
                    else:
                        blobs[np.where(blobs==(l+1))]=0
                else:
                    blobs[np.where(blobs==(l+1))]=0
            #print(thr, number_of_blobs, ss)

            #blobs[np.where(blobs>1)]=1
            #tmp = np.copy(t1)
            #tmp[:,:,channel] = x * blobs
            #t = np.concatenate((t2,tmp),axis=1)
            #x = convert_tensor_to_rgb(t,channels=channels, rgb_map = RGB_MAP)
            #cv2.imwrite(f'{destination}/{records[i].sirna}-{records[i].id_code}-{records[i].plate}_{thr}.jpg',x)
            if ss>=max_ss:
                max_ss = ss
                max_thr = thr

            if ss<max_ss-10:  # to speed things up
                break

        max_ss2 = 0
        max_thr2 = 0
        for thr in range(1,185):
            x = np.copy(t1[:,:,0])
            x[np.where((thr<=x) )]=255
            x[np.where(x<255)]=0
            blobs, number_of_blobs = ndimage.label(x)

            ss = 0
            for l in range(number_of_blobs):
                s = np.sum(blobs==(l+1))
                if s>100 and s<2000:     # channel 0
                    w = np.where(blobs==(l+1))
                    xx = w[0]
                    yy = w[1]
                    if (max(xx)-min(xx))*(max(yy)-min(yy)) < s*2.5:
                        ss+=1
                    else:
                        blobs[np.where(blobs==(l+1))]=0
                else:
                    blobs[np.where(blobs==(l+1))]=0
            #print(thr, number_of_blobs, ss)

            #blobs[np.where(blobs>1)]=1
            #tmp = np.copy(t1)
            #tmp[:,:,0] = x * blobs
            #t = np.concatenate((t2,tmp),axis=1)
            #x = convert_tensor_to_rgb(t,channels=channels, rgb_map = RGB_MAP)
            #cv2.imwrite(f'{destination}/{records[i].sirna}-{records[i].id_code}-{records[i].plate}_{thr}.jpg',x)
            
            
            if ss>=max_ss2:
                max_ss2 = ss
                max_thr2 = thr

            if ss<max_ss2-15:  # to speed things up
                break


        x = np.copy(t1[:,:,2])  # aktin... take median norm
        median = np.mean(x[np.where(x>0)])
        #mm = np.median(x[np.where(x>0)])
        print(f"{records[i].id_code}-{records[i].plate}.1  {max_thr}  {max_thr2}  {median}  {max_ss}  {max_ss2}  {time.ctime()}")
        #print(f'{np.sum(x<1)} {np.sum(x==1)}  {np.sum(x>1)} {np.sum(x==0)}')
        #import matplotlib.pyplot as plt
        #plt.imshow(blobs)
        #plt.show()
        
        #blobs[np.where(blobs>1)]=1
        #t1[:,:,channel] = x * blobs

        # reload without blur 
        t1 = load_site(location, records[k].experiment, records[k].plate, records[k].well, site, blur=False)

        # move to zero
        t1 = MoveAllChannelsToZero(t1)
        t2 = np.copy(t1)

        t1 = t1.astype(np.float32)
        t2 = t2.astype(np.float32)
       
        C = 20 # 60

        correctionF[site-1,0] = max_thr2+1
        correctionF[site-1,1] = max_thr+1
        correctionF[site-1,2] = median
        correctionF[site-1,3] = max_thr+1

        #t1[:,:,3] = np.where(t1[:,:,0]<max_thr2,0,t1[:,:,3]) # filter only for nucleus


        t2[:,:,0] *= 20/8.5
        t1[:,:,0] *= (C/(max_thr2+1))

        t2[:,:,1] *= 20/17
        t1[:,:,1] *= (C/(max_thr+1))
        
        t2[:,:,2] *= 20/7.47
        t1[:,:,2] *= (C/median)
        
        t2[:,:,3] *= 20/17
        t1[:,:,3] *= (C/(max_thr+1))

        for j in (4,5):
          x = t1[:,:,j]
          mean2 = np.mean(x[np.where(x>0)])
          correctionF[site-1,j] = mean2
          t2[:,:,j] *= C/20
          t1[:,:,j] *= C/mean2
        #print(f'MEANS:  {np.mean(t1[:,:,0])}  {np.mean(t1[:,:,1])}  {np.mean(t1[:,:,2])}   {np.mean(t1[:,:,3])}  {np.mean(t1[:,:,4])}  {np.mean(t1[:,:,5])}')
        
        if 1==1:
            t = np.concatenate((t2,t1),axis=1)
            x = convert_tensor_to_rgb(t,channels=channels, rgb_map = RGB_MAP)
            cv2.imwrite(f'{destination}/{records[i].sirna}-{records[i].id_code}-{records[i].plate}_{max_thr}_{max_thr2}.jpg',x)

        import imageio
        directory = f'{destination}/{records[i].experiment}/Plate{records[i].plate}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        data8 = np.uint8(t1[:,:,0])
        imageio.imwrite(f'{directory}/{records[i].well}_s{site}_w1.png', data8)
    

    #print(correctionF)
    myC = np.mean(correctionF,axis=0)

    if 1==1:
      for row in ('02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23'):
        for column in ('B','C','D','E','F','G','H','I','J','K','L','M','N','O'):
            for site in (1,2):
                t1 = load_site(location, records[k].experiment, records[k].plate, f'{column}{row}', site, blur=False)
                if not t1 is None:
                    t1 = MoveAllChannelsToZero(t1)
                    t1 = t1.astype(np.float32)
                    #print(getNumberOfCells(np.copy(t1[:,:,0]),myC[0],200,2000))
                    #print(getNumberOfCells(np.copy(t1[:,:,1]),myC[1],400,4000))
                    
                    #t1[:,:,3] = np.where(t1[:,:,0]<myC[0],0,t1[:,:,3]) # filter only for nucleus

                    for j in range(6):
                        t1[:,:,j] *= C/myC[j]
                        data8 = np.uint8(t1[:,:,j])
                        imageio.imwrite(f'{directory}/{column}{row}_s{site}_w{j+1}.png', data8)
                    print(f'converted: {location}, {records[k].experiment}, {records[k].plate}, {column}{row} {site}')
    #exit()


exit()




for experiment in range(4,6):
    for plate in range(1,5):
        for site in (1,2):
            expName=f'{cell}-{experiment}'
            if experiment<10:
                expName=f'{cell}-0{experiment}'
            t1 = load_site('test', expName, plate, 'B02', site)
            x = convert_tensor_to_rgb(t1,channels=channels, rgb_map = RGB_MAP)
            cv2.imwrite(f'negatives/{cell}-{experiment}_{plate}_{site}.jpg',x)
            #exit()
