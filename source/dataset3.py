__author__ = 'Hao - 6/2020'

import os
import numpy as np
from PIL import Image
import math
import random
from torch.utils.data import Dataset
import cv2
import time

patch_size = 32
split = 0.9
DEBUG = 0

angle_train = [0.02, 0.2, 0.05, 5, 0.5]



class fabricSet_All(Dataset):
    """
    fabric images data loader // just for one image with angles

    load date from normal dataset (angle 0) and rotated dataset with angles

    """
    def __init__(self,
                root,
                img_name,
                angle,
                patch_size=patch_size,
                stride=4,
                transform=None,
                read_disk=True,
                Gabor=True,
                datasets=None): # dataset is rotated data


        datasets2 = fabricSet_One(root, img_name, angle=0, patch_size=patch_size, stride=stride, transform = None)

        # emphasize rotated features, (like add attention to it) 

        self.train_feats = datasets2.train_feats[:int(len(datasets2.train_feats)*0.1)]

        # choose how many angles to add

        # for a in angle_train:
        #     datasets = fabricSet_One(root, img_name, angle=a, patch_size=patch_size, stride=stride, transform = None)
        #     self.train_feats = np.concatenate((self.train_feats, datasets.train_feats), axis=0)

        # datasets = fabricSet_One(root, img_name, angle=angle, patch_size=patch_size, stride=stride, transform = None)
        self.train_feats = np.concatenate((datasets.train_feats, self.train_feats ), axis=0)

        print("train shape: ", self.train_feats.shape)

        self.dim_in = len(self.train_feats[0])




    def __len__(self):
        return len(self.train_feats)


    def __getitem__(self, idx):
        return self.train_feats[idx]




class fabricSet_One(Dataset):
    """
    fabric images data loader // just for one image
    """
    def __init__(self,
                root,
                img_name,
                angle,
                patch_size=patch_size,
                stride=4,
                transform=None,
                read_disk=True,
                Gabor=True):

        # print("******************loading data******************")

        self.angle = angle

        self.img_name = img_name
        self.raw_path  = root + "rotationraw_paper/" + str(self.angle) + "/"
        self.mask_path = root + "rotationmask_paper/" + str(self.angle) + "/"
        # print(self.raw_path, self.mask_path)
        self.root = root
        self.patch_size = patch_size
        self.stride = stride

        self.normal = []
        self.normal_train = []
        self.normal_val = []
        self.defective = []

        self.pos = [] # position: pos[i] = (left, top, right, btm)
        self.pos_train_p = []
        self.pos_val_p = []
        self.pos_n = []
        self.time = 0

        # self._process_data()
        self._image_patches_gray()
        self._random()

        self.gabor = Gabor

        if self.gabor:
            
            self.time = time.perf_counter()

            self.filteredImages = self.loadGaborFeatures()

            self.time = time.perf_counter() - self.time

            self.train_feats, self.val_feats, self.neg_feats = self.generateFeatvector()
            self.dim_in = len(self.train_feats[0])

            # self.rearrangeFeats()

            # # ways to normalize data. its useful in autoencoder,
            self.train_feats = np.asarray(self.train_feats, dtype=np.float32)
            # self.train_feats *= 1.0/self.train_feats.max()
            self.train_feats = (self.train_feats - self.train_feats.min())/(self.train_feats.max() - self.train_feats.min())
            # self.train_feats = (self.train_feats - np.mean(self.train_feats))/(np.std(self.train_feats))
            # self.train_feats = self.train_feats.reshape(-1, 2, 90)

            self.val_feats = np.asarray(self.val_feats, dtype=np.float32)
            # self.val_feats *= 1.0/self.val_feats.max()
            self.val_feats = (self.val_feats - self.val_feats.min())/(self.val_feats.max() - self.val_feats.min())
            # self.val_feats = (self.val_feats - np.mean(self.val_feats))/(np.std(self.val_feats))
            # self.val_feats = self.val_feats.reshape(-1, 2, 90)

            self.neg_feats = np.asarray(self.neg_feats, dtype=np.float32)
            # self.neg_feats *= 1.0/self.neg_feats.max()
            self.neg_feats = (self.neg_feats - self.neg_feats.min())/(self.neg_feats.max() - self.neg_feats.min()) # good for old and relu
            # self.neg_feats = (self.neg_feats - np.mean(self.neg_feats))/(np.std(self.neg_feats))
            # self.neg_feats = self.neg_feats.reshape(-1, 2, 90)



            print("train shape: ", self.train_feats.shape)
            print("valid shape: ", self.val_feats.shape)
            print("negat shape: ", self.neg_feats.shape)



    '''
    create Gabor filter bank and filter image
    '''
    def loadGaborFeatures(self):

        img = cv2.imread(self.raw_path+self.img_name, cv2.IMREAD_GRAYSCALE)
        k_gaborSize = setGaborsize() 
        gammaSigmaPsi = setGammaSigmaPsi()
        lambdas = setLambda(img)
        thetas = setTheta()
        # print(lambdas, thetas)

        # # old gabor
        filters = createGaborfiltersbank(lambdas, k_gaborSize, thetas, gammaSigmaPsi)
        filteredImages, featureParams = applyFiltersbank(filters, img)
        filteredImages = np.asarray(filteredImages, dtype=np.float32)

        return filteredImages

    '''
    create feature vector. 
    '''
    def generateFeatvector(self):
        feats_train = []
        for i in range(len(self.pos_train_p)):
            feat_patch = featurePatch(self.img_name, row=self.pos_train_p[i][1], col=self.pos_train_p[i][0], filteredImages=self.filteredImages, patch_size=patch_size, save=0)
            feats_train.append(feat_patch)
        # print("traning features shape: {0} by {1} ".format(len(feats_train), len(feats_train[0])))

        feats_val = []
        for i in range(len(self.pos_val_p)):
            feat_patch = featurePatch(self.img_name, row=self.pos_val_p[i][1], col=self.pos_val_p[i][0], filteredImages=self.filteredImages, patch_size=patch_size, save=0)
            feats_val.append(feat_patch)
        # print("validating features shape: {0} by {1} ".format(len(feats_val), len(feats_val[0])))

        feats_neg = []
        for i in range(len(self.pos_n)):
            feat_patch = featurePatch(self.img_name, row=self.pos_n[i][1], col=self.pos_n[i][0], filteredImages=self.filteredImages, patch_size=patch_size, save=DEBUG)
            feats_neg.append(feat_patch)
        # print("negative features shape: {0} by {1} ".format(len(feats_neg), len(feats_neg[0])))
        return feats_train, feats_val, feats_neg

    def rearrangeFeats(self):
        stride = len(self.train_feats[0])/2
        # print(stride)
        for i in range(len(self.train_feats)):
            for j in range(int(stride)):
                if j % 2 == 1:
                    temp = self.train_feats[i][j]
                    self.train_feats[i][j] = self.train_feats[i][j+int(stride)]
                    self.train_feats[i][j+int(stride)] = temp


        stride = len(self.val_feats[0])/2
        # print(stride)
        for i in range(len(self.val_feats)):
            for j in range(int(stride)):
                if j % 2 == 1:
                    temp = self.val_feats[i][j]
                    self.val_feats[i][j] = self.val_feats[i][j+int(stride)]
                    self.val_feats[i][j+int(stride)] = temp

        stride = len(self.neg_feats[0])/2
        # print(stride)
        for i in range(len(self.neg_feats)):
            for j in range(int(stride)):
                if j % 2 == 1:
                    temp = self.neg_feats[i][j]
                    self.neg_feats[i][j] = self.neg_feats[i][j+int(stride)]
                    self.neg_feats[i][j+int(stride)] = temp
                

        
    def _random(self):

        mapIndexPosition = list(zip(self.normal, self.pos))
        random.seed(4)
        random.shuffle(mapIndexPosition)
        self.normal, self.pos = zip(*mapIndexPosition)

        num_train = int(len(self.normal)*split)

        # print("total: ", len(self.normal))
        
        self.normal_train = self.normal[0:num_train]
        self.normal_val = self.normal[num_train:]
        self.pos_train_p = self.pos[0:num_train]
        self.pos_val_p = self.pos[num_train:]

        # print("train, val, def(img): ", len(self.normal_train), len(self.normal_val), len(self.defective))
        # print("train, val, def(pos): ", len(self.pos_train_p), len(self.pos_val_p), len(self.pos_n))

    def _process_data(self):
        i = 0
        for file in os.listdir(self.raw_path):
            print("Now processing ", file)
            self._image_patches(file, patch_size=patch_size, find_mask=True)
            i += 1
            if i == 1:
                print("Done")
                break

    '''
    crop image patches
    '''

    def _image_patches_gray(self,
                            patch_size=patch_size,
                            find_mask=True):

        img = Image.open(self.raw_path + self.img_name)
        width, height = img.size

        for top in range(0, img.size[1]-patch_size, self.stride):
            for left in range(0, img.size[0]-patch_size, self.stride):
                # print(y,x)
                crop = img.crop((left, top, left+patch_size, top+patch_size)).convert("L")
                if find_mask:
                    if self._find_mask(left, top) == 1:
                        crop = np.array(crop).astype(np.float32)
                        crop /= 255.0
                        crop = crop[np.newaxis, ...]
                        self.normal.append(crop)
                        self.pos.append((left,top, left+patch_size, top+patch_size))

                    elif self._find_mask(left, top) == -1:
                        crop = np.array(crop).astype(np.float32)
                        crop /= 255.0
                        crop = crop[np.newaxis, ...]
                        self.defective.append(crop)
                        self.pos_n.append((left,top, left+patch_size, top+patch_size))
                    else:
                        continue

                else:
                    crop = np.array(crop).astype(np.float32)
                    crop /= 255.0
                    crop = crop[np.newaxis, ...]
                    self.normal.append(crop)
                    self.pos.append((left,top, left+patch_size, top+patch_size))
                    



    def _find_mask(self,
                    left,
                    top,
                    patch_size=patch_size):

        """
        -1 : defective
         1 : normal
         0 : abandon
        """

        mask = Image.open(self.mask_path + self.img_name.replace("jpg", "tif")) 
        # crop = mask.crop((col, row, col+patch_size, row+patch_size)) 
        mask_pixels = mask.load()
        thre = 1
        c = 0
        for i in range(top, top+patch_size):
            for j in range(left, left+patch_size):
                if mask_pixels[i,j] == (255, 0, 0):
                    c += 1
                if c >= thre:
                    return -1
        return 1




    def __len__(self):
        if self.gabor:
            return len(self.train_feats)
        else:
            return len(self.normal_train)

    def __getitem__(self, idx):
        if self.gabor:
            return self.train_feats[idx]
        else:
            return self.normal_train[idx]



def featurePatch(filename, row=0, col=0, filteredImages=None, patch_size=32, save=0):
    num_filter = len(filteredImages)

    train_feat = []

    for i in range(num_filter):
        feat_patch = filteredImages[i][row:row+patch_size, col:col+patch_size]
        train_feat.append(np.mean(feat_patch))
        # train_feat.append(np.median(feat_patch))
        train_feat.append(np.std(feat_patch))
        if save:
            savefeat(feat_patch, i, filename, row, col)

    train_feat = np.asarray(train_feat, dtype=np.float32)

    return train_feat


def createGaborfiltersbank(lambdas, gaborsize, thetas, gammaSigmaPsi):
    """
    create gabor filter banks
    """
    filters = []

    thetaInRad = [ -x/180*math.pi for x in thetas] #np.deg2rad(x)
    # for ksize in gaborsize:
    for theta in thetaInRad:
        i = 0
        for lamb in lambdas:
            ksize = gaborsize[i]
            params = {'ksize': (0, 0), 'sigma': 0.3*((ksize-1)*0.5 - 1) + 0.8, 'theta': theta, 'lambd': lamb,
                   'gamma':gammaSigmaPsi[0], 'psi': gammaSigmaPsi[2], 'ktype': cv2.CV_64F}
            kernel = cv2.getGaborKernel(**params)
            # kernel /= 1.5* kernel.sum()
            filters.append((kernel, params))
            i += 1
    return filters



def applyFiltersbank(filters, img):
    """
    Apply gabor filters to img
    """

    featureImages = []
    featureParams = []
    for filter in filters:
        kernel, params = filter
        filter_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        # print(kernel.dtype)
        # print(filter_img.dtype)
        featureImages.append(filter_img)
        featureParams.append(params)
    # print("total filters is: ", len(featureImages))
    return featureImages, featureParams


def setTheta():
    """
    create different orientations

    thetas - orientation of the normal to the parallel stripes
    """
    thetas = []
    for i in range(0, 180, 10):
        thetas.append(i)
    # thetas.extend([0, 45, 90, 135])
    # print("thetas is: ", thetas, "length is: ", len(thetas))
    return thetas

def setLambda(img):
    """
    set lambda - wavelength of the sunusoidal factor
    convert radial frequencies to wavelengths.
    
    """

    height, width = img.shape

    #calculate radial frequencies.
    max = (width/4) * math.sqrt(2)
    min = 4 * math.sqrt(2)
    temp = min
    radialFrequencies = []

    # Lambda 
    while(temp < max):
        radialFrequencies.append(temp)
        temp = temp * 2

    radialFrequencies.append(max)
    # radialFrequencies = [4,6,8,10,12]
    lambdaVals = []
    for freq in radialFrequencies:
        lambdaVals.append(width/freq)

    lambdaVals = [4,6,8,10,12]
    # print("lambdaVals is: ", lambdaVals, "length is: ", len(lambdaVals))

    return lambdaVals


def setGammaSigmaPsi():
    """
    sigma - standard deviation of the gaussian function
    gamma - spatial aspect ratio
    psi - phase offset

    # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8 in gaussian 
    """

    gamma = 0.5 
    sigma = 2*np.pi  # 14 -- 1/ lambda
    psi =  np.pi * 0.5
    gammaSigmaPsi=[]
    gammaSigmaPsi.append(gamma)
    gammaSigmaPsi.append(sigma)
    gammaSigmaPsi.append(psi)
    return gammaSigmaPsi

def setGaborsize():
    """
    create ksizes - size of gabor filter (n, n)
    """
    
    k_gaborSize = [65,97,127,159,191]

    return k_gaborSize




