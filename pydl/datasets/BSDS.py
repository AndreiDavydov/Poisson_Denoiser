#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 22:58:26 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmatis@skoltech.ru
"""

import os
import numpy as np
import torch as th
import torch.utils.data as data
from pydl.utils import gen_imdb_BSDS500_fromList, imfilter2D_SpatialDomain,\
getPad2RetainShape,crop2D
from matplotlib import image as Img

class BSDS(data.Dataset):
    
    def __init__(self,stdn,random_seed=20180102,filepath='',train=True,\
                 color=False,shape=(180,180),im2Tensor = True):
        
        assert(isinstance(stdn,(float,int,tuple))),"stdn is expected to be either "\
        +"a float or an int or a tuple"
                
        if isinstance(stdn,float):
            stdn = (stdn,)
        if isinstance(stdn,tuple):
            stdn = tuple(float(i) for i in stdn)
        
        self.stdn = np.asarray(stdn) 
        self.train = train
        self.rng = np.random.RandomState(random_seed)
        
        if im2Tensor:
            fshape = (3,2,0,1)
        else:
            fshape = (3,0,1,2)
              
        if self.train:
            if os.path.isfile(filepath):
                f = np.load(filepath)
                self.train_gt = f['train_set'].transpose(fshape)
            else:
                currentPath = os.path.dirname(os.path.realpath(__file__))
                listPath = os.path.join(currentPath,"../../datasets/BSDS500/BSDS_validation_list.txt")
                imdbPath = os.path.join(currentPath,"../../datasets/BSDS500/")
                self.train_gt = gen_imdb_BSDS500_fromList(color=color,\
                                    listPath = listPath, imdbPath = imdbPath,\
                                    shape=shape,data ='train').transpose(fshape)
            
            self.train_data = self.generate_NoisyData()
        else:
            if os.path.isfile(filepath):
                f = np.load(filepath)
                self.test_gt = f['test_set'].transpose(fshape)
            else:
                currentPath = os.path.dirname(os.path.realpath(__file__))
                listPath = os.path.join(currentPath,"../../datasets/BSDS500/BSDS_validation_list.txt")
                imdbPath = os.path.join(currentPath,"../../datasets/BSDS500/")
                self.test_gt = gen_imdb_BSDS500_fromList(color=color,\
                                    listPath = listPath, imdbPath = imdbPath,\
                                    shape=shape,data ='test').transpose(fshape)
            
            self.test_data = self.generate_NoisyData()
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, noise_std).
        """
        
        if self.train:
            img, target, noise_std = self.train_data[index],\
                                     self.train_gt[index%len(self.train_gt)],\
                                     self.stdn.astype(self.train_gt.dtype)[index//len(self.train_gt)]
        else:
            img, target, noise_std = self.test_data[index],\
                                     self.test_gt[index%len(self.test_gt)],\
                                     self.stdn.astype(self.test_gt.dtype)[index//len(self.test_gt)]
        
        return img,target,noise_std
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
    def generate_NoisyData(self):
        r"""Create noisy observations using the ground-truth data."""
        if self.train:
            input = self.train_gt
        else:
            input = self.test_gt
            
        shape = input.shape
        dtype = input.dtype
                        
        ndata_shape = (shape[0]*len(self.stdn),)+shape[1:]
        ndata = np.empty(ndata_shape,dtype=dtype)
        
        for i in range(len(self.stdn)):
            noise = self.stdn[i]*self.rng.randn(*shape)
            noise = noise.astype(dtype)
            ndata[shape[0]*i:shape[0]*(i+1),...] = input+noise
        
        return ndata

class BSDS_v2(data.Dataset):
    
    def __init__(self,stdn,random_seed=20180102,filepath='',train=True,\
                 color=False,shape=(180,180),im2Tensor = True):
        
        assert(isinstance(stdn,(float,int,tuple))),"stdn is expected to be either "\
        +"a float or an int or a tuple"
                
        if isinstance(stdn,(int,float)):
            stdn = (stdn,)
        if isinstance(stdn,tuple):
            stdn = tuple(float(i) for i in stdn)
        
        self.stdn = np.asarray(stdn) 
        self.train = train
        self.rng = np.random.RandomState(random_seed)
        
        if im2Tensor:
            fshape = (3,2,0,1)
        else:
            fshape = (3,0,1,2)
              
        if self.train:
            if os.path.isfile(filepath):
                f = np.load(filepath)
                self.train_gt = f['train_set'].transpose(fshape)
            else:
                currentPath = os.path.dirname(os.path.realpath(__file__))
                listPath = os.path.join(currentPath,"../../datasets/BSDS500/BSDS_validation_list.txt")
                imdbPath = os.path.join(currentPath,"../../datasets/BSDS500/")
                self.train_gt = gen_imdb_BSDS500_fromList(color=color,\
                                    listPath = listPath, imdbPath = imdbPath,\
                                    shape=shape,data ='train').transpose(fshape)
            
            self.train_data = self.generate_NoisyData()
            self.train_obs = self.train_data
        else:
            if os.path.isfile(filepath):
                f = np.load(filepath)
                self.test_gt = f['test_set'].transpose(fshape)
            else:
                currentPath = os.path.dirname(os.path.realpath(__file__))
                listPath = os.path.join(currentPath,"../../datasets/BSDS500/BSDS_validation_list.txt")
                imdbPath = os.path.join(currentPath,"../../datasets/BSDS500/")
                self.test_gt = gen_imdb_BSDS500_fromList(color=color,\
                                    listPath = listPath, imdbPath = imdbPath,\
                                    shape=shape,data ='test').transpose(fshape)
            
            self.test_data = self.generate_NoisyData()
            self.test_obs = self.test_data
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, noise_std).
        """
        
        if self.train:
            img, target, noise_std = self.train_data[index],\
                                     self.train_gt[index%len(self.train_gt)],\
                                     self.stdn.astype(self.train_gt.dtype)[index//len(self.train_gt)]
            
            #obs = np.nan if self.train_obs is None else self.train_obs[index]
            obs = self.train_obs[index]   
        else:
            img, target, noise_std = self.test_data[index],\
                                     self.test_gt[index%len(self.test_gt)],\
                                     self.stdn.astype(self.test_gt.dtype)[index//len(self.test_gt)]

            #obs = np.nan if self.test_obs is None else self.test_obs[index]
            obs = self.test_obs[index]
            
        return img,target,noise_std,obs
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
    def generate_NoisyData(self):
        r"""Create noisy observations using the ground-truth data."""
        if self.train:
            input = self.train_gt
        else:
            input = self.test_gt
            
        shape = input.shape
        dtype = input.dtype
                        
        ndata_shape = (shape[0]*len(self.stdn),)+shape[1:]
        ndata = np.empty(ndata_shape,dtype=dtype)
        
        for i in range(len(self.stdn)):
            noise = self.stdn[i]*self.rng.randn(*shape)
            noise = noise.astype(dtype)
            ndata[shape[0]*i:shape[0]*(i+1),...] = input+noise
        
        return ndata

class BSDS68(data.Dataset):
    
    def __init__(self,stdn,random_seed=20180102,tall=True,color=False,\
                 im2Tensor=True,dtype='f', filename=\
                 "../../datasets/BSDS500/BSDS_validation_list.txt"):
        
        assert(isinstance(stdn,(float,int,tuple))),"stdn is expected to be either "\
        +"a float or an int or a tuple"
                
        if isinstance(stdn,(float,int)):
            stdn = (stdn,)
        if isinstance(stdn,tuple):
            stdn = tuple(float(i) for i in stdn)
        
        self.stdn = np.asarray(stdn) 
        self.tall = tall
        self.rng = np.random.RandomState(random_seed)
        
        if im2Tensor:
            fshape = (3,2,0,1)
        else:
            fshape = (3,0,1,2)
               
        currentPath = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(currentPath,filename)
        dbPath = os.path.dirname(filename)
        
        with open(filename) as f:
            imList = f.readlines()
            imList = [f.strip() for f in imList]

        if color:
            dbPath = os.path.join(dbPath,'color')
        else:
            dbPath = os.path.join(dbPath,'gray')
        
        
        img = np.ndarray(0)
        if tall:
            for i in imList:
                tmp = Img.imread(os.path.join(dbPath,i)).astype(dtype)
                if color:
                    tmp.shape += (1,)
                else:
                    tmp.shape += (1,1)
                if tmp.shape[0] > tmp.shape[1]:
                    img = np.concatenate((img,tmp),axis=3) if img.size else tmp
        else:
            for i in imList:
                tmp = Img.imread(os.path.join(dbPath,i)).astype(dtype)
                if color:
                    tmp.shape += (1,)
                else:
                    tmp.shape += (1,1)
                if tmp.shape[0] <= tmp.shape[1]:
                    img = np.concatenate((img,tmp),axis=3) if img.size else tmp
                    
        
        self.img_gt = img.transpose(fshape)
        self.img_noisy = self.generate_NoisyData()
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img, target, noise_std).
        """
        
        img, target, noise_std = self.img_noisy[index],\
                                 self.img_gt[index%len(self.img_gt)],\
                                 self.stdn.astype(self.img_gt.dtype)[index//len(self.img_gt)]
        
        return img,target,noise_std
    
    def __len__(self):
        
        return len(self.img_noisy)
    
    def generate_NoisyData(self):
        r"""Create noisy observations using the ground-truth data."""
            
        shape = self.img_gt.shape
        dtype = self.img_gt.dtype
                        
        ndata_shape = (shape[0]*len(self.stdn),)+shape[1:]
        ndata = np.empty(ndata_shape,dtype=dtype)
        
        for i in range(len(self.stdn)):
            noise = self.stdn[i]*self.rng.randn(*shape)
            noise = noise.astype(dtype)
            ndata[shape[0]*i:shape[0]*(i+1),...] = self.img_gt+noise
        
        return ndata
    
class BSDS_deblur(data.Dataset):
    
    def __init__(self,kernel,stdn,random_seed=20180102,filepath='',train=True,\
                 color=False,shape=(256,256),padType='valid',batchSize = 40,\
                 mask = None):
        r"""batchSize is used internally to create the blurred data so that 
        the system memory can be suffient.
        mask: In case we don't want to use all the training or testing ground
        truth data the mask should contain the indices of the images to be used.
        Otherwise it should be set to None.
        """        
        assert(isinstance(stdn,(float,int,tuple))),"stdn is expected to be either "\
        +"a float or an int or a tuple"
                
        if isinstance(stdn,float):
            stdn = (stdn,)
        if isinstance(stdn,tuple):
            stdn = tuple(float(i) for i in stdn)
        
        assert(th.is_tensor(kernel)), 'The blur kernel must be a torch tensor.'
        
        self.kernel = kernel
        self.padType = padType
        self.stdn = th.tensor(stdn).type_as(kernel)
        self.train = train
        self.rnd_seed = random_seed
        self.batchSize = batchSize
        
        crop = None
        if self.padType == "valid":
            crop = getPad2RetainShape(kernel.shape)
        
        fshape = (3,2,0,1)
              
        if self.train:
            if os.path.isfile(filepath):
                f = np.load(filepath)
                self.train_gt = f['train_set'].transpose(fshape)
            else:
                currentPath = os.path.dirname(os.path.realpath(__file__))
                listPath = os.path.join(currentPath,"../../datasets/BSDS500/BSDS_validation_list.txt")
                imdbPath = os.path.join(currentPath,"../../datasets/BSDS500/")
                self.train_gt = gen_imdb_BSDS500_fromList(color=color,\
                                    listPath = listPath, imdbPath = imdbPath,\
                                    shape=shape,data ='train').transpose(fshape)
            
            if mask is None:
                self.train_gt = th.from_numpy(self.train_gt).type_as(kernel)
            else:
                self.train_gt = th.from_numpy(self.train_gt[mask,...]).type_as(kernel)                
                
            self.train_data = self.generate_BlurredNoisyData()
            if crop is not None:
                self.train_gt = crop2D(self.train_gt,crop)
        else:
            if os.path.isfile(filepath):
                f = np.load(filepath)
                self.test_gt = f['test_set'].transpose(fshape)
            else:
                currentPath = os.path.dirname(os.path.realpath(__file__))
                listPath = os.path.join(currentPath,"../../datasets/BSDS500/BSDS_validation_list.txt")
                imdbPath = os.path.join(currentPath,"../../datasets/BSDS500/")
                self.test_gt = gen_imdb_BSDS500_fromList(color=color,\
                                    listPath = listPath, imdbPath = imdbPath,\
                                    shape=shape,data ='test').transpose(fshape)
            if mask is None:                        
                self.test_gt = th.from_numpy(self.test_gt).type_as(kernel)
            else:
                self.test_gt = th.from_numpy(self.test_gt[mask,...]).type_as(kernel)
            
            self.test_data = self.generate_BlurredNoisyData()
            if crop is not None:
                self.test_gt = crop2D(self.test_gt,crop)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, noise_std).
        """
        
        if self.train:
            img, target, blurKernel, noise_std = self.train_data[index],\
                                     self.train_gt[index%len(self.train_gt)],\
                                     self.kernel,\
                                     self.stdn.type_as(self.train_gt)[index//len(self.train_gt)]
        else:
            img, target, blurKernel, noise_std = self.test_data[index],\
                                     self.test_gt[index%len(self.test_gt)],\
                                     self.kernel,\
                                     self.stdn.type_as(self.test_gt)[index//len(self.test_gt)]
        
        return img,target,blurKernel,noise_std
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
    
    def generate_BlurredNoisyData(self):
        r"""Create blurred+noisy observations using the ground-truth data."""
        if self.train:
            input = self.train_gt
        else:
            input = self.test_gt
            
        shape = tuple(input.shape)
        if self.padType == 'valid':
            bshape = (1,1)+tuple(self.kernel.shape[-2:])
            shape = tuple(shape[i]-bshape[i]+1 for i in range(0,len(shape)))
        dtype = input.dtype
                        
        ndata_shape = (shape[0]*len(self.stdn),)+shape[1:]
        ndata = th.empty(ndata_shape,dtype=dtype)
        
        th.manual_seed(self.rnd_seed)
        
        batches = input.size(0)//self.batchSize # number of batches
        if input.size(0)%self.batchSize != 0:
            batches = batches + 1

        # Compute the blurred input images        
        for j in range(0,batches):
            idx1 = (j+1)*self.batchSize
            if idx1 > shape[0]:
                idx1 = shape[0]
            ndata[j*self.batchSize:idx1,...]=imfilter2D_SpatialDomain(\
                 input[j*self.batchSize:idx1,...],self.kernel,padType=self.padType,mode="conv")
            
        # Add noise to the blurred data
        for i in range(1,len(self.stdn)):
            noise = self.stdn[i]*th.randn(*shape,dtype=dtype)
            ndata[i*shape[0]:(i+1)*shape[0],...] = ndata[0:shape[0],...] + noise
        
        noise = self.stdn[0]*th.randn(*shape,dtype=dtype)
        ndata[0:shape[0],...] += noise
        
        return ndata    
    
#    def generate_BlurredNoisyData(self):
#        r"""Create blurred+noisy observations using the ground-truth data."""
#        if self.train:
#            input = self.train_gt
#        else:
#            input = self.test_gt
#            
#        shape = tuple(input.shape)
#        if self.padType == 'valid':
#            bshape = (1,1)+tuple(self.kernel.shape[-2:])
#            shape = tuple(shape[i]-bshape[i]+1 for i in range(0,len(shape)))
#        dtype = input.dtype
#                        
#        ndata_shape = (shape[0]*len(self.stdn),)+shape[1:]
#        ndata = th.empty(ndata_shape,dtype=dtype)
#        
#        th.manual_seed(self.rnd_seed)
#        
#        batches = input.size(0)//self.batchSize # number of batches
#        if input.size(0)%self.batchSize != 0:
#            batches = batches + 1
#        
#        for i in range(len(self.stdn)):
#            noise = self.stdn[i]*th.randn(*shape,dtype=dtype)
#            for j in range(0,batches):
#                idx1 = shape[0]*i+(j+1)*self.batchSize
#                if idx1 > shape[0]*(i+1):
#                    idx1 = shape[0]*(i+1)
#                
#                idx2 = (j+1)*self.batchSize
#                if idx2 > input.size(0):
#                    idx2 = input.size(0)
#                ndata[shape[0]*i+j*self.batchSize:idx1,...] = \
#                    imfilter2D_SpatialDomain(input[j*self.batchSize:idx2,...],\
#                        self.kernel,padType=self.padType,mode="conv")\
#                                             +noise[j*self.batchSize:idx2,...]
#        
#        return ndata