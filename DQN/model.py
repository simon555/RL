# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:23:42 2017

@author: simon
"""


import torch
import matplotlib.pyplot as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class QEstimator(nn.Module):
    def __init__(self,inputDim=2,outputAction=3,learningRate=0.001):
        
        super(QEstimator, self).__init__()

        useCuda=torch.cuda.is_available()
        self.useCuda=useCuda
        
        self.Dense1=nn.Linear(inputDim,8)
        self.Dense2=nn.Linear(8,16)
        self.Dense3=nn.Linear(16,32)
        self.Dense4=nn.Linear(32,64)
        
        self.Dense5=nn.Linear(64,32)
        self.Dense6=nn.Linear(32,16)
        self.Dense7=nn.Linear(16,8)
        self.Dense8=nn.Linear(8,outputAction)        
        
                   
        if (self.useCuda):
            self.cuda()             
            
        print('use CUDA : ',self.useCuda)
        
        
        
        
        
        print('model loaded')
        

 

    def forward(self,observation):
        
        x=self.Dense1(observation)
        x=self.Dense2(x)
        x=self.Dense3(x)
        x=self.Dense4(x)
        x=self.Dense5(x)
        x=self.Dense6(x)
        x=self.Dense7(x)
        x=self.Dense8(x)
                 
        return(x)
        
        
    