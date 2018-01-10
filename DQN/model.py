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
    def __init__(self,
                 inputFeatures=1,
                 outputAction=4,
                 learningRate=0.00025):
        
        super(QEstimator, self).__init__()
        self.codeDim=7*7*64
        self.outputAction=outputAction
        self.inputFeatures=inputFeatures
        self.learningRate=learningRate
        
        useCuda=torch.cuda.is_available()
        self.useCuda=useCuda
        
        
        self.Conv1=nn.Conv2d(self.inputFeatures,32,8,stride=4)
        self.Conv2=nn.Conv2d(32,64,4,stride=2)
        self.Conv3=nn.Conv2d(64,64,3,stride=1)
        self.Dense1=nn.Linear(self.codeDim,512)
        self.Dense2=nn.Linear(512,self.outputAction)  
        
                   
        if (self.useCuda):
            self.cuda()             
            
        print('use CUDA : ',self.useCuda) 
        print('model loaded')
        

 

    def forward(self,observation):
        x=self.Conv1(observation)
        x=self.Conv2(x)
        x=self.Conv3(x)
        x=x.view(x.size()[0],-1) 
        x=self.Dense1(x)
        x=self.Dense2(x)      
                 
        return(x)

        
        
    