
import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
import torch
from utils import plotting
from collections import deque, namedtuple
import scipy
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


env = gym.envs.make("Breakout-v0")
env=env.unwrapped


# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
VALID_ACTIONS = [0, 1, 2, 3]


class StateProcessor():
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        self.offset_crop_x=34
        self.offset_crop_y=0
        self.crop_x=160
        self.crop_y=160
        
#           
    def rgb2gray(self,rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    def process(self, state):
        """
        Args:
            state: A raw RGB numpy array of shape (210,160,3)

        Returns:
            A processed [1, 1, 84, 84] pytorch tensor state representing grayscale values.
        """
        img=self.rgb2gray(state)
        img=img[self.offset_crop_x:self.offset_crop_x+self.crop_x,self.offset_crop_y:self.offset_crop_y+self.crop_y]
        img=scipy.misc.imresize(img, (84,84), interp='nearest', mode=None)
        img=torch.from_numpy(img).float()
        img=img.unsqueeze(0).unsqueeze(0)
        
        return (img)
    
    def buildState(self, frames):
        """ 
        Takes as input 4 frames already processed and concatenates them
        
        
        Args = [4 pytorch tensors of size (1,1,84,84)]

        Returns = a pytorch tensor of size (1,4,84,84)
        """
        output=torch.cat(frames,dim=1)
        if torch.cuda.is_available():
           output=output.cuda()
        
        return(Variable(output))
        
   
     
    
class Estimator(nn.Module):
    def __init__(self,
                 inputFeatures=4,
                 outputAction=4,
                 learningRate=0.00025):
        
        super(Estimator, self).__init__()
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
        
        
        self.criterion=torch.nn.MSELoss()
                   
        if (self.useCuda):
            self.cuda()  
            self.criterion=self.criterion.cuda()
            
        print('use CUDA : ',self.useCuda) 
        self.optimizer=optim.Adam(self.parameters(), lr=self.learningRate)
        print('model loaded')    
        
        
    def forward(self,observation):
        x=self.Conv1(observation)
        x=self.Conv2(x)
        x=self.Conv3(x)
        x=x.view(x.size()[0],-1) 
        x=self.Dense1(x)
        x=self.Dense2(x)      
                 
        return(x)


    def predict(self, s):
        """
        Predicts action values.

        Args:
          s: State input of shape [batch_size, 4, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
          action values.
        """
        return (self.forward(s))

    def update(self, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          s: pytorch tensor State input of shape [batch_size, 4, 84, 84]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        optimizer.zero_grad()
        
        
        
        
        
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss
    
#    
img1=np.random.random((210,160,3))
img2=np.random.random((210,160,3))
img3=np.random.random((210,160,3))
img4=np.random.random((210,160,3))


sp=StateProcessor()
img1=sp.process(img1)
img2=sp.process(img2)
img3=sp.process(img3)
img4=sp.process(img4)

frames=[img1,img2,img3,img4]
state=sp.buildState(frames)
        
model=Estimator()





output=model.predict(state)
