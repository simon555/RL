#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 12:43:50 2017

@author: sebbaghs
"""
import time


import gym
env = gym.make('GridWorld')
env.reset()
for i in range(1000):
    env.render()
    env.step()
    
    time.sleep(0.01)
    
    