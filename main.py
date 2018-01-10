#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 12:43:50 2017

@author: sebbaghs
"""
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    done=False
    env.reset()
    for i in range(100):
        env.render()
        _,_,done,_=env.step(env.action_space.sample()) # take a random action