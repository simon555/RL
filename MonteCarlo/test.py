 # -*- coding: utf-8 -*-

import gym
from envs import blackjack
import numpy as np


def sample_policy(observation):
    """
    A policy that sticks if the player score is > 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1



env =blackjack.BlackjackEnv()
observation=env.reset()
for _ in range(1000):
    print(observation, end='')
    action=sample_policy(observation)
    print(' action selected : ',action)
    observation,reward,done,x=env.step(env.action_space.sample()) # take a random action
    if done:
        print('-------------------------')
        print('end of the game, your score {}, dealer score {}'.format(observation[0],observation[1]))
        print('reward : ',reward)
        env.reset()
        print('-------------------------')
        print('\n \n')