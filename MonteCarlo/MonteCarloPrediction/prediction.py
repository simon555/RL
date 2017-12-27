# -*- coding: utf-8 -*-

import gym
import matplotlib
import numpy as np
import sys
import matplotlib.pyplot as pl
from collections import defaultdict

from envs.blackjack import BlackjackEnv
from lib import plotting
import envs
#matplotlib.style.use('ggplot')


env = BlackjackEnv()


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final value function
    V = defaultdict(float)
    evolution=[]
    for k in range(num_episodes):
        
        
        
        delta=0
        episode=generate_episode(sample_policy,env)
        visited=defaultdict(bool)        
        #visited_states=set([transition[0] for transition in episode])
        for i,transition in enumerate(episode):
            state,action,reward=transition
            if not visited[state]:
                visited[state]=True
                return_i=compute_return(episode,i,discount_factor)
                returns_sum[state]+=return_i
                returns_count[state]+=1
                newMean=returns_sum[state]/returns_count[state]
                delta+=np.abs(V[state]-newMean)
                V[state]=newMean
        evolution+=[delta]
        print(k)      
    return (V,evolution)




def sample_policy(observation):
    """
    A policy that sticks if the player score is > 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return np.array([1.0, 0.0]) if score >= 20 else np.array([0.0, 1.0])

def generate_episode(policy,env):
    #episode = [(state,action,reward)]
    episode=[]
    obs=env.reset()
    done=False
    while not done:
        action=np.argmax(sample_policy(obs))
        newObs,reward,done,_=env._step(action)
        episode+=[(obs,action,reward)]
        obs=newObs
    return(episode)

def compute_return(episode,begin,discount_factor):
    output=0
    for i,transition in enumerate(episode[begin:]):
        #print(transition[2]*(discount_factor**i))
        output+=transition[2]*(discount_factor**i)
    return(output)
    

def plot_results(evolution,index):
    fig=pl.figure()
    pl.plot(evolution)
    pl.title('variations of the computed means')
    pl.xlabel('iteration')
    pl.ylabel('L1 difference of variations')
    pl.legend()
    pl.savefig(index+'_evolutionMean.png')
    pl.show()
    
    

V_10k,evolution = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")
plot_results(evolution,'10K')



V_500k,evolution = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")
plot_results(evolution,'500K')


