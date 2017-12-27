# -*- coding: utf-8 -*-
import gym
import matplotlib
import numpy as np
import sys
import matplotlib.pyplot as pl


from collections import defaultdict

from envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')


env = BlackjackEnv()    

def generate_episode(policy,env):
    #episode = [(state,action,reward)]
    episode=[]
    obs=env.reset()
    done=False
    while not done:
        action=np.argmax(policy(obs))
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
    

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        alea=np.random.rand()
        if alea<epsilon:
            return(np.random.rand(nA))
        else:
            actionToTake=np.argmax(Q[observation])
            output=np.zeros(nA)
            output[actionToTake]=1
            return(output)
        
    return policy_fn


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_count = defaultdict(float)
    returns_sum = defaultdict(float)
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    evolution=[]
    
    for k in range(num_episodes):
        # The policy we're following
    
        #generate episode using this policy
        episode=generate_episode(policy,env)       
        
        visited=defaultdict(bool)   
        delta=0
        for i,transition in enumerate(episode):
            state,action,reward=transition
            if not visited[(state,action)]:
                visited[(state,action)]=True
                return_i=compute_return(episode,i,discount_factor)
                returns_count[(state,action)]+=1
                #Q[state][action]+=(return_i-Q[state][action])/returns_count[(state,action)]
                returns_sum[(state,action)] += return_i
                newMean=returns_sum[(state,action)] / returns_count[(state,action)]
                delta+=np.abs(Q[state][action] -newMean)
                Q[state][action] = newMean
        evolution+=[delta]
        
        
        
        print('iteration ',k)
    return (Q, policy,evolution)


def plot_results(evolution,index):
    fig=pl.figure()
    pl.plot(evolution)
    pl.title('variations of the computed means')
    pl.xlabel('iteration')
    pl.ylabel('L1 difference of variations')
    pl.legend()
    pl.savefig(index+'_evolutionMean.png')
    pl.show()
    

Q, policy,evolution = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)


# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")

plot_results(evolution,'5000K')



