# -*- coding: utf-8 -*-
import gym
import matplotlib
import numpy as np
import sys
import matplotlib.pyplot as pl


from collections import defaultdict

from envs.blackjack import BlackjackEnv
from lib import plotting
from lib.plotting import plot_policy
matplotlib.style.use('ggplot')


env = BlackjackEnv()    

def generate_episode(policy,env):
    #episode = [(state,action,reward)]
    episode=[]
    obs=env.reset()
    done=False
    while not done:
        probs = policy(state)
        action = np.random.choice(np.arange(len(probs)), p=probs)
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
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
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
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    evolution=[]
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        delta=0
        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Find all (state, action) pairs we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            # Find the first occurance of the (state, action) pair in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            # Sum up all rewards since the first occurance
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            newMean=returns_sum[sa_pair] / returns_count[sa_pair]
            delta+=np.abs(Q[state][action]-newMean)
            Q[state][action] = newMean
        evolution+=[delta]
        # The policy is improved implicitly by changing the Q dictionar
    
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
    
#on 500K iterations
Q500K, policy,evolution500K = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)
# For plotting: Create value function from action-value function
# by picking the best action at each state
V500K = defaultdict(float)
for state, actions in Q500K.items():
    action_value = np.max(actions)
    V500K[state] = action_value
plotting.plot_value_function(V500K, title="Optimal Value Function_500K")

plot_results(evolution500K,'evolution_500K')
plot_policy(Q500K,policy,'policy_500K')


#on 1M iterations
Q1M, policy,evolution1M = mc_control_epsilon_greedy(env, num_episodes=1000000, epsilon=0.1)
# For plotting: Create value function from action-value function
# by picking the best action at each state
V1M = defaultdict(float)
for state, actions in Q1M.items():
    action_value = np.max(actions)
    V1M[state] = action_value
plotting.plot_value_function(V1M, title="Optimal Value Function_1M")

plot_results(evolution1M,'evolution_1M')

plot_policy(Q1M,policy,'policy_1M')


#on 5M iterations
Q5M, policy,evolution5M = mc_control_epsilon_greedy(env, num_episodes=5000000, epsilon=0.1)
# For plotting: Create value function from action-value function
# by picking the best action at each state
V5M = defaultdict(float)
for state, actions in Q5M.items():
    action_value = np.max(actions)
    V5M[state] = action_value
plotting.plot_value_function(V5M, title="Optimal Value Function_5M")

plot_results(evolution5M,'evolution_5M')

plot_policy(Q5M,policy,'policy_5M')

