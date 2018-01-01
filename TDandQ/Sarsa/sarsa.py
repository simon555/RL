
import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys


from collections import defaultdict
from envs.windy_gridworld import WindyGridworldEnv
from utils import plotting



env = WindyGridworldEnv()

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


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        state=env.reset()
        probs=policy(state)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        
        done=False
        length_episode=0
        reward_episode=0
        while not done:
            next_state, reward, done, _ = env.step(action)
            reward_episode+=reward
            probs_p=policy(next_state)
            action_p = np.random.choice(np.arange(len(probs_p)), p=probs_p)

            Q[state][action]+=alpha*(reward+discount_factor*Q[next_state][action_p]-Q[state][action])
            
            state=next_state
            action=action_p
            length_episode+=1
            
        stats.episode_lengths[i_episode]=length_episode
        stats.episode_rewards[i_episode]=reward_episode
        # Implement this!
    
    return Q, stats

Q, stats = sarsa(env, 200)

plotting.plot_episode_stats(stats)