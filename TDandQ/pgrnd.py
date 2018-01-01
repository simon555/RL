import gym
import numpy as np
import sys


from envs.cliff_walking import CliffWalkingEnv


env = CliffWalkingEnv()

print(env.reset())
env.render()

print(env.step(0))
env.render()

print(env.step(1))
env.render()

print(env.step(1))
env.render()

print(env.step(2))
env.render()

