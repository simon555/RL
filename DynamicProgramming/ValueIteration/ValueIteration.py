# -*- coding: utf-8 -*-



import numpy as np
import pprint
from envs.GridWorld import GridworldEnv

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()
import matplotlib.pyplot as pl


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
    """
    

    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])
    k=0
    historical=[]
    errors=[]
    while True:
        delta=0
        historical+=[np.copy(V)]
        error=0
        for s in range(env.nS):
            v=V[s]
            choiceActions=computeTerm(env,discount_factor,V,s)
            V[s]=np.max(choiceActions)
            delta=max(delta,np.abs(v-V[s]))
            error+=np.abs(v-V[s])
        k+=1
        print('iteration ',k, 'error %.3e'%delta)
        errors+=[error/env.nS]
        if delta<theta:
            break
    
    for s in range(env.nS):
        policy[s][np.argmax(V[s])]=1
        
    
    
    return policy, V,historical,errors




def computeTerm(env,discount_factor,V,s):
    output=np.zeros([env.nA])
    for a in range(env.nA): 
        temp=0
        for sp in range(len(env.P[s][a])):
            transition=env.P[s][a][sp]
            temp+=transition[0]*(transition[2]+discount_factor*V[transition[1]])
        output[a]=temp
    return(output)


policy, v,historical,errors = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")


# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

pl.clf()
fig=pl.figure(figsize=(12,5))
for i in range(len(historical)):
    ax=fig.add_subplot('15%d'%(i+1))
    ax.imshow(np.reshape(historical[i],env.shape))
    pl.title('iteration {}'.format(i))
    
    
ax=fig.add_subplot('155')
ax.imshow(np.reshape(expected_v,env.shape))
pl.title('expected value')

pl.savefig('ValueIteration.png')
pl.legend()
pl.show()


fig=pl.figure()
pl.plot(errors)
pl.xlabel('iteration')
pl.ylabel('L1 difference')
pl.title('variation of the Value Function')
pl.legend()
fig.savefig('errorValueIteration.png')

pl.show()
