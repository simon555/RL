# -*- coding: utf-8 -*-

import numpy as np
import pprint
from envs.GridWorld import GridworldEnv
import matplotlib.pyplot as pl


pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv(shape=[4,4])

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)



def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    k=0
    historical=[]
    errors=[]
    while True:
        policyStable=True
        error=0
        V=policy_eval_fn(policy,env)
        historical+=[np.copy(V)]
        for s in range(env.nS):
            b=np.argmax(policy[s])
            policy[s]=computeTerm(env,discount_factor,V,s)
            #print(policy[s])
            actionSelected=np.argmax(policy[s])
            if (b!=actionSelected):
                policyStable=False
                error+=1
            policy[s] = np.eye(env.nA)[actionSelected]
        k+=1
        print('iteration ',k, 'error {}'.format(error))
        errors+=[error]
                
        if policyStable==True:
            return (policy, V,historical,errors)
    
    


def computeTerm(env,discount_factor,V,s):
    output=np.zeros([env.nA])
    for a in range(env.nA): 
        temp=0
        for sp in range(len(env.P[s][a])):
            transition=env.P[s][a][sp]
            temp+=transition[0]*(transition[2]+discount_factor*V[transition[1]])
        output[a]=temp
    return(output)



            
    
policy, v, historical,errors = policy_improvement(env)
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



    
fig=pl.figure(figsize=(12,5))
for i in range(len(historical)):
    ax=fig.add_subplot('14%d'%(i+1))
    ax.imshow(np.reshape(historical[i],env.shape))
    pl.title('iteration {}'.format(i))
    
ax=fig.add_subplot('144')
ax.imshow(np.reshape(expected_v,env.shape))
pl.title('expected value')

pl.savefig('PolicyIteration.png')
pl.legend()
pl.show()


fig=pl.figure()
pl.plot(errors)
pl.xlabel('iteration')
pl.ylabel('number of changes')
pl.title('number of changes done on the action selection')
pl.legend()
fig.savefig('errorPolicyIteration.png')

pl.show()






