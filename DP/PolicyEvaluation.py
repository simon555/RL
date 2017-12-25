# -*- coding: utf-8 -*-


# =============================================================================
# Loading Modules
# =============================================================================
import numpy as np
from envs.GridWorld import GridworldEnv
import matplotlib.pyplot as pl



# =============================================================================
# Create Environment
# =============================================================================
env = GridworldEnv()




# =============================================================================
# Evaluate Policy
# =============================================================================


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
    k=0
    diff=1
    errors=[]
    while (diff>theta):
        newV=np.zeros(env.nS)
        for s in range(env.nS):
            temp1=0
            for a in range(env.nA):
                temp2=0
                for sp in range(len(env.P[s][a])):
                    transition=env.P[s][a][sp]
                    temp2+=transition[0]*(transition[2]+discount_factor*V[transition[1]])
                    #print('action ',a, 'temp ',temp2)
                temp1+=policy[s][a]*temp2
            newV[s]=temp1
        diff=np.abs(np.subtract(V, newV)).mean()  
        errors+=[diff]
        V=newV
        k+=1
        print('iteration = ',k,'  delta = %.3e'%diff)
      
    
    return (np.array(V),errors)


random_policy = np.ones([env.nS, env.nA]) / env.nA
v,errors = policy_eval(random_policy, env)

expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

print("computed Value Function:")
print(v.reshape(env.shape))
print("")

print("expected Value Function:")
print(expected_v.reshape(env.shape))
print("")

fig=pl.figure()
pl.plot(errors)
pl.xlabel('iteration')
pl.ylabel('L1 difference of deltas')
pl.title('evolution of the Value function deltas at each iteration')