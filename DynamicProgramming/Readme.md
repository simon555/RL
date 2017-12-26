# Dynamic Programming

We start our RL journey by implementing the basic RL algorithms, using Dynamic Programming

## Policy Evaluation
Below, we track the evolution of the differences between the updates of the Value function that is computed. Through the training, the difference converges towards 0 meaning that the value function converged towards the expected value.
![alt text](https://github.com/simon555/RL/blob/master/DynamicProgramming/PolicyEvaluation/PolicyEvaluation.png)

As final output, we get the value function on the gridworld : 
![alt text](https://github.com/simon555/RL/blob/master/DynamicProgramming/PolicyEvaluation/EvolutionPolicyEvaluation.png)


## Policy Iteration
Using the policy evaluation, we can derive the Policy Iteration algorithm. At each iteration, we improve the policy using a 1-step sight, and compute the Value Function corresponding. When the algorithm converges we reached the optimal policy, below we track the number of different action choices done at each iteration.
![alt text](https://github.com/simon555/RL/blob/master/DynamicProgramming/PolicyIteration/errorPolicyIteration.png)

As final output, we get the value function on the gridworld : 
![alt text](https://github.com/simon555/RL/blob/master/DynamicProgramming/PolicyIteration/PolicyIteration.png)


## Value Iteration
Implementation of the Value Iteration Algorithm, below the variation of the computed value function over iterations.
![alt text](https://github.com/simon555/RL/blob/master/DynamicProgramming/ValueIteration/errorValueIteration.png)

As final output, we get the value function on the gridworld : 
![alt text](https://github.com/simon555/RL/blob/master/DynamicProgramming/ValueIteration/ValueIteration.png)








