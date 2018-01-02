# SARSA Algorithm

In this sectionm, we use the SARSA algorithm for 200 iterations, to approximate the optimal policy. We use a windy gridworld environment.

 
![](https://github.com/simon555/RL/blob/master/TDandQ/Sarsa/sarsa_reward.png)
![](https://github.com/simon555/RL/blob/master/TDandQ/Sarsa/sarsa_length.png)

  
  
# Q learning

This method is similar to SARSA but we use a different update rule for the estimate of the Q function.

![](https://github.com/simon555/RL/blob/master/TDandQ/QLearning/QLearning_reward.png)
![](https://github.com/simon555/RL/blob/master/TDandQ/QLearning/QLearning_length.png)


# Comments
In this exemple, it seems that SARSA is doing better : the Q learning method gives more noisy convergence even though they both tend to find the optimal policy after 200 iterations. Also, SARSA tends to be more fast.
