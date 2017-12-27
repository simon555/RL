## Monte Carlo methods

In this section, we study more realistic ways to solve an MDP. We do not know the dynamics of the system, and we go through experiments in order to estimate relevant quantitites (such as value function).

## Monte Carlo prediction
Using this method, we track the variations of the computed means that estimate the value function. We also display a representation of the value function that we estimate

  ## Using 10K iterations ##

![](https://github.com/simon555/RL/blob/master/MonteCarlo/MonteCarloPrediction/10K_evolutionMean.png)
![](https://github.com/simon555/RL/blob/master/MonteCarlo/MonteCarloPrediction/10%2C000%20Steps%20(No%20Usable%20Ace).png
)
![](https://github.com/simon555/RL/blob/master/MonteCarlo/MonteCarloPrediction/10%2C000%20Steps%20(Usable%20Ace).png
)

  ## using 500K iterations ##
![](https://github.com/simon555/RL/blob/master/MonteCarlo/MonteCarloPrediction/500K_evolutionMean.png)
![](https://github.com/simon555/RL/blob/master/MonteCarlo/MonteCarloPrediction/500%2C000%20Steps%20(No%20Usable%20Ace).png
)
![](https://github.com/simon555/RL/blob/master/MonteCarlo/MonteCarloPrediction/500%2C000%20Steps%20(Usable%20Ace).png
)



## Monte Carlo Control - epsilon greedy
Samely as before, we use the Monte Carlo method to approximate the optimal policy. To do this, we compute the Q function using Monte Carlo estimators. At the end of the training, the algorithm converges towards the optimal policy, and allows us to plot the optimal value fonction. Below we plot the evolution of the variation of the computed means that estimate the Q function, and the V function plotted as a surface.

  ## Using 500K iterations ##

![](https://github.com/simon555/RL/blob/master/MonteCarlo/MonteCarloControl-eGreedy/500K_evolutionMean.png)
![](https://github.com/simon555/RL/blob/master/MonteCarlo/MonteCarloControl-eGreedy/Optimal%20Value%20Function%20(No%20Usable%20Ace).png)
![](https://github.com/simon555/RL/blob/master/MonteCarlo/MonteCarloControl-eGreedy/Optimal%20Value%20Function%20(Usable%20Ace).png)


