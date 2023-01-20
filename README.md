# Cartpole Reinforcement Learning

This repository follows a long with the [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) on Reinforcement Learning, specifically Deep Q Learning (DQN).

We make use of [OpenAI Gym](https://gymnasium.farama.org/).

## DQN algorithm

Our environment is deterministic, so all equations presented here are also formulated deterministically for the sake of simplicity. In the reinforcement learning literature, they would also contain expectations over stochastic transitions in the environment.

Our aim will be to train a policy that tries to maximize the discounted, cumulative reward 

$$R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t$$

where $R_{t_0}$ is also known as the *return*. The discount, $\gamma$, should be a constant between $0$ and $1$ that ensures the sum converges. A lower $\gamma$ makes rewards from the uncertain far future less important for our agent than the ones in the near future that it can be fairly confident about. It also encourages agents to collect reward closer in time than equivalent rewards that are temporally far away in the future.

The main idea behind Q-learning is that if we had a function $Q^*: State \times Action \rightarrow \mathbb{R}$, that could tell us what our return would be, if we were to take an action in a given state, then we could easily construct a policy that maximizes our rewards: 

$$\pi^*(s) = \arg\max_a Q^{\ast}(s, a)$$

However, we don’t know everything about the world, so we don’t have access to $Q^{\ast}$ . But, since neural networks are universal function approximators, we can simply create one and train it to resemble $Q^{\ast}$.

For our training update rule, we’ll use a fact that every $Q$ function for some policy obeys the *Bellman equation*:

$$Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))$$

The difference between the two sides of the equality is known as the *temporal difference* error, $\delta$:

$$\delta = Q(s, a) - (r + \gamma \max_a' Q(s', a))$$

To minimise this error, we will use the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss). The Huber loss acts like the mean squared error when the error is small, but like the mean absolute error when the error is large - this makes it more robust to outliers when the estimates of $Q$ are very noisy. We calculate this over a batch of transitions, $B$, sampled from the replay memory:

$$
\mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta)$$

$$\text{where} \quad \mathcal{L}(\delta) = 
\begin{cases} 
  \frac{1}{2}{\delta^2} & \text{for} |\delta| \le 1, \\ 
  |\delta| - \frac{1}{2} & \text{otherwise.} 
\end{cases}
​$$