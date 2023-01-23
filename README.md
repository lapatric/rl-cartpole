# Cartpole Reinforcement Learning (DQN)

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

$$\delta = Q(s, a) - (r + \gamma \max_a Q(s', a))$$

To minimise this error, we will use the [Huber loss](https://en.wikipedia.org/wiki/Huber_loss). The Huber loss acts like the mean squared error when the error is small, but like the mean absolute error when the error is large - this makes it more robust to outliers when the estimates of $Q$ are very noisy. We calculate this over a batch of transitions, $B$, sampled from the replay memory:

$$
\mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta)$$

$$\text{where} \quad \mathcal{L}(\delta) = 
\begin{cases} 
  \frac{1}{2}{\delta^2} & \text{for} |\delta| \le 1, \\ 
  |\delta| - \frac{1}{2} & \text{otherwise.} 
\end{cases}
​$$


## Implementation

We begin by instantiating the environment, `env = gym.make('CartPole-v1')`. Details about the state space and action space of the CartPole environment can be found [here](https://gymnasium.farama.org/environments/classic_control/cart_pole/). In short, the *state space* (pole position and velocity) is given by 4 real values. The *action space* is $\{0, 1\}$, indicating the direction (left or right) in which the cart is pushed with a fixed force. Since the goal is to keep the pole upright for as long as possible, a reward of $+1$ is allotted for every step taken, including the termination step. The threshold for rewards is $475$ for `v1`.

Similar to model-free Q-learning, training begins by resetting the environment to some initial state $s_0$ and picking an action to peform in this state. Depending on our *exploration threshold*, $\epsilon$, we pick an action based on the current state of the policy network or uniformly at random from the action space, $\{\text{left}, \text{right}\}$. We then use the environment object `env` to take a step in the chosen direction and obtain the next state, the received reward, and whether a terminal state has been reached.

```python
# reset environment
state, _ = env.reset()

# pick action to take
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
if random.random() > eps_threshold:
  with torch.no_grad():
    return policy_net(state).max(1)[1].view(1, 1)
else:
  return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# take step with chosen action
next_state, reward, terminated, truncated, _ = env.step(action.item())
next_state = None if terminated else torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
```

Given this transition $(s, a) \to (r, s')$ w.r.t. our environment, we can now train our policy network to predict the expected discounted return $Q(s, a)$ for this state-action pair. We do so by passing $s$ through the network which yields $[Q(s, \text{left})$, $Q(s, \text{right})]$. We then compute a loss on the relevant value $Q(s, a)$, $a \in \{\text{left}, \text{right}\}$, using insights from the *Bellman Equation* as discussed above:

$$loss(Q(s, a), \ r + \gamma \max_a Q(s', a)).$$

```python
# compute [Q(s, left), Q(s, right)] and select the relevant one for the performed action
state_action_value = policy_net(state).gather(1, action)

# compute expected return of the next state max_{a'} Q(s', a')
if next_state is None:
  next_state_value = 0
else:
  with torch.no_grad():
    next_state_value = target_net(next_state).max(1)[0]

# compute expected discounted return r + gamma * max_{a'} Q(s', a')
expected_state_action_value = reward + (GAMMA * next_state_value)

# Compute the Huber loss on the relevant output of the policy network
# Finally we peform back-propagation to update the weights of the policy_net (omitted here)
loss = nn.SmoothL1Loss(state_action_value, expected_state_action_value)

```

Notice how we predict the *expected discounted return*, $\max_a Q(s', a)$, using a second neural network `target_net`. This approach is taken for added stability during training. Furthermore, we only compute the extected return value on *non-terminal* states. For *terminal* states, we take the known return value of $0$ as our ground-truth.

Both the `policy` and `target` network are identical in architecture, however, instead of using back-propagation as we do for the `policy` network, we use a manual approach to softly update the weights of the `target` network, leveraging the newly updated weights of our `policy` network:

```python
# Soft update of the target network's weights θ′ ← τ θ + (1 −τ )θ′
target_net_state_dict = target_net.state_dict()
policy_net_state_dict = policy_net.state_dict()
for key in policy_net_state_dict:
  target_net_state_dict[key] = TAU * policy_net_state_dict[key] + (1-TAU) * target_net_state_dict[key]
target_net.load_state_dict(target_net_state_dict)
```

## Memory Replay

As layed out in the sections above, each step in our environment yields a transition $(s, a) \to (r, s')$ which we use to train our policy (and target) network using the contraint enforced by the Bellman Equation, $loss(Q_{policy}(s, a), \ r + \gamma \max_a Q_{target}(s', a))$. Now, to speed up the learning process of our model we store each observed transition in a list and subsequently train our model over a batch of transitions at a time. In particular, each batch is sampled u.a.r. from the list of all observed transitions, making the transitions that build up a batch decorrelated. This has been shown to greatly stabilize and improve the DQN training procedure.

## Training on a GPU cloud

We use [Lambda Cloud](https://lambdalabs.com/service/gpu-cloud) for training as it provides a powerful and cost-efficient environment for training neural networks on an HPC. This section assumes that you've already [set up an account](https://cloud.lambdalabs.com/sign-up) at Lambda Labs. 

### Setting up SSH access

To begin, we create a *public-private key pair* which we will use to connect to the cloud via SSH. Upon using the `ssh-keygen` command as specified below, you'll be prompted to specify a name and secret passphrase for your key pair. We'll name our key "lambda_ai" and leave the passphrase blank. Finally, we use `chmod` to restrict the access permissions to our private key, thus keeping it safe.

```bash
cd ~/.ssh
# generate public-private key pair
ssh-keygen -t ed25519
chmod 600 ./lambda_ai
# print public key 
cat ./lambda_ai.pub
```

Now, we enter our [Lambda Dashboard](https://cloud.lambdalabs.com/instances) and add our newly generated public SSH key to our set of trusted keys under *SSH keys > Add SSH key*. When prompted, paste in all the contents of your *public* key (see `cat lambda_ai.pub` above) into the text field and confirm. Never share the *private* part of your public-private key pair with anyone!

### Launching a GPU instance

To launch a GPU instance from the [Lambda Dashboard](https://cloud.lambdalabs.com/instances) we navigate to *Instances > Launch Instance* and proceed by specifying the *Instance type*, *Region* and *Filesystem* and picking the SSH key we added previously.

Once our instance is *Running* (refer to *STATUS*), we copy the *SSH LOGIN* details and run the following command inside a terminal. We use `-i ~/.ssh/lambda_ai` to specify the private key associated with our public key.

```bash
ssh -i ~/.ssh/lambda_ai <ubuntu@instance-ip-address>
```

Now we should be logged in to our very own remote GPU instance. Awesome! To check the instance specifications we can run the `nvidia-smi` command inside the terminal.