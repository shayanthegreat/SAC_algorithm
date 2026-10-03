# Soft Actor-Critic (SAC) from Scratch

A PyTorch implementation of **Soft Actor-Critic (SAC)** for continuous-control reinforcement learning.

This project implements SAC from scratch, including the main components of the algorithm:

* **Actor** — learns a stochastic policy for selecting continuous actions.
* **Twin Critics** — two Q-networks used to reduce overestimation bias.
* **Value Network** — estimates the value of each state.
* **Target Value Network** — slowly updated copy of the Value Network for stable training.
* **Replay Buffer** — stores and samples past experiences for off-policy learning.
* **Agent** — coordinates action selection, learning, target updates, and model saving/loading.

## Architecture

```text
                State
                  │
          ┌───────┴───────┐
          │               │
        Actor           Value
          │
        Action
          │
     ┌────┴────┐
     │         │
    Q1         Q2
     │         │
     └────┬────┘
          │
     SAC Updates
```

The implementation follows the **original SAC formulation with a separate Value Network**. Modern SAC implementations typically remove this additional network.

## Key Features

* Continuous action spaces
* Gaussian stochastic policy
* Reparameterization trick
* Twin Q-networks
* Entropy-regularized policy optimization
* Experience replay
* Soft target-network updates
* CPU and GPU support
* Model checkpointing

## Training

The agent follows the standard SAC training procedure:

1. Collect transitions from the environment.
2. Store them in the replay buffer.
3. Sample mini-batches of past experiences.
4. Update the Value Network.
5. Update the Actor.
6. Update both Critics.
7. Periodically update the Target Value Network.

## Requirements

* Python 3.x
* PyTorch
* NumPy
* Gymnasium
* tqdm

## References

* Haarnoja et al., **Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning**, 2018.
* Haarnoja et al., **Soft Actor-Critic Algorithms and Applications**, 2018.

## Purpose

This project was developed to understand and implement the core ideas of **Soft Actor-Critic from scratch**, rather than relying on a pre-built reinforcement learning library.
