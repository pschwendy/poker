# poker

This repository contains the code for training a deep reinforcement learning poker bot through self-play. The general algorithm generally follows Deep Counterfactual Regret Minimization [(Brown et. al, 2019)](https://arxiv.org/abs/1811.00164). This project, however, mainly serves as a proof of concept and playground for new ideas, so we take a few liberations from the original algorithm to save memory and compute time at the cost of performance (at least for now, more on this below). Currently, a value network (~50K parameters) trained under this scheme can achieve an exploitability in the range of 0-250 mbb/game, achieved at no cost ($0 spent).


## Deep Counterfactual Regret Minimization

Deep Counterfactual Regret Minimization is an equilibrium-finding algorithm for imperfect information games (specifically poker) that uses a deep neural network to approximate a Nash equilibrium [(Brown et al., 2019)](https://arxiv.org/abs/1811.00164). In the original algorithm, two neural networks compete as players in a heads-up poker game. Over **T** iterations, each network performs **K** traversals through poker games. 

During each traversal, the goal is to collect action advantages **r_t(h)** for given states **h** into an ever-growing advantage set **M_Vp** (corresponding to player **p**). In between traversals, the parameters **θ_p** are trained from scratch using the collected datasets **M_Vp**. Both players also contribute to a global policy dataset **M_Pi**, which contains mappings from states to policies \(**h** -> **σ(h)**). Further details, including pseudocode, can be found in the original paper.

To save on memory and time, we take the following liberties at the cost of perfomance:
- Train a single model with parameters θ to play against itself
- Only collect M_V, not M_Pi
- Utilize the most recently trained value network as our evaluation bot (Note: in theory, randomly sampling across all trained for each move value networks is the same as training on M_Pi. This shouldn't be too difficult to implement, but we haven't gotten to it yet) 

Our current variation of the deep cfr algorithm is as follows (in pythonic pseudocode):
```python
# Algorithm 1: Deep CFR
def deepcfr():
  init V_θ # with parameters θ such that they output 0
  init dataset M_V
  for CFR iteration t = 1 -> T:
    for traversal k = 1 -> K:
      init empty state h
      traverse(h,0,θ,M_V, t) # traverse through player 0 to collect advantage set
    Reinit V_θ
    Optimize E[t * MSE(r_t(h) - V_θ(h))] with (h, t, r_t(h)) ~ M_V
  # Use final value net V_θ from timestep T as poker bot
```

```python
# Algorithm 2: CFR Traversal
def traverse(h,p,θ,M_V, t):
  if h is terminal:
    return u(p) # utility of player p
  elif player at h != p:
    values = V_θ(h) 
    σ(h) = R_+(values) # compute policy distribution using regret matching
    action ~ σ(h) # sample action from policy distribution
    h <- action # update state/history with action
    return traverse(h,(p + 1) % 2,θ,M_Vp, t)
  elif player at h == p:
    for each possible action a:
      h' <- a
      u[a] = traverse(h',(p + 1) % 2,θ,M_Vp, t) # traverse to find estimated utiltiy of taking action a
    values = V_θ(h) 
    σ(h) = R_+(values) # compute policy distribution using regret matching
    r_t(h) = u[a] - sum(σ(h) * u[a])
    M_V.add(h, r_t(h), t) # add state, action values, and timestep t to value dataset M_V
    return sum(σ(h) * u[a])
```

Note: in our code, chance nodes (i.e. dealing cards) are handled by the state class itself (not inside the traverse function)

## Metrics (calculating exploitability)

Theoretically, exploitability in a two-player zero-sum game (like heads-up poker) is defined as how much worse a strategy σ does versus its corresponding best response strategy BR(σ) compared to how a Nash equilibrium strategy σ∗ does against BR(σ∗). In practice, we can measure this as the sum of utilities of σ vs BR(σ) for each player p. That is u(σ1, BR(σ1)) + u(σ2, BR(σ2)), where u is the given utility function. We do this over a large number of repeated games utilizing the LocalBR algorithm to approximate BR(σ) [(Lisý et. al, 2016)](https://arxiv.org/abs/1612.07547). As in many papers, we measure exploitability in milli big blinds (mbb) / game. A milli big blind is 1/1000th of a big blind, so with a big blind of $100, winning $200 would be equivalent to winning 2000 mbb.

## Dependencies

- pytorch (we use 2.4.1+cu121)
- numpy

## Usage

To train a bot simply run (preferably in a GPU available environment)

```
python train_cfr.py
```

However, in practice, we made use of Kaggle's free GPU hours to train our models. We do this through `pokerv4mini.ipynb`, which mixes all important files in this repository into a single large, experimental notebook.

## Goals
- [ ] Implement variance reduction techniques like AVIAT [(Burch et. al, 2016)](https://arxiv.org/abs/1612.06915) to obtain more accurate exploitability measurements
- [ ] Implement real-time search algorithms such as those seen in Libratus [(Brown et. al, 2017)](https://www.science.org/doi/10.1126/science.aao1733)
- [ ] Try to employ adaptive strategies (END GOAL!)
- [ ] Expand to larger game sizes (such as HUNL)
