# Project Report — Navigation

## Learning Algorithm

This project implements a **Double DQN** agent with a **Dueling network architecture** to solve the Banana environment.

### Double DQN

Standard DQN uses the same target network to both select and evaluate the best next action:

```
Q_target = r + γ * max_a Q_target(s', a)
```

This causes systematic overestimation of Q-values because the `max` operator introduces a positive bias — noise in the value estimates gets amplified.

**Double DQN** (van Hasselt et al., 2016) decouples action selection from evaluation:

```
a* = argmax_a Q_local(s', a)         # local net selects
Q_target = r + γ * Q_target(s', a*)  # target net evaluates
```

The local network selects which action is best, but the target network evaluates how good that action actually is. Since the two networks have different parameters, their estimation errors are less correlated, producing more accurate Q-value estimates and more stable training.

### Dueling Architecture

The standard DQN maps states directly to Q-values for each action. The **Dueling architecture** (Wang et al., 2016) instead decomposes Q-values into two streams:

- **Value stream V(s)**: How good is this state overall?
- **Advantage stream A(s,a)**: How much better is action *a* compared to the average?

These combine as:

```
Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
```

Subtracting the mean advantage ensures identifiability — without it, V(s) and A(s,a) could shift by a constant without changing Q.

**Why this helps in the Banana environment**: Many states have similar values (e.g., no bananas nearby, all actions equally unimportant). The dueling architecture lets the agent learn "this state is uninteresting" through V(s) without needing to evaluate every individual action.

### Experience Replay

The agent stores transitions (s, a, r, s', done) in a replay buffer of size 100,000 and samples random minibatches of 64 for learning. This breaks temporal correlations between consecutive samples and improves data efficiency.

### Soft Target Updates

Instead of periodically copying the full local network to the target network, we use soft updates every 4 timesteps:

```
θ_target = τ * θ_local + (1 - τ) * θ_target
```

with τ = 0.001, providing smooth, stable updates to the target network.

## Network Architecture

```
Input (37) → FC1 (128, ReLU)
  ├── Value stream:     FC (64, ReLU) → V(s) (1)
  └── Advantage stream: FC (64, ReLU) → A(s,a) (4)

Output: Q(s,a) = V(s) + (A(s,a) - mean(A))
```

| Layer | Parameters |
|---|---|
| FC1: 37 → 128 | 4,864 |
| Value FC: 128 → 64 | 8,256 |
| Value Out: 64 → 1 | 65 |
| Advantage FC: 128 → 64 | 8,256 |
| Advantage Out: 64 → 4 | 260 |
| **Total** | **21,701** |

The shared first layer (128 units, wider than the standard 64) accommodates the 37-dimensional state space. The split streams each use 64 units, keeping the total parameter count low (~22K) for fast CPU training.

## Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Replay buffer size | 100,000 | Sufficient diversity for this environment |
| Batch size | 64 | Standard, efficient for CPU |
| Discount factor (γ) | 0.99 | Standard, values future rewards highly |
| Soft update rate (τ) | 0.001 | Slow blending for target stability |
| Learning rate | 5e-4 | Adam optimizer, well-tested for DQN |
| Update frequency | Every 4 steps | Balances learning frequency vs stability |
| Epsilon start | 1.0 | Full exploration initially |
| Epsilon end | 0.01 | Maintains minimal exploration |
| Epsilon decay | 0.995 | Reaches ~0.01 by episode 920 |

## Plot of Rewards

![Training Scores](scores_plot.png)

The plot shows the score per episode (light) and the 100-episode rolling average (dark). The agent first solved the environment (average >= 13.0) at approximately **episode 502**, then continued training to reach a robust average of **15.05** at episode 938.

**Training dynamics**: Scores start near 0 (random policy), begin climbing around episode 100 as the replay buffer fills and epsilon decays. By episode 300, the average reaches ~7.7, and scores accelerate as the agent masters banana discrimination. Performance continues improving past the 13.0 threshold, reaching 15+ by episode 938. In greedy evaluation (100 test episodes with epsilon=0), the agent achieved an average score of **15.20**, confirming robust performance well above the 13.0 solve condition.

## Ideas for Future Work

1. **Prioritized Experience Replay** (Schaul et al., 2016): Sample transitions with high TD error more frequently. This focuses learning on "surprising" experiences and typically accelerates convergence by 2x.

2. **Noisy Networks** (Fortunato et al., 2018): Replace epsilon-greedy exploration with learned noise in the network weights. This provides state-dependent exploration — the agent explores more in unfamiliar states and less in well-understood ones.

3. **Rainbow DQN** (Hessel et al., 2018): Combine all six extensions (Double, Dueling, Prioritized Replay, Multi-step returns, Distributional DQN, Noisy Networks). Rainbow achieves state-of-the-art performance on Atari benchmarks and would likely solve this environment much faster.

4. **Learning from Pixels**: Replace the 37-dimensional state vector with raw pixel observations from the environment. This would require convolutional layers and significantly more training time, but demonstrates a more general approach applicable to real-world visual tasks.
