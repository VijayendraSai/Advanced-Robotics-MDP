# Actor-Critic Reinforcement Learning for Noisy Robot Navigation

## 🧭 Overview

This project revisits the 2D navigation problem with a formalized Markov Decision Process (MDP), modeling force-based control under stochastic dynamics. A custom actor-critic algorithm is implemented using neural networks to learn optimal policies in the MuJoCo simulator.

---

## 🔍 MDP Formulation

### ➤ Action Space
Each action `a_t = (fx, fy)` represents force applied in the x and y directions:
- `fx, fy ∈ [−1, 1]` Newton

### ➤ State Space
Each state `s_t = (x, y, ẋ, ẏ)` includes position and velocity in 2D.

- Initial positions are uniformly sampled from the workspace.
- Initial velocities are set to `0`.

### ➤ Transition Dynamics

Given:
- Time step `Δt = 0.1s`
- Robot mass = 1 kg
- Gaussian noise `ρx, ρy ∼ N(0, 0.1)`

The dynamics evolve as:
ẋ_{t+1} = ẋ_t + (fx_t − ρx_t) * Δt
ẏ_{t+1} = ẏ_t + (fy_t − ρy_t) * Δt
x_{t+1} = x_t + ẋ_t * Δt
y_{t+1} = y_t + ẏ_t * Δt


### ➤ Reward Function

- `R(s_t) = 1` if `||s_t − s_g|| ≤ ε`
- `R(s_t) = 0` otherwise  
Where goal state `s_g = (x_g, y_g, 0, 0)` is fixed, and `ε` is a chosen threshold.

### ➤ Discount Factor

- `γ = 0.99`

---

## 🤖 Actor-Critic Algorithm

The agent is trained using a basic actor-critic framework, based on the lecture "Policy Gradients and Actor Critics" (Slide 40).

### Components:

- **Actor Network:** Outputs Gaussian-distributed actions `a_t ~ π_θ(s_t)`
- **Critic Network:** Predicts value estimates `V_w(s_t)` for the current policy
- **Loss Functions:** 
  - Critic: MSE between predicted and actual returns
  - Actor: Policy gradient with advantage estimation

---

## ⚙️ Implementation Steps

1. **MuJoCo Setup**  
   - Reused environment setup from HW1  
   - Modified XML and Python simulator to match MDP

2. **Neural Networks**  
   - Simple feedforward architectures for actor and critic  
   - Implemented with PyTorch

3. **Training Loop**  
   - User-defined episode horizon  
   - Batch training with gradient steps  
   - Gaussian policy sampling

4. **Evaluation**  
   - Learning curve: average reward per step vs episodes

---

## 📈 Results

The learning curve below shows the average reward per time step as training progresses. The agent gradually learns to reach the goal more consistently under stochastic dynamics.
