# Advanced-Robotics-MDP
Let us revisit the same navigation problem that you solved in HW 1, and con-
sider the following MDP:
• Action at = (fx,t,fy,t), where fx,t is the force applied on the robot in the x-dimension, and fy,t is the force applied on the robot in the y-dimension. Both fx,t and fy,t are limited to values in the interval [−1 Newton, 1 Newton].
• State st = (xt,yt,x ̇t,y ̇t) is the position and velocity of the robot. At the beginning of each episode, the initial state is obtained by sampling the position of the robot uniformly in the workspace of the robot, and setting its initial velocity to 0.
• The transition function is defined as follows:
 x ̇ t+1 = y ̇t+1 = xt+1 = yt+1 =
x ̇ t + (fx,t − ρx,t )∆t y ̇t + (fy,t − ρy,t)∆t xt+x ̇t∆t xt+y ̇t∆t
where ρx,t and ρy,t are small independent noises, sampled from N (0, 0.1) at each time-step t, and ∆t is set to 0.1 seconds. We are assuming here that the robot has a mass of 1 Kg. You can imagine the force noises as air resistance or random friction (although air resistance should scale up as a function of velocity).
• The reward function is defined as: R(st) = 1 if ∥st − sg∥2 ≤ ε, and R(st) = 0 otherwise. sg = (xg, yg, 0, 0). It’s up to you to choose the values of the goal coordinates (xg,yg), but they should be fixed in advance. It’s also up to you to choose a reasonable value for the goal threshold ε.
• Set the discount factor γ = 0.99. What you need to do:
1
1. Leverage the Mujoco setup you had in HW 1 and modify it to simulate the MDP described above.
2. Design a small actor neural network that predicts actions at ∼ πθ(st) as Gaussians, and a second small critic network that predicts values vw(st) of policy πθ.
3. Implement the actor-critic algorithm explained in slide 40 of the lecture “Policy Gradients and Actor Critics”. It is up to you to define the length of the episodes (horizon) and the gradient step-sizes.
4. Report the average reward per step as a function of the number of episodes that you used for training. This is called the learning curve.
5. Submit the code and a small writeup on canvas. The writeup includes an explanation of what you did, and the results (the learning curve).
