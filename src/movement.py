import mujoco
import glfw
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F 

# Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        value = self.fc2(x)
        return value

# Actor Network with Gaussian Outputs
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_sigma = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        mu = torch.tanh(self.fc_mu(x))  # Mean output bounded by [-1, 1]
        sigma = F.softplus(self.fc_sigma(x)) + 1e-5  # Ensure sigma is positive
        return mu, sigma

def sample_action(mu, sigma):
    # Sample from the Gaussian distribution defined by mu and sigma
    dist = torch.distributions.Normal(mu, sigma)
    action = dist.sample()
    return action.clamp(-1, 1)  # Clamp action to force limits [-1, 1]

# State Transition Function for the MDP
def transition(state, action, dt=0.1):
    
    # Extract current position and velocity
    x, y, vx, vy = state
    fx, fy = action

    # Sample noise from a normal distribution for the dynamics
    noise_x, noise_y = np.random.normal(0, 0.1, 2)

    # Apply MDP dynamics with noise
    new_vx = vx + (fx - noise_x) * dt
    new_vy = vy + (fy - noise_y) * dt
    new_x = x + new_vx * dt
    new_y = y + new_vy * dt

    # Return the new state
    return np.array([new_x, new_y, new_vx, new_vy])

def reward_function(state, goal_position, epsilon=0.10):
    distance_to_goal = np.linalg.norm(state[:2] - goal_position[:2])
    print(distance_to_goal)
    return 1 if distance_to_goal <= epsilon else 0

def actor_critic_training(start_pos, goal_position, actor, critic, walls, gamma=0.99, alpha=1e-5, beta=1e-5, max_steps=30, num_episodes=1500, log=True, plot=True):
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=alpha)
    critic_optimizer = optim.Adam(critic.parameters(), lr=beta)
    episode_rewards = []

    for episode in range(num_episodes):
        
        # Start each episode from a random position
        start_pos = random_position(walls)
        state = start_pos
        total_reward = 0

        for t in range(max_steps):
            
            # Convert the state to a tensor for model input
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            reward = reward_function(state, goal_position)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            total_reward += reward

            # Get action from Actor and value from Critic
            mu, sigma = actor(state_tensor)
            action = sample_action(mu, sigma).numpy()[0]
            value = critic(state_tensor)

            # Compute next state
            next_state = transition(state, action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            next_value = critic(next_state_tensor).detach()

            # Calculate TD error
            td_error = reward_tensor + gamma * next_value - value

            # Update Critic
            critic_loss = td_error.pow(2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Update Actor
            actor_loss = -td_error.detach() * torch.distributions.Normal(mu, sigma).log_prob(torch.tensor(action)).sum()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Logging
            if log:
                print(f"Episode {episode}, Step {t}")
                print(f"  - TD Error: {td_error.item()}")
                print(f"  - Critic Loss: {critic_loss.item()}, Actor Loss: {actor_loss.item()}")
                print(f"  - Action Mean: {mu}, Action Std: {sigma}")
                print(f"  - Action Taken: {action}")
                print(f"  - Reward: {reward}, Total Reward: {total_reward}")

            # Move to the next state
            state = next_state

            # Stop if goal is reached
            if reward == 1:
                break

        # Calculate average reward per step in this episode
        avg_reward = total_reward / (t + 1)
        episode_rewards.append(avg_reward)

    # Plot learning curve after training
    if plot:
        plt.figure()
        plt.plot(episode_rewards, label="Average Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.title("Learning Curve")
        plt.legend()
        plt.show()

    return actor, critic

def move_ball(model, data, actor, window, scene, context, options, viewport, camera, goal_position, render_enabled=True, Tmax=120):
    
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    dt = 0.1
    time_elapsed = 0

    # Separate the start position into position and velocity
    start_pos = np.array([0, 0, 0, 0]) 
    position = start_pos[:2]  # [x, y]
    velocity = start_pos[2:]  # [vx, vy]

    # Set the initial position and velocity in MuJoCo
    data.qpos[0:2] = position
    data.qvel[0:2] = velocity

    while time_elapsed < Tmax:
        
        # Get current position and velocity of the ball
        ball_pos = data.qpos[0:2]
        ball_vel = data.qvel[0:2]
        state = np.hstack((ball_pos, ball_vel))
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Get action parameters (mu and sigma) from Actor network
        with torch.no_grad():
            mu, sigma = actor(state_tensor)
            action = sample_action(mu, sigma).numpy()[0]

        print(f"Time: {time_elapsed:.2f}s, Position: {ball_pos}, Velocity: {ball_vel}, Action: {action}")        

        # Apply transition dynamics in simulation
        next_state = transition(state, action)
        data.qpos[0:2] = next_state[:2]  # Update position [x, y]
        data.qvel[0:2] = next_state[2:]  # Update velocity [vx, vy]

        mujoco.mj_step(model, data)

        # Check if goal is reached
        distance_to_goal = np.linalg.norm(goal_position[:2] - ball_pos)
        if distance_to_goal < 0.1:
            print("Goal reached.")
            break

        # Render the scene if enabled
        if render_enabled:
            render_scene(model, data, options, scene, context, viewport, camera, window)

        time_elapsed += dt

def random_position():
    # Generate random starting position and set initial velocity to zero
    x, y = random.uniform(-1, 1), random.uniform(-1, 1)
    vx, vy = 0.0, 0.0  # Initial velocity is zero
    return np.array([x, y, vx, vy])

def is_inside_wall(x, y, walls):
    for wall in walls.values():
        x1, y1 = wall[0]
        x2, y2 = wall[2]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

def random_position(walls):
    while True:
        x = random.uniform(-0.5, 1.5)  # x-bounds of the map
        y = random.uniform(-0.4, 0.4)  # y-bounds of the map
        vx, vy = 0.0, 0.0  # Initial velocity is zero
        
        # Check if the generated position is inside any wall
        if not is_inside_wall(x, y, walls):
            return np.array([x, y, vx, vy])

def render_scene(model, data, options, scene, context, viewport, camera, window):
    
    mujoco.mjv_updateScene(model, data, options, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    glfw.poll_events()
    glfw.swap_buffers(window)
    
    return

def init_glfw_window(model):
    
    if not glfw.init():
        raise Exception("Could not initialize glfw")

    window = glfw.create_window(1200, 900, "MuJoCo Simulation", None, None)
    
    if not window:
        glfw.terminate()
        raise Exception("Could not create glfw window")

    glfw.make_context_current(window)
    
    camera = mujoco.MjvCamera()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    options = mujoco.MjvOption()
    
    camera.distance = 3.0
    camera.elevation = -90.0
    camera.azimuth = 0.0

    viewport = mujoco.MjrRect(0, 0, 1200, 900)
    
    return window, camera, scene, context, options, viewport

def model_creation(actor, goal_position):
    
    model = mujoco.MjModel.from_xml_path("ball_square.xml")
    data = mujoco.MjData(model)
    
    window, camera, scene, context, options, viewport = init_glfw_window(model)
    move_ball(model, data, actor, window, scene, context, options, viewport, camera, goal_position)
    glfw.terminate()
    
    return

def main():
    
    goal_position = np.array([1.0, 0.0])
    actor, critic = None, None 
    outside_walls = [
    [[-0.5, -0.4], [-0.5, 0.4]],
    [[1.5, -0.4], [1.5, 0.4]],
    [[-0.5, 0.4], [1.5, 0.4]],
    [[-0.5, -0.4], [1.5, -0.4]]]
    walls = {"wall_3": [[0.5, -0.15], [0.5, 0.15], [0.6, 0.15], [0.6, -0.15]]}
    start_pos = random_position(walls) # Initialize Actor and Critic as None to check if they are trained
    
    while True:
        print("\nMenu:")
        print("1. Run Model Simulation")
        print("2. Train the Networks")
        print("3. Quit")

        choice = input("Enter your choice: ")
        if choice.isdigit():
            choice = int(choice)
        else:
            print("Invalid input. Please enter a number.")
            continue

        if choice == 1:
            if actor is not None and critic is not None:
                model_creation(actor, goal_position)
            else:
                print("Please train the networks first by selecting option 2.")
        elif choice == 2:
            actor = ActorNetwork(4, 128, 2)
            critic = CriticNetwork(4, 128)
            actor, critic = actor_critic_training(start_pos, goal_position, actor, critic, walls)
            print("Training completed.")
        elif choice == 3:
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
