import mujoco
import glfw
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F 
import matplotlib.patches as patches

# Backbone network for shared feature extraction
class BackboneNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(BackboneNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return x

# Actor Head Network with Gaussian Outputs
class ActorHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(ActorHead, self).__init__()
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_sigma = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent):
        mu = torch.tanh(self.fc_mu(latent))  # Mean output bounded by [-1, 1]
        sigma = F.softplus(self.fc_sigma(latent)) * 50 # Ensure sigma is positive
        return mu, sigma

# Critic Head Network
class CriticHead(nn.Module):
    def __init__(self, hidden_dim):
        super(CriticHead, self).__init__()
        self.fc_value = nn.Linear(hidden_dim, 1)

    def forward(self, latent):
        value = self.fc_value(latent)
        return value

# Combined Actor-Critic Model with Shared Backbone
class ActorCriticModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, action_dim):
        super(ActorCriticModel, self).__init__()
        self.backbone = BackboneNetwork(input_dim, hidden_dim1, hidden_dim2)
        self.actor_head = ActorHead(hidden_dim2, action_dim)
        self.critic_head = CriticHead(hidden_dim2)

    def forward(self, state):
        latent = self.backbone(state)
        mu, sigma = self.actor_head(latent)
        value = self.critic_head(latent)
        return mu, sigma, value

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

def reward_function(state, goal_position, epsilon=0.1):
    distance_to_goal = np.linalg.norm(state[:2] - goal_position[:2])
    return 1 if distance_to_goal <= epsilon else 0

def actor_critic_training(start_pos, goal_position, model, walls, outside_walls, gamma=0.99, alpha=.01, max_steps=3000, 
                          num_episodes=100, log=True, plot=True, render_enabled=False, mujoco_model=None, mujoco_data=None, fixed=False):
    
    optimizer = optim.Adam(model.parameters(), lr=alpha)
    episode_rewards = []
    starting_points = []

    # Initialize rendering if enabled
    if render_enabled:
        window, camera, scene, context, options, viewport = init_glfw_window(mujoco_model, walls)

    # Initialize epsilon values
    epsilon_start = 0.25
    epsilon_end = 0.1

    for episode in range(num_episodes if not render_enabled else 1):  # Only run one episode if rendering
        
        # Update epsilon based on episode progress
        if not fixed:
            progress_ratio = episode / max(num_episodes, 1)
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * progress_ratio
            start_pos = random_position(walls, outside_walls)
            print(epsilon)
        else:
            epsilon = epsilon_end

        # Start each episode from a random position
        mujoco_data.qpos[0:2] = start_pos[:2]
        mujoco_data.qvel[0:2] = [0, 0]
        total_reward = 0

        for t in range(max_steps):
            
            # Get the current state from MuJoCo
            ball_pos = mujoco_data.qpos[0:2]
            ball_vel = mujoco_data.qvel[0:2]
            state = np.hstack((ball_pos, ball_vel))
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # Calculate reward
            reward = reward_function(state, goal_position, epsilon)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            total_reward += reward

            # Get action from the model
            mu, sigma, value = model(state_tensor)
            action = sample_action(mu, sigma).numpy()[0]

            # Apply forces in MuJoCo
            mujoco_data.xfrc_applied[mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "ball"), :2] = action[:2]
            mujoco.mj_step(mujoco_model, mujoco_data)
            mujoco_data.xfrc_applied[mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "ball"), :2] = [0, 0]

            # Get next state
            next_ball_pos = mujoco_data.qpos[0:2]
            next_ball_vel = mujoco_data.qvel[0:2]
            next_state = np.hstack((next_ball_pos, next_ball_vel))
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            next_value = model(next_state_tensor)[2].detach()

            # Calculate TD error and losses
            td_error = reward_tensor + gamma * next_value - value
            critic_loss = td_error.pow(2).mean()
            actor_loss = -td_error.detach() * torch.distributions.Normal(mu, sigma).log_prob(
                torch.tensor(action)
            ).sum()

            # Backpropagate
            loss = actor_loss + critic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if log:
                print(f"Episode {episode}, Step {t}")
                print(f"  - TD Error: {td_error.item()}")
                print(f"  - Critic Loss: {critic_loss.item()}, Actor Loss: {actor_loss.item()}")
                print(f"  - Action Mean: {mu}, Action Std: {sigma}")
                print(f"  - Reward: {reward}, Total Reward: {total_reward}")

            # Render the scene if enabled
            if render_enabled:
                render_scene(
                    mujoco_model,
                    mujoco_data,
                    options,
                    scene,
                    context,
                    viewport,
                    camera,
                    window,
                )

            # Stop if goal is reached
            if reward == 1:
                break

        avg_reward = total_reward / (t + 1)
        episode_rewards.append(avg_reward)
        starting_points.append(start_pos)

    # Clean up rendering if enabled
    if render_enabled:
        glfw.terminate()

    if plot:
        plt.figure()
        plt.plot(episode_rewards, label="Average Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.title("Learning Curve")
        plt.legend()
        plt.show()
        plot_starting_points(starting_points, goal_position, outside_walls, walls)

    return model

def random_position(walls, outside_walls, margin=0.05):
   
    # Box boundaries
    x_min = -0.15 + margin  # Right edge of gcase_a
    x_max = 1.05 - margin   # Left edge of gcase_b
    y_min = -0.32 + margin  # Top edge of gcase_d
    y_max = 0.32 - margin   # Bottom edge of gcase_c

    while True:
        # Generate a random position within the box
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        vx, vy = 0.0, 0.0  # Velocity is initialized to zero

        # Check if the position is inside wall_3
        if not is_inside_wall(x, y):
            return np.array([x, y, vx, vy])

def is_inside_wall(x, y):
    
    # Wall 3 dimensions from the XML
    wall_3_x_min = 0.5 - 0.1  # Center x - half width
    wall_3_x_max = 0.5 + 0.1  # Center x + half width
    wall_3_y_min = -0.15  # Center y - half height
    wall_3_y_max = 0.15   # Center y + half height

    # Check if the point is inside the wall bounds
    return wall_3_x_min <= x <= wall_3_x_max and wall_3_y_min <= y <= wall_3_y_max

def render_scene(model, data, options, scene, context, viewport, camera, window):
    
    mujoco.mjv_updateScene(model, data, options, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    glfw.poll_events()
    glfw.swap_buffers(window)
    
    return

def init_glfw_window(model, walls):
    
    if not glfw.init():
        raise Exception("Could not initialize glfw")

    window = glfw.create_window(1200, 900, "MuJoCo Simulation - Overhead View", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Could not create glfw window")

    glfw.make_context_current(window)

    # Initialize MuJoCo visualization components
    camera = mujoco.MjvCamera()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    options = mujoco.MjvOption()

    # Calculate the center of the walls/map
    x_coords = [point[0] for wall in walls.values() for point in wall]
    y_coords = [point[1] for wall in walls.values() for point in wall]
    map_center = [sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)]

    # Set up the camera for an overhead view
    camera.lookat = np.array([map_center[0], map_center[1], 0.0])  # Center in x, y, and z=0
    camera.distance = 2.5  # Adjust the distance for a better overhead view
    camera.elevation = -90.0  # Directly overhead
    camera.azimuth = 0.0  # No rotation around the map

    viewport = mujoco.MjrRect(0, 0, 1200, 900)

    return window, camera, scene, context, options, viewport

def plot_starting_points(starting_positions, goal_position, outside_walls, walls):
    
    plt.figure()

    # Plot starting points
    starting_positions = np.array(starting_positions)
    plt.scatter(starting_positions[:, 0], starting_positions[:, 1], color='blue')
    plt.scatter(goal_position[0], goal_position[1], color='red', marker='*', s=100)
    
    # Plot outside walls
    for wall in outside_walls:
        x_values = [wall[0][0], wall[1][0]]
        y_values = [wall[0][1], wall[1][1]]
        plt.plot(x_values, y_values, color="black")

    # Plot inner walls from walls dictionary
    for name, vertices in walls.items():
        vertices = np.array(vertices)
        polygon = patches.Polygon(vertices, closed=True, color="grey", alpha=0.5)
        plt.gca().add_patch(polygon)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Starting Points on 2D Map")
    plt.show()

    return

def main():
    
    # Define the goal position and walls
    goal_position = np.array([1.0, 0.0])
    outside_walls = [
        [[-0.5, -0.4], [-0.5, 0.4]],
        [[1.5, -0.4], [1.5, 0.4]],
        [[-0.5, 0.4], [1.5, 0.4]],
        [[-0.5, -0.4], [1.5, -0.4]],
    ]
    walls = {"wall_3": [[0.5, -0.15], [0.5, 0.15], [0.6, 0.15], [0.6, -0.15]]}

    # Load the MuJoCo model for training
    mujoco_model = mujoco.MjModel.from_xml_path("ball_square.xml")
    data = mujoco.MjData(mujoco_model)
    model = ActorCriticModel(4, 128, 128, 2)

    while True:
        print("\nMenu:")
        print("1. Render Fixed")
        print("2. Render One Training Episode")
        print("3. Train the Network")
        print("4. Quit")

        choice = input("Enter your choice: ")
        if choice.isdigit():
            choice = int(choice)
        else:
            print("Invalid input. Please enter a number.")
            continue

        if choice == 1:
            print("Rendering one training episode with fixed starting point and epsilon...")
            model = actor_critic_training(
                start_pos=np.array([0, 0, 0, 0]),
                goal_position=goal_position,
                model=model,
                walls=walls,
                outside_walls=outside_walls,
                log=False,
                plot=True,
                render_enabled=True,
                mujoco_model=mujoco_model,
                mujoco_data=data,
                fixed=True)
            glfw.terminate()
        if choice == 2:
            print("Rendering one training episode...")
            model = actor_critic_training(
                start_pos=random_position(walls, outside_walls),
                goal_position=goal_position,
                model=model,
                walls=walls,
                outside_walls=outside_walls,
                log=False,
                plot=True,
                render_enabled=True,
                mujoco_model=mujoco_model,
                mujoco_data=data)
            glfw.terminate()
        if choice == 3:
            model = actor_critic_training(
                start_pos=random_position(walls, outside_walls),
                goal_position=goal_position,
                model=model,
                walls=walls,
                outside_walls=outside_walls,
                log=True,
                plot=True,
                render_enabled=False,  # No rendering during initial training
                mujoco_model=mujoco_model,
                mujoco_data=data)
        elif choice == 4:
            print("Exiting the program. Goodbye!")
            break

if __name__ == "__main__":
    main()
