import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mujoco
import glfw
import os
import torch
from datetime import datetime

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc_mu = nn.Linear(hidden_dim3, output_dim)
        self.fc_sigma = nn.Linear(hidden_dim3, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc_mu, self.fc_sigma]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mu = torch.tanh(self.fc_mu(x))  
        sigma = F.softplus(self.fc_sigma(x)) + 1e-5
        return mu, sigma

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc_value = nn.Linear(hidden_dim3, 1)
        self.initialize_weights()

    def initialize_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc_value]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        value = self.fc_value(x)
        return value

class ActorCriticModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, action_dim):
        super(ActorCriticModel, self).__init__()
        self.actor = ActorNetwork(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, action_dim)
        self.critic = CriticNetwork(input_dim, hidden_dim1, hidden_dim2, hidden_dim3)

    def forward(self, state):
        mu, sigma = self.actor(state)
        value = self.critic(state)
        return mu, sigma, value

def sample_action(mu, sigma):
    dist = torch.distributions.Normal(mu, sigma)
    action = dist.sample()
    return action.clamp(-1, 1), dist

def actor_critic_training(
    model,
    goal_position,
    walls,
    outside_walls,
    actor_optimizer,
    critic_optimizer,
    num_episodes=1000,
    max_steps=100,
    gamma=0.99,
    log_interval=1,
    render=False,
    mujoco_model=None,
    mujoco_data=None,
    plot=False,
    epsilon=0.2,
    fixed=False,
    log=True):
    
    starting_positions = []
    episode_rewards = []
    actor_losses = []
    critic_losses = []

    if render:
        window, camera, scene, context, viewport, option = init_mujoco_render(mujoco_model)

    goal_state = np.array([goal_position[0], goal_position[1], 0.0, 0.0])
    
    for episode in range(num_episodes):
        
        if fixed:
            state = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            state = random_position(walls, outside_walls, goal_position)
            
        starting_positions.append(state[:2])
        if mujoco_model is not None and mujoco_data is not None:
            mujoco_data.qpos[0:2] = state[0:2]
            mujoco_data.qvel[0:2] = state[2:4]
            mujoco.mj_forward(mujoco_model, mujoco_data)

        total_reward = 0
        
        for t in range(max_steps):

            # forward pass to get action and value
            critic_optimizer.zero_grad()
            actor_optimizer.zero_grad()
            
            state_tensor = torch.tensor(state, dtype=torch.float32)
            mu, sigma, value = model(state_tensor)
            action, dist = sample_action(mu, sigma)
            log_prob = dist.log_prob(action).sum()

            # apply action in mujoco to the ball
            if mujoco_model is not None and mujoco_data is not None:
                ball_body_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "ball")
                mujoco_data.xfrc_applied[ball_body_id, :2] = action.detach().numpy()
                mujoco.mj_step(mujoco_model, mujoco_data)
                next_state = np.hstack((mujoco_data.qpos[0:2], mujoco_data.qvel[0:2]))
            else:
                # optional use the custom transition function
                next_state = transition(state, action.detach().numpy())

            # calculate reward
            reward = calculate_reward(state, goal_position, epsilon, t)

            # update total reward
            total_reward += reward
            done = np.linalg.norm(next_state[:2] - goal_position) <= epsilon
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            # compute value of next state
            with torch.no_grad():
                next_value = model.critic(next_state_tensor)

            td_target = reward + gamma * next_value
            advantage = td_target - value
            critic_loss = F.mse_loss(value, td_target.detach())
            actor_loss = -(advantage.detach() * log_prob).mean()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item()) 
            
            # backpropagation
            critic_loss.backward()
            actor_loss.backward()

            # optimizer steps
            critic_optimizer.step()
            actor_optimizer.step()

            if log:
                print(f"Episode {episode}, Step {t}")
                print(f"  - TD Error: {advantage.item()}")
                print(f"  - Critic Loss: {critic_loss.item()}, Actor Loss: {actor_loss.item()}")
                print(f"  - Action Mean: {mu}, Action Std: {sigma}")
                print(f"  - Action Taken: {action}")

            if render and mujoco_model is not None and mujoco_data is not None:
                render_mujoco_scene(mujoco_model, mujoco_data, scene, context, viewport, camera, window, option)

            if done:
                break
            
            state = next_state

        # store the total reward for the episode
        episode_rewards.append(total_reward/t if t > 0 else 0)
        if episode % log_interval == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    if render:
        glfw.terminate()

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards, label="Avg. Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Avg. Reward")
        plt.title("Learning Curve")
        plt.legend()
        plt.show()
        plot_starting_points(starting_positions, goal_position, outside_walls, walls)

        # Plot the losses
        plt.figure(figsize=(12, 6))

        # Plot both losses on the same chart
        plt.plot(critic_losses, label="Critic Loss", alpha=0.7)
        plt.plot(actor_losses, label="Actor Loss", alpha=0.7)
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Critic and Actor Loss Over Training")
        plt.legend()
        plt.show()

    return model

def calculate_reward(state, goal_position, epsilon, t):
    
    distance_to_goal = np.linalg.norm(state[:2] - goal_position)
    reward = -distance_to_goal * 2
    if distance_to_goal <= epsilon:
        reward += 5000
    reward -= .01 * t

    return reward

def transition(state, action, dt=0.01):
    
    x, y, vx, vy = state
    fx, fy = action
    noise_x, noise_y = np.random.normal(0, 0.1, 2)
    new_vx = vx + (fx - noise_x) * dt
    new_vy = vy + (fy - noise_y) * dt
    new_x = x + new_vx * dt
    new_y = y + new_vy * dt

    return np.array([new_x, new_y, new_vx, new_vy])

def plot_starting_points(starting_positions, goal_position, outside_walls, walls):

    plt.figure()
    starting_positions = np.array(starting_positions)
    plt.scatter(starting_positions[:, 0], starting_positions[:, 1], color='blue', label='Starting Positions')
    plt.scatter(goal_position[0], goal_position[1], color='red', marker='*', s=100, label='Goal Position')
    
    for wall in outside_walls:
        x_values = [wall[0][0], wall[1][0]]
        y_values = [wall[0][1], wall[1][1]]
        plt.plot(x_values, y_values, color="black")
    
    for wall in walls:
        x_min = wall['x_min']
        x_max = wall['x_max']
        y_min = wall['y_min']
        y_max = wall['y_max']
        rectangle = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, color='grey', alpha=0.5)
        plt.gca().add_patch(rectangle)

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Starting Points on 2D Map")
    plt.legend()
    plt.show()

    return

def random_position(walls, outside_walls, goal_position):
    
    x_min = -0.2 
    x_max = 1.1  
    y_min = -0.36
    y_max = 0.36 

    while True:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        vx, vy = 0.0, 0.0

        position = np.array([x, y])
        if not is_inside_wall(x, y):
            return np.array([x, y, vx, vy])

def is_inside_wall(x, y):
    
    safety_margin = 0.2

    case_x_min = -0.2 + safety_margin
    case_x_max = 1.1 - safety_margin
    case_y_min = -0.36 + safety_margin
    case_y_max = 0.36 - safety_margin

    if not (case_x_min <= x <= case_x_max and case_y_min <= y <= case_y_max):
        return True

    wall_3_x_min = 0.5 - safety_margin
    wall_3_x_max = 0.5 + 0.1 + safety_margin
    wall_3_y_min = -0.15 - safety_margin
    wall_3_y_max = 0.15 + safety_margin

    if wall_3_x_min <= x <= wall_3_x_max and wall_3_y_min <= y <= wall_3_y_max:
        return True

    return False

def init_mujoco_render(model):
    if not glfw.init():
        raise Exception("Could not initialize GLFW")

    window = glfw.create_window(800, 600, "MuJoCo Simulation", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Could not create GLFW window")

    glfw.make_context_current(window)
    camera = mujoco.MjvCamera()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    viewport = mujoco.MjrRect(0, 0, 800, 600)
    option = mujoco.MjvOption()

    camera.azimuth = 0
    camera.elevation = -90
    camera.distance = 2.0
    camera.lookat = np.array([0.5, 0.0, 0.0])

    return window, camera, scene, context, viewport, option

def render_mujoco_scene(model, data, scene, context, viewport, camera, window, option):
    mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()
    return

def save_model(model, total_episodes, save_dir="../model"):
    os.makedirs(save_dir, exist_ok=True)      
    save_path = os.path.join(save_dir, f"actor_critic_model_{total_episodes}_episodes.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return

def main():
    
    goal_position = np.array([0.8, 0.0])  
    epsilon = 0.2
    total_episodes = 0

    walls = [
        {"x_min": 0.5, "x_max": 0.6, "y_min": -0.15, "y_max": 0.15}
    ]
    outside_walls = [
        [[-0.5, -0.4], [-0.5, 0.4]],
        [[1.5, -0.4], [1.5, 0.4]],
        [[-0.5, 0.4], [1.5, 0.4]],
        [[-0.5, -0.4], [1.5, -0.4]],]

    mujoco_model = mujoco.MjModel.from_xml_path("ball_square.xml")
    mujoco_model.opt.timestep = 0.1
    mujoco_data = mujoco.MjData(mujoco_model)
    model = ActorCriticModel(input_dim=4, hidden_dim1=256, hidden_dim2=128, hidden_dim3=64, action_dim=2)
    
    # optimize the parameters
    actor_optimizer = optim.Adam(model.actor.parameters(), lr=1e-5)
    critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-5)
   
    while True:
        print("\nMenu:")
        print("1. Render one training episode")
        print("2. Train the network")
        print("3. Render with fixed starting position")
        print("4. Quit")

        choice = input("Enter your choice: ")
        if choice.isdigit():
            choice = int(choice)
        else:
            print("Invalid input. Please enter a number.")
            continue

        if choice == 1:
            print("Rendering one training episode...")
            num_training_episodes = 1
            actor_critic_training(
                model=model,
                goal_position=goal_position,
                walls=walls,
                outside_walls=outside_walls,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                num_episodes=num_training_episodes,
                max_steps=1800,
                gamma=0.99,
                log_interval=1,
                render=True,
                mujoco_model=mujoco_model,
                mujoco_data=mujoco_data,
                epsilon=epsilon,
                fixed=False
            )
            total_episodes += num_training_episodes
        elif choice == 2:
            print("Training the network...")
            num_training_episodes = 100
            model = actor_critic_training(
                model=model,
                goal_position=goal_position,
                walls=walls,
                outside_walls=outside_walls,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                num_episodes=num_training_episodes,
                mujoco_model=mujoco_model,
                mujoco_data=mujoco_data,
                max_steps=1800,
                gamma=0.99,
                log_interval=1,
                render=True,
                plot=True,
                epsilon=epsilon
            )
            total_episodes += num_training_episodes
        elif choice == 3:
            print("Rendering with fixed starting position...")
            num_training_episodes = 1
            actor_critic_training(
                model=model,
                goal_position=goal_position,
                walls=walls,
                outside_walls=outside_walls,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
                num_episodes=num_training_episodes,
                max_steps=1800,
                gamma=0.99,
                log_interval=1,
                render=True,
                mujoco_model=mujoco_model,
                mujoco_data=mujoco_data,
                epsilon=epsilon,
                fixed=True
            )
            total_episodes += num_training_episodes
        elif choice == 4:
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select from the menu.")

if __name__ == "__main__":
    main()
