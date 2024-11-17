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

# Neural network components
class BackboneNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(BackboneNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return x

class ActorHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(ActorHead, self).__init__()
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.log_sigma = nn.Parameter(torch.zeros(output_dim))

    def forward(self, latent):
        mu = torch.tanh(self.fc_mu(latent))
        sigma = torch.exp(self.log_sigma) + 1e-5
        return mu, sigma

class CriticHead(nn.Module):
    def __init__(self, hidden_dim):
        super(CriticHead, self).__init__()
        self.fc_value = nn.Linear(hidden_dim, 1)

    def forward(self, latent):
        value = self.fc_value(latent)
        return value

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
    dist = torch.distributions.Normal(mu, sigma)
    action = dist.sample()
    return action.clamp(-1, 1), dist

def actor_critic_training(
    model,
    goal_position,
    walls,
    outside_walls,
    num_episodes=1000,
    max_steps=100,
    gamma=0.99,
    alpha=.05,  # Critic learning rate
    beta=.05,   # Actor learning rate
    log_interval=100,
    render=False,
    mujoco_model=None,
    mujoco_data=None,
    plot=False,
    epsilon=0.2,
    fixed=False):

    # Separate optimizers for actor and critic
    actor_params = list(model.actor_head.parameters())
    critic_params = list(model.backbone.parameters()) + list(model.critic_head.parameters())
    
    actor_optimizer = optim.Adam(actor_params, lr=beta)
    critic_optimizer = optim.Adam(critic_params, lr=alpha)

    starting_positions = []
    episode_rewards = []

    if render:
        window, camera, scene, context, viewport, option = init_mujoco_render(mujoco_model)

    # Goal state (xg, yg, 0, 0)
    goal_state = np.array([goal_position[0], goal_position[1], 0.0, 0.0])

    for episode in range(num_episodes):
        
        if fixed:
            # Fixed starting position
            state = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            state = random_position(walls, outside_walls, goal_position, epsilon)
            
        starting_positions.append(state[:2])  # Collect starting positions
        
        if mujoco_model is not None and mujoco_data is not None:
            mujoco_data.qpos[0:2] = state[0:2]
            mujoco_data.qvel[0:2] = state[2:4]
            mujoco.mj_forward(mujoco_model, mujoco_data)  # Ensure the simulation state is updated

        for t in range(max_steps):

            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32)
            state_tensor.requires_grad = True  # Enable gradient computation

            # Forward pass to get action and value
            mu, sigma, value = model(state_tensor)
            action, dist = sample_action(mu, sigma)
            log_prob = dist.log_prob(action).sum()

            # Apply action in MuJoCo
            if mujoco_model is not None and mujoco_data is not None:
                ball_body_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "ball")
                mujoco_data.xfrc_applied[ball_body_id, :2] = action.detach().numpy()
                mujoco.mj_step(mujoco_model, mujoco_data)
                # Get next state from MuJoCo simulation
                next_state = np.hstack((mujoco_data.qpos[0:2], mujoco_data.qvel[0:2]))
            else:
                # If MuJoCo is not available, you can use the custom transition function
                next_state = transition(state, action.detach().numpy())

            # Check if the goal is reached
            if np.linalg.norm(next_state - goal_state) <= epsilon:
                reward = 1.0  # Accumulate total reward
                done = True
            else:
                reward = 0.0
                done = False

            # Convert next_state to tensor
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            # Compute value of next state
            with torch.no_grad():
                next_value = model.critic_head(model.backbone(next_state_tensor))

            # Compute TD target
            td_target = reward + gamma * next_value 

            # Compute advantage by subtracting the baseline (value estimate)
            advantage = td_target - value

            # Compute losses
            critic_loss = advantage.pow(2)
            actor_loss = -advantage.detach() * log_prob

            # Backward passes
            critic_loss.backward(retain_graph=True)
            actor_loss.backward()

            # Optimizer steps
            critic_optimizer.step()
            actor_optimizer.step()
            
            if render and mujoco_model is not None and mujoco_data is not None:
                render_mujoco_scene(mujoco_model, mujoco_data, scene, context, viewport, camera, window, option)

            if done:
                episode_rewards.append(reward)
                break
            
            state = next_state

        # Append the total reward for the episode
        critic_optimizer.zero_grad()
        actor_optimizer.zero_grad()

    if render:
        glfw.terminate()
    
    print(episode_rewards)
    
    if plot:
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward per Episode (Scaled by Steps)")
        plt.title("Training Progress")
        plt.show()
        plot_starting_points(starting_positions, goal_position, outside_walls, walls)

    return model

def transition(state, action, dt=0.01):
    
    # This function is kept for situations where MuJoCo is not used
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

    # Plot starting points
    starting_positions = np.array(starting_positions)
    plt.scatter(starting_positions[:, 0], starting_positions[:, 1], color='blue', label='Starting Positions')
    plt.scatter(goal_position[0], goal_position[1], color='red', marker='*', s=100, label='Goal Position')
    
    # Plot outside walls
    for wall in outside_walls:
        x_values = [wall[0][0], wall[1][0]]
        y_values = [wall[0][1], wall[1][1]]
        plt.plot(x_values, y_values, color="black")

    # Plot inner walls from walls list
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

def is_collision(position, walls):
    x, y = position
    for wall in walls:
        if (wall['x_min'] <= x <= wall['x_max']) and (wall['y_min'] <= y <= wall['y_max']):
            return True
    return False

def random_position(walls, outside_walls, goal_position, epsilon, margin=0.02):
    
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

        position = np.array([x, y])

        # Check if the position is inside wall_3 and not within epsilon of the goal
        if not is_inside_wall(x, y) and np.linalg.norm(position - goal_position) > epsilon:
            return np.array([x, y, vx, vy])

def is_inside_wall(x, y):
    
    # Wall 3 dimensions from the XML
    wall_3_x_min = 0.5 - 0.1  # Center x - half width
    wall_3_x_max = 0.5 + 0.1  # Center x + half width
    wall_3_y_min = -0.15  # Center y - half height
    wall_3_y_max = 0.15   # Center y + half height

    # Check if the point is inside the wall bounds
    return wall_3_x_min <= x <= wall_3_x_max and wall_3_y_min <= y <= wall_3_y_max

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
    option = mujoco.MjvOption()  # Initialize MjvOption

    # Set up the camera for an overhead view
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

def main():
    
    goal_position = np.array([1.0, 0.0])  # Fixed goal position (xg, yg)
    epsilon = 0.2  # Goal threshold
    alpha = 0.001  # Critic learning rate
    beta = 0.001   # Actor learning rate

    walls = [
        {"x_min": 0.5, "x_max": 0.6, "y_min": -0.15, "y_max": 0.15}
    ]
    outside_walls = [
        [[-0.5, -0.4], [-0.5, 0.4]],
        [[1.5, -0.4], [1.5, 0.4]],
        [[-0.5, 0.4], [1.5, 0.4]],
        [[-0.5, -0.4], [1.5, -0.4]],
    ]

    # Initialize MuJoCo model for rendering
    mujoco_model = mujoco.MjModel.from_xml_path("ball_square.xml")
    mujoco_data = mujoco.MjData(mujoco_model)
    model = ActorCriticModel(input_dim=4, hidden_dim1=256, hidden_dim2=256, action_dim=2)
    
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
            actor_critic_training(
                model=model,
                goal_position=goal_position,
                walls=walls,
                outside_walls=outside_walls,
                num_episodes=1,
                max_steps=1800,
                gamma=0.99,
                alpha=alpha,
                beta=beta,
                log_interval=1,
                render=True,
                mujoco_model=mujoco_model,
                mujoco_data=mujoco_data,
                epsilon=epsilon,
                fixed=False  # Not fixed starting position
            )
        elif choice == 2:
            model = actor_critic_training(
                model=model,
                goal_position=goal_position,
                walls=walls,
                outside_walls=outside_walls,
                num_episodes=10,
                max_steps=1800,
                gamma=0.99,
                alpha=alpha,
                beta=beta,
                log_interval=100,
                render=False,
                plot=True,  # Plot training progress
                epsilon=epsilon
            )
        elif choice == 3:
            print("Rendering with fixed starting position...")
            actor_critic_training(
                model=model,
                goal_position=goal_position,
                walls=walls,
                outside_walls=outside_walls,
                num_episodes=1,
                max_steps=3600,
                gamma=0.99,
                alpha=1e-3,
                beta=1e-3,
                log_interval=1,
                render=True,
                mujoco_model=mujoco_model,
                mujoco_data=mujoco_data,
                epsilon=epsilon,
                fixed=True  # Fixed starting position and action selection
            )
        elif choice == 4:
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please select from the menu.")

if __name__ == "__main__":
    main()
