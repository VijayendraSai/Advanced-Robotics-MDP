import mujoco
import glfw
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Outputting a single value for V(s)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        value = self.fc2(x)
        return value

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Outputting policy parameters

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = torch.softmax(self.fc2(x), dim=-1)  # Output a probability distribution
        return action_probs

def reward_function(state, goal_position, epsilon=0.1):
    distance_to_goal = np.linalg.norm(state[:2] - goal_position[:2])
    return 1 if distance_to_goal <= epsilon else 0

def actor_critic_training(start_pos, goal_position, gamma=0.99, alpha=0.001, beta=0.001, max_steps=100):
    # Hyperparameters
    input_dim = 4  # Assuming state has 4 dimensions (e.g., x, y, velocity x, velocity y)
    hidden_dim = 128
    action_dim = 2  # Number of actions (e.g., move left/right, up/down)

    # Initialize neural networks
    actor = ActorNetwork(input_dim, hidden_dim, action_dim)
    critic = CriticNetwork(input_dim, hidden_dim)

    # Optimizers for the actor and critic networks
    actor_optimizer = optim.Adam(actor.parameters(), lr=alpha)
    critic_optimizer = optim.Adam(critic.parameters(), lr=beta)

    state = torch.tensor(start_pos, dtype=torch.float32).unsqueeze(0)  # Shape (1, input_dim)
    epsilon = 0.1

    for t in range(max_steps):
        # Calculate the reward based on the current state and goal position
        reward = reward_function(state[0].numpy(), goal_position, epsilon)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)

        # Actor forward pass: Get action probabilities
        action_probs = actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        # Critic forward pass: Get value of the current state
        value = critic(state)
        
        # Simulate the environment step (assume action moves position by a small amount)
        next_state_np = state[0].numpy() + (action.item() * 0.1)  # Example of state transition
        next_state = torch.tensor(next_state_np, dtype=torch.float32).unsqueeze(0)
        
        # Critic forward pass for the next state
        next_value = critic(next_state).detach()  # Do not backpropagate from this
        td_error = reward_tensor + gamma * next_value - value
        
        # Critic loss and update
        critic_loss = td_error.pow(2).mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        # Actor loss and update
        actor_loss = -action_dist.log_prob(action) * td_error.detach()  # Multiply log-prob by TD error
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        # Move to the next state
        state = next_state

        # Logging for debugging
        print(f"Step {t}: State={state[0].numpy()}, Reward={reward}, TD Error={td_error.item()}, Actor Loss={actor_loss.item()}, Critic Loss={critic_loss.item()}")

        # Check if the goal is reached
        if reward == 1:
            print(f"Goal reached at step {t}!")
            break

    return actor, critic

def is_in_goal_area(point, goal_area):
    
    x_min = min([coord[0] for coord in goal_area])
    x_max = max([coord[0] for coord in goal_area])
    y_min = min([coord[1] for coord in goal_area])
    y_max = max([coord[1] for coord in goal_area])

    return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max

def sample_random_position(goal_area=None, goal_bias=1, bias_strength=0.5):
    
    x = random.uniform(-0.5, 1.5)
    y = random.uniform(-0.4, 0.4)
    
    return np.array([x, y])

def simulate(pid_x, pid_y, position, target_position, tolerance=0.05, max_steps=100, dt=0.01, max_speed=3, min_speed=1, slowdown_distance=1.0, logging=True):
    
    current_position = np.array(position)

    for step in range(max_steps):

        # put here the logic for mdp
        
        if logging:
            print(f"Step {step}: Position={current_position}, Distance to Target={distance_to_target}, Desired Speed={desired_speed}")

        if distance_to_target <= tolerance:
            return current_position
    
    return current_position

def render_scene(model, data, options, scene, context, viewport, camera, window):
    
    mujoco.mjv_updateScene(model, data, options, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    glfw.poll_events()
    glfw.swap_buffers(window)
    
    return

def move_ball(model, data, path, window=None, scene=None, context=None, options=None, viewport=None, camera=None, plot_enabled=True, render_enabled=True, logging=True, Tmax=120):
   
    # set the parameters for the model
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    dt = 0.01
    time_data, deviation_data = [], []
    time_elapsed = 0
    max_speed, min_speed = 15, 1.25
    slowdown_distance = .75
    
    # random position
    start_pos = random_position() 

    while True:
        
        ball_pos = data.xpos[ball_id][:2]
        distance_to_goal = np.linalg.norm(end_pos - ball_pos)
        
        # break when a point is reached
        if distance_to_goal < 0.075:
            if logging:
                print("Reached node", i + 1)
            break
        
        # update the speed based on the slowdown distance
        desired_speed = update_speed(distance_to_goal, slowdown_distance, max_speed, min_speed)

        if logging:
            print(f"Distance to goal: {distance_to_goal}, Desired speed: {desired_speed}")
       
        # update the controllers based on new ball position
        control_x, control_y = apply_mdp()

        if logging:
            print(f"Control signals: control_x={control_x}, control_y={control_y}")

        data.ctrl[0], data.ctrl[1] = control_x, control_y
        mujoco.mj_step(model, data)
        
        # render the 3d
        if render_enabled:
            render_scene(model, data, options, scene, context, viewport, camera, window)

        time_elapsed += dt

        if time_elapsed > Tmax:
            break

    if plot_enabled:
        plt.close(fig) 
        plt.close('all')

    return time_elapsed

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
    
    # camera settings
    camera.distance = 3.0
    camera.elevation = -45.0
    camera.azimuth = 0.0

    viewport = mujoco.MjrRect(0, 0, 1200, 900)
    
    return window, camera, scene, context, options, viewport

def model_creation(start_pos, goal_area, walls, outside_walls):
   
    # load the MuJoCo model
    model = mujoco.MjModel.from_xml_path("ball_square.xml")
    data = mujoco.MjData(model)
        
    # create the window
    window, camera, scene, context, options, viewport = init_glfw_window(model)
    
    # run the 3d simulation with the path and window objects
    move_ball(pid_x, pid_y, model, data, path, window, scene, context, options, viewport, camera)
    glfw.terminate()
    
    return

def generate_random_seed():
    seed = random.randint(0, 2**32 - 1)
    return seed

def main():
    
    # define the goal area, walls, and the starting position
    goal_area = [[0.9, -0.3], [0.9, 0.3], [1.1, 0.3], [1.1, -0.3]]
    outside_walls = [
        [[-0.5, -0.4], [-0.5, 0.4]],
        [[1.5, -0.4], [1.5, 0.4]],
        [[-0.5, 0.4], [1.5, 0.4]],
        [[-0.5, -0.4], [1.5, -0.4]]]
    walls = {"wall_3": [[0.5, -0.15], [0.5, 0.15], [0.6, 0.15], [0.6, -0.15]]}
    start_pos = [0, 0]

    print("\nMenu:")
    print("1. Model Simulation")
    print("2. Train the networks")
    print("2. Execution Time Success Rate")
    print("3. Quit")
    
    # allow the user to determine which objective to run
    choice = input("Enter your choice: ")
    if choice.isdigit():
        choice = int(choice)
    else:
        print("Invalid input. Please enter a number.")
        return

    if choice == 1:
        model_creation(start_pos, goal_area, walls, outside_walls)
    elif choice = 2:
        actor_critic_training(start_pos, goal_position, gamma=0.99, alpha=0.001, beta=0.001, max_steps=100)
    else choice == 3:
        num_trials = 30
        # run all the trails in a loop with changing time
        for Tmax in [30, 20, 10, 5]:
            print(f'Starting {num_trials} trails for {Tmax} seconds')
            run_execution_trials(start_pos, goal_area, walls, outside_walls, num_trials, Tmax)
    elif choice == 3:
        print("Exiting the program. Goodbye!")
    else:
        print("Invalid option. Please try again.")
    return

if __name__ == "__main__":
    main()
