import time
import numpy as np
import os
from datetime import datetime
import random
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pprint
from random import *

import psutil
from torch.utils.tensorboard import SummaryWriter
import torch
# In Flatland you can use custom observation builders and predicitors
# Observation builders generate the observation needed by the controller
# Preditctors can be used to do short time prediction which can help in avoiding conflicts in the network
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
# First of all we import the Flatland rail environment
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env import RailEnvActions
# Import the railway generators
from flatland.envs.custom_rail_generator import rail_custom_generator
from flatland.envs.rail_env_utils import delay_a_train, make_a_deterministic_interruption
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
# Import the schedule generators
from flatland.envs.custom_schedule_generator import custom_schedule_generator
from flatland.envs.plan_to_follow_utils import action_to_do, divide_trains_in_station_rails, control_timetable
# Import the different structures needed
from configuration import railway_example, stations, timetable_example, example_training
# Import the agent class
from flatland.envs.agent import RandomAgent
from flatland.envs.step_utils.states import TrainState
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from flatland.utils.timer import Timer
from flatland.utils.observation_utils import normalize_observation
from reinforcement_learning.dddqn_policy import DDDQNPolicy
# Import training and observation parameters
from parameters import training_params, obs_params

# Check the maximum possible delay...180 not good for now
def calculate_metric(env, timetable):
    positions = env.cur_episode
    prev_station = 0
    delta = 250
    metric_result = []
    for i_agent in range(env.get_num_agents()):
        if not env.agents[i_agent].state == TrainState.MALFUNCTION:
            station_vector = [delta] * len(timetable[i_agent][0])
            for i_station in range(len(timetable[i_agent][0])):
                for step in range(len(positions)):
                    if positions[step][i_agent] == timetable[i_agent][0][i_station] and positions[step][i_agent] != prev_station:
                        prev_station = positions[step][i_agent]
                        distance_delay = ((step - timetable[i_agent][1][i_station])**2)**(1/2)
                        station_vector[i_station] = distance_delay
            metric_result.append(station_vector)
    metric_sum = sum(sum(x) for x in metric_result)
    dimension = 0
    for i in range(len(metric_result)):
        for j in range(len(metric_result[i])):
            dimension += 1
    metric_normalized = 1 - (metric_sum / (delta*dimension))
    return metric_normalized

def choose_a_random_training_configuration(env, max_steps):
    case = randint(0,3)
    if case == 0:
        make_a_deterministic_interruption(env.agents[1], max_steps)
        make_a_deterministic_interruption(env.agents[2], max_steps)
        return    
    elif case == 1:
        env.agents[1].initial_position = (6,15)
        env.agents[2].initial_position = (5,15)
        make_a_deterministic_interruption(env.agents[1], max_steps)
        make_a_deterministic_interruption(env.agents[2], max_steps)
        return
    elif case == 2:
        make_a_deterministic_interruption(env.agents[1], max_steps)
        env.agents[2].malfunction_handler.malfunction_down_counter = max_steps
        return       
    elif case == 3:
        env.agents[1].malfunction_handler.malfunction_down_counter = max_steps
        env.agents[2].initial_position = (5,15)
        make_a_deterministic_interruption(env.agents[2], max_steps)
        return

def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼", "↓"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


###### TRAINING PARAMETERS #######
n_episodes = 4000
eps_start = 1
eps_end = 0.01
eps_decay = 0.99
max_steps = 250     # 1440 one day
checkpoint_interval = 100
training_id = '0' 
render = False

######### FLAGS ##########
# Flag for the first training
training_flag = example_training
# Flag active in case of interruptions
interruption = True
# Flag to select the agent ----> multi agent or external controller
multi_agent = True
# Flag to save the video or not
video_save = False


# The specs for the custom railway generation are taken from structures.py file
specs = railway_example

widht = len(specs[0])
height = len(specs)

stations_position = []

# Defining the name of the different stations
for i in range(1, len(stations)):
    stations_position.append(stations[i][0])

# Timetable conteins the station where the train should pass, from starting station to aim, and conteins the time at which
# each train has to pass in the station, the last number represent the velocity of train (high velocity, intercity or regional)
# Each row represent a different train

print('------ Calculating the timetable')
print()
timetable = timetable_example

# Number of agents is the rows of the timetable
num_of_agents = len(timetable)

# Check if the timetable is feaseble or not, the function is in schedule_generators
# A timetable is feaseble if the difference of times between two stations is positive and let the trains to reach the successive station
# if two stations are very distant from each other the difference of times can't be very small
seed = 2

# Generating the railway topology, with stations
# Arguments of the generator (specs of the railway, position of stations, timetable)
rail_custom = rail_custom_generator(specs, stations_position, timetable)

transition_map_example, agent_hints = rail_custom(widht, height, num_of_agents)

divide_trains_in_station_rails(timetable, transition_map_example)

control_timetable(timetable,transition_map_example)

for i in range(len(timetable)):
    print(timetable[i])
 
time.sleep(3)

# We can now initiate the schedule generator with the given speed profiles
schedule_generator_custom = custom_schedule_generator(timetable = timetable)

print()
print('------- Calculating the action scheduled')
actions_scheduled = action_to_do(timetable, transition_map_example)

# DEBUG
for i in range(len(actions_scheduled)):
    print()
    print(actions_scheduled[i])
    print()

time.sleep(3)

if multi_agent:

    observation_parameters = Namespace(**obs_params)

    observation_tree_depth = observation_parameters.observation_tree_depth
    observation_radius = observation_parameters.observation_radius
    observation_max_path_depth = observation_parameters.observation_max_path_depth

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    Observer = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)
else:
    Observer = GlobalObsForRailEnv()
    # Ricordarsi che noi vogliamo applicare il RL solo in un intorno della linea dove c'è stata l'interruzione
    # Vogliamo in questo caso un osservatore globale? Forsse meglio valutarne anche uno limitato
    # Ragiona se costruire un osservatore che consideri solo i binari possa essere tanto vantaggioso o no?

stochastic_data = MalfunctionParameters(
    malfunction_rate = 0,  # Rate of malfunction occurence
    min_duration = 15,  # Minimal duration of malfunction
    max_duration = 40  # Max duration of malfunction
)

malfunction_generator = ParamMalfunctionGen(stochastic_data)

env = RailEnv(  width= widht,
                height= height,
                rail_generator = rail_custom,
                line_generator=schedule_generator_custom,
                number_of_agents= num_of_agents,
                malfunction_generator = malfunction_generator,
                obs_builder_object=Observer,
                remove_agents_at_target=True,
                record_steps=True,
                max_episode_steps = max_steps - 1
                )

env.reset()


# If I want I can delay a specific train a specific time
'''
delay_a_train(delay = 250, train = env.agents[1], delay_time = 2, time_of_train_generation = 1, actions = actions_scheduled)
delay_a_train(delay = 250, train = env.agents[2], delay_time = 2, time_of_train_generation = 1, actions = actions_scheduled)
'''

for i in range(len(actions_scheduled)):
    print(actions_scheduled[i])

env_renderer = RenderTool(env,
                          screen_height=480,
                          screen_width=720)  # Adjust these parameters to fit your resolution


# This thing is importand for the RL part, initialize the agent with (state, action) dimension
# Initialize the agent with the parameters corresponding to the environment and observation_builder
if multi_agent:
    n_agents = env.get_num_agents()
    n_features_per_node = env.obs_builder.observation_dim
    n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
    state_size = n_features_per_node * n_nodes

    action_size = env.action_space[0]

    action_count = [0] * action_size
    action_dict = dict()
    agent_obs = [None] * n_agents
    agent_prev_obs = [None] * n_agents
    agent_prev_action = [2] * n_agents
    update_values = [False] * n_agents

    controller = RandomAgent(state_size, action_size)

    # Smoothed values used as target for hyperparameter tuning
    smoothed_normalized_score = -1.0
    smoothed_eval_normalized_score = -1.0
    smoothed_completion = 0.0
    smoothed_eval_completion = 0.0

    train_params = training_params

    policy = DDDQNPolicy(state_size, action_size, train_params)

    # TensorBoard writer
    writer = SummaryWriter()
    writer.add_hparams(vars(train_params), {})
    writer.add_hparams(vars(train_params), {})
    writer.add_hparams(vars(observation_parameters), {})

    training_timer = Timer()
    training_timer.start()

else:
    n_agents = env.get_num_agents()
    state_size = (widht * height)
    # The number of actions is the combination of the number of actions by the number of agents
    action_size = env.action_space[0] ** env.get_num_agents()

    action_count = [0] * action_size
    action_dict = dict()
    agent_obs = [None] * n_agents
    agent_prev_obs = [None] * n_agents
    agent_prev_action = [2] * n_agents
    update_values = [False] * n_agents

    controller = RandomAgent(state_size, action_size)

    q_table = np.zeros([state_size, action_size])


    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    # For plotting metrics
    all_epochs = []
    all_penalties = []


# Lets try to enter with all of these agents at the same time
action_dict = dict()

# Now that you have seen these novel concepts that were introduced you will realize that agents don't need to take
# an action at every time step as it will only change the outcome when actions are chosen at cell entry.
# Therefore the environment provides information about what agents need to provide an action in the next step.
# You can access this in the following way.

# Chose an action for each agent
for a in range(env.get_num_agents()):
    action = controller.act(0)
    action_dict.update({a: action})
# Do the environment step

observations, rewards, dones, information = env.step(action_dict)

print("\n The following agents can register an action:")
print("========================================")
for info in information['action_required']:
    print("Agent {} needs to submit an action.".format(info))

# We recommend that you monitor the malfunction data and the action required in order to optimize your training
# and controlling code.

# Let us now look at an episode playing out 

print("\nStart episode...")

# Reset the rendering system
env_renderer.reset()

# Here you can also further enhance the provided observation by means of normalization
# See training navigation example in the baseline repository

score = 0
# Run episode
frame_step = 0

os.makedirs("output/frames", exist_ok=True)

for episode_idx in range(n_episodes + 1):
    
    deterministic_interruption_activation = False

    step_timer = Timer()
    reset_timer = Timer()
    learn_timer = Timer()
    preproc_timer = Timer()
    inference_timer = Timer()

    # Reset environment
    reset_timer.start()

    # Reset environment and get initial observations for all agents
    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
    reset_timer.end()
    for idx in range(env.get_num_agents()):
        tmp_agent = env.agents[idx]
        tmp_agent.speed_counter.speed = 1 / (idx + 1)  # TODO rigestisci le velocità iniziali
    env_renderer.reset()

    if train_params.render:
        env_renderer.set_new_rail()

    score = 0
    nb_steps = 0
    actions_taken = []

    if multi_agent:
        # Build initial agent-specific observations
        for agent in env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth, observation_radius=observation_radius)
                agent_prev_obs[agent] = agent_obs[agent].copy()
    else:
        for agent in env.get_agent_handles():
            agent_obs[agent] = obs[agent]
            agent_prev_obs[agent] = agent_obs[agent].copy()

    # Run episode (one day long, 1 step is 1 minute) 1440
    for step in range(max_steps):
        if video_save:
            env_renderer.gl.save_image("output/frames/flatland_frame_step_{:04d}.bmp".format(step))

        inference_timer.start() 

    # Here define the actions to do
        # Broken agents
        if training_flag == 'training0' and not deterministic_interruption_activation:
            choose_a_random_training_configuration(env, max_steps)
        if training_flag == 'training1':
            make_a_deterministic_interruption(env.agents[2], max_steps)
            make_a_deterministic_interruption(env.agents[3], max_steps)
        if training_flag == 'training1.1':
            make_a_deterministic_interruption(env.agents[2], max_steps)

        # Chose an action for each agent in the environment
        # If not interruption, the actions to do are stored in a matrix
        #       - each row of the matrix is a train
        #       - each column represent the action the train has to do at each time instant
        
        for a in range(env.get_num_agents()):
            if env.agents[a].state == TrainState.DONE:
                env.dones[a] = True
            if env.agents[a].state == TrainState.MALFUNCTION:
                interruption = True
            if not multi_agent and interruption: # debug 
                break
            if step >= timetable[a][1][0]:
                # Normal plan to follow
                if not interruption and (step - timetable[a][1][0]) < len(actions_scheduled[a]):
                    action = actions_scheduled[a][step - timetable[a][1][0]]
                # Interruption
                elif interruption:
                    if multi_agent:
                        if info['action_required'][a]:
                            update_values[a] = True
                            action = policy.act(agent_obs[a], eps=eps_start)

                            action_count[action] += 1
                            actions_taken.append(action)
                        else:
                            # An action is not required if the train hasn't joined the railway network,
                            # if it already reached its target, or if is currently malfunctioning.
                            update_values[a] = False
                            action = 0
                    else:
                        action = np.random.choice([RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT, RailEnvActions.MOVE_LEFT, 
                        RailEnvActions.STOP_MOVING, RailEnvActions.REVERSE])
                # choose random from all the possible actions
                else:
                    action = np.random.choice([RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT, RailEnvActions.MOVE_LEFT, 
                        RailEnvActions.STOP_MOVING, RailEnvActions.REVERSE])

                action_dict.update({a: action})


        inference_timer.end()

        # Environment step which returns the observations for all agents, their corresponding
        # reward and whether their are done
        # Environment step
        step_timer.start()
        next_obs, all_rewards, done, info = env.step(action_dict)
        step_timer.end()

        # Render an episode at some interval
        if render:
            env_renderer.render_env(
                    show=True, show_observations = False, frames = True, episode = True, step = True
                )
        # Update replay buffer and train agent
        if multi_agent:
            for agent in env.get_agent_handles():
                if update_values[agent] or done['__all__']:
                    # Only learn from timesteps where somethings happened
                    learn_timer.start()
                    policy.step(agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent], agent_obs[agent], done[agent])
                    learn_timer.end()

                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                # Preprocess the new observations
                if next_obs[agent]:
                    preproc_timer.start()
                    agent_obs[agent] = normalize_observation(next_obs[agent], observation_tree_depth, observation_radius=observation_radius)
                    preproc_timer.end()

                score += all_rewards[agent]

            nb_steps = step
        else:
            for a in range(env.get_num_agents()):
                controller.step((obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
                score += all_rewards[a]
        obs = next_obs.copy()
        if done['__all__']:
            break
        #break if the first agent has done
        if ((training_flag == 'training0') and (env.dones[0] == True)) or \
            ((training_flag == 'training1') and (env.dones[0] == True) and (env.dones[1] == True)) or \
            ((training_flag == 'training1.1') and (env.dones[0] == True) and (env.dones[1] == True)):
            break
    print('Episode Nr. {}\t Score = {}'.format(episode_idx, score))
    print()
    
    # metric near to 1 is great result
    metric = calculate_metric(env, timetable)
    
    if multi_agent:
        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        # Collect information about training
        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        normalized_score = score / (max_steps * env.get_num_agents())
        action_probs = action_count / np.sum(action_count)
        action_count = [1] * action_size

        smoothing = 0.99
        smoothed_normalized_score = smoothed_normalized_score * smoothing + normalized_score * (1.0 - smoothing)
        smoothed_completion = smoothed_completion * smoothing + completion * (1.0 - smoothing)

        # Print logs
        if episode_idx % checkpoint_interval == 0:
            '''
            torch.save(policy.qnetwork_local, './checkpoints/' + training_id + '-' + str(episode_idx) + '.pth')
            if save_replay_buffer:
                policy.save_replay_buffer('./replay_buffers/' + training_id + '-' + str(episode_idx) + '.pkl')
            '''

            if train_params.render:
                env_renderer.close_window()

        print(
            '\r🚂 Episode {}'
            '\t 🏆 Score: {:.3f}'
            ' Avg: {:.3f}'
            '\t 💯 Done: {:.2f}%'
            ' Avg: {:.2f}%'
            '\t 🎲 Epsilon: {:.3f} '
            '\t 🔀 Action Probs: {}'
            '\t Metric {}'.format(
                episode_idx,
                normalized_score,
                smoothed_normalized_score,
                100 * completion,
                100 * smoothed_completion,
                eps_start,
                format_action_prob(action_probs),
                metric
            ), end=" ")

    interruption = False



# Change the position of the interrupted agents
env.reset()

# NEW CONFIGURATION
if training_flag == 'training0':
    env.agents[1].initial_position = (6,10)
    env.agents[2].initial_position = (5,10)

env.reset()

#################
##### TEST 1 ####
#################

for step in range(max_steps):
    # Broken agents
    if training_flag == 'training0':
        make_a_deterministic_interruption(env.agents[1], max_steps)
        make_a_deterministic_interruption(env.agents[2], max_steps)
    if training_flag == 'training1':
        make_a_deterministic_interruption(env.agents[2], max_steps)
        make_a_deterministic_interruption(env.agents[3], max_steps)
    if training_flag == 'training1.1':
        make_a_deterministic_interruption(env.agents[2], max_steps)
    for a in range(env.get_num_agents()):
        update_values[a] = True
        action = policy.act(agent_obs[a], eps=eps_start)

        action_count[action] += 1
        actions_taken.append(action)
        action_dict.update({a: action})
        
        next_obs, all_rewards, done, info = env.step(action_dict)
    
    policy.test()
    
    if done['__all__']:
        break
    #break if the first agent has done
    if ((training_flag == 'training0') and (env.dones[0] == True)) or \
        ((training_flag == 'training1') and (env.dones[0] == True) and (env.dones[1] == True)) or \
        ((training_flag == 'training1.1') and (env.dones[0] == True) and (env.dones[1] == True)):
        break

# metric most possible near to 0
metric = calculate_metric(env, timetable)

tasks_finished = sum(done[idx] for idx in env.get_agent_handles())

print(  'Test 1 concluded:'
        '\t 🏆 Score: {:.3f}'
        '\t Agent completed {}'
        '\t Metric {}'.format(
            score,
            tasks_finished,
            metric
        ), end=" ")
    

# Change the position of the interrupted agents
env.reset()

# NEW CONFIGURATION
if training_flag == 'training0':
    env.agents[1].initial_position = (6,8)

env.reset()

#################
##### TEST 2 ####
#################

for step in range(max_steps):
    # Broken agents
    if training_flag == 'training0':
        make_a_deterministic_interruption(env.agents[1], max_steps)
    if training_flag == 'training1':
        make_a_deterministic_interruption(env.agents[2], max_steps)
        make_a_deterministic_interruption(env.agents[3], max_steps)
    if training_flag == 'training1.1':
        make_a_deterministic_interruption(env.agents[2], max_steps)
    for a in range(env.get_num_agents()):
        update_values[a] = True
        action = policy.act(agent_obs[a], eps=eps_start)

        action_count[action] += 1
        actions_taken.append(action)
        action_dict.update({a: action})
        
        next_obs, all_rewards, done, info = env.step(action_dict)
    
    policy.test()
    
    if done['__all__']:
        break
    #break if the first agent has done
    if ((training_flag == 'training0') and (env.dones[0] == True)) or \
        ((training_flag == 'training1') and (env.dones[0] == True) and (env.dones[1] == True)) or \
        ((training_flag == 'training1.1') and (env.dones[0] == True) and (env.dones[1] == True)):
        break
    
# metric most possible near to 0
metric = calculate_metric(env, timetable)

tasks_finished = sum(done[idx] for idx in env.get_agent_handles())

print(  'Test 2 concluded:'
        '\t 🏆 Score: {:.3f}'
        '\t Agent completed {}'
        '\t Metric {}'.format(
            score,
            tasks_finished,
            metric
        ), end=" ")
    