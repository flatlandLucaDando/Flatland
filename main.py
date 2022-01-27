import time
import numpy as np
import os
from argparse import Namespace
from pprint import pprint
from random import *
from datetime import datetime

import torch

from torch.utils.tensorboard import SummaryWriter
# In Flatland you can use custom observation builders and predicitors
# Observation builders generate the observation needed by the controller
# Preditctors can be used to do short time prediction which can help in avoiding conflicts in the network
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv, GlobalObsModifiedRailEnv, TreeTimetableObservation
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
from flatland.utils.deadlock_check import find_and_punish_deadlock

from flatland.utils.timer import Timer
from flatland.utils.observation_utils import normalize_global_observation, normalize_observation
from reinforcement_learning.dddqn_policy import DDDQNPolicy
# Import training and observation parameters
from parameters import training_params, obs_params


import matplotlib.pyplot as plt
import matplotlib.animation
plt.rcParams["animation.html"] = "jshtml"

def display_episode(frames):
    fig, ax = plt.subplots(figsize=(12,12))
    imgplot = plt.imshow(frames[0])
    def animate(i):
        imgplot.set_data(frames[i])
    animation = matplotlib.animation.FuncAnimation(fig, animate, frames=len(frames))
    return animation

def check_conflicts(env):
    for a in range(len(env.agents)):
        if env.agents[a].state_machine.st_signals.movement_conflict == True:
            return True

def choose_a_random_training_configuration(env, max_steps):
    if example_training == 'training0':
        case = 0
        if case == 0:
            env.agents[1].initial_position = (6,8)
            env.agents[2].initial_position = (5,8)
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
            env.agents[1].initial_position = (6,8)
            make_a_deterministic_interruption(env.agents[1], max_steps)
            env.agents[2].malfunction_handler.malfunction_down_counter = max_steps
            return       
        elif case == 3:
            env.agents[1].malfunction_handler.malfunction_down_counter = max_steps
            env.agents[2].initial_position = (5,15)
            make_a_deterministic_interruption(env.agents[2], max_steps)
            return
    else:
        env.agents[1].initial_position = (5,8)
        make_a_deterministic_interruption(env.agents[1], max_steps)
        

def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼", "↓"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


###### TRAINING PARAMETERS #######
n_episodes = 10000
eps_start = 1
eps_end = 0.01
eps_decay = 0.99975
max_steps = 150     # 1440 one day
checkpoint_interval = 100
 # Unique ID for this training
now = datetime.now()
training_id = now.strftime('%y%m%d%H%M%S')

#########################################################
# Parameters that should change the results:

# eps_decay 
# step_maximum_penality
# % used in the sparse reward
# penalty given in the function find_and_punish_deadlock
#########################################################

render = False

######### FLAGS ##########
# Flag for the first training
training_flag = example_training
# Flag active in case of interruptions
interruption = False
# Flag to select the agent ----> multi agent or external controller
multi_agent = True
# Flag to save the video or not
video_save = False
# Flag to applicate RL also in case of no interruptions
reinforcemente_learning = True
# Flag to output different things important for the debug
debug = False
# flag to select the tree observer
tree_observer = True

# The specs for the custom railway generation are taken from structures.py file
specs = railway_example

widht = len(specs[0])
height = len(specs)

stations_position = []
    
# Positions of the stations
for i in range(len(stations)):
    stations_position.append(stations[i].position)

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

print('Station | Departure time |  Train id')
print('-------------------------------------')
for i in range(len(timetable)):
    for j in range(len(timetable[i][0])):
        print(timetable[i][0][j].name, ' | ' ,timetable[i][1][j], '  |  ', timetable[i][2].id)
        print('-------------------------------------')
        
 
time.sleep(3)

# We can now initiate the schedule generator with the given speed profiles
schedule_generator_custom = custom_schedule_generator(timetable = timetable)

print()
print('------- Calculating the action scheduled')
actions_scheduled = action_to_do(timetable, transition_map_example)

# DEBUG
if debug:
    for i in range(len(actions_scheduled)):
        print()
        print(actions_scheduled[i])
        print()

    time.sleep(3)

if multi_agent and tree_observer:

    observation_parameters = Namespace(**obs_params)

    observation_tree_depth = observation_parameters.observation_tree_depth
    observation_radius = observation_parameters.observation_radius
    observation_max_path_depth = observation_parameters.observation_max_path_depth

    # Observation builder
    predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
    Observer = TreeTimetableObservation(max_depth=observation_tree_depth, predictor=predictor)
elif multi_agent:
    Observer = GlobalObsModifiedRailEnv()
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
if debug:
    for i in range(len(actions_scheduled)):
        print(actions_scheduled[i])

env_renderer = RenderTool(env,
                          screen_height=1080,
                          screen_width=1080)  # Adjust these parameters to fit your resolution


# This thing is importand for the RL part, initialize the agent with (state, action) dimension
# Initialize the agent with the parameters corresponding to the environment and observation_builder
if multi_agent:
    if tree_observer:
        n_features_per_node = env.obs_builder.observation_dim
        n_nodes = sum([np.power(4, i) for i in range(observation_tree_depth + 1)])
        state_size = n_features_per_node * n_nodes
    else:
        observation = env.obs_builder.get()
        state_size = observation.size
        
    n_agents = env.get_num_agents()
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
    writer.add_hparams(hparam_dict = {'Number of episodes': n_episodes, 'Starting epsilon': eps_start, 'Ending epsilon': eps_end , 'Epsilon decay': eps_decay,
                                      'Max steps': max_steps, 'Checkpoint interval': checkpoint_interval, 'Training id': training_id, 'Render': False}, metric_dict={})
    writer.flush()
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

observations, rewards, done, information = env.step(action_dict)

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
frames = []

# Conflicts 
num_of_conflict = 0
avg_num_of_conflict = 0

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
    metric = 0
    actions_taken = []

    if multi_agent:
        # Build initial agent-specific observations
        for agent in env.get_agent_handles():
            if tree_observer:
                agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth, observation_radius=observation_radius)
            else:
                agent_obs[agent] = normalize_global_observation(obs[agent])
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
        if training_flag == 'training0' and not deterministic_interruption_activation or example_training == 'one_rail':
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
            if env.agents[a].state == TrainState.MALFUNCTION:
                interruption = True
            if not multi_agent and interruption: # debug 
                break
            if step >= timetable[a][1][0]:
                # Normal plan to follow
                if not interruption and (step - timetable[a][1][0]) < len(actions_scheduled[a]):
                    if not reinforcemente_learning:
                        action = actions_scheduled[a][step - timetable[a][1][0]]
                # Interruption
                if interruption or reinforcemente_learning:
                    if multi_agent:
                        if info['action_required'][a]:
                            update_values[a] = True
                            action = policy.act(a, agent_obs[a], eps=eps_start)

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
        for agent_handle in env.get_agent_handles():
                done[agent_handle] = (env.agents[agent_handle].state == TrainState.DONE)
        step_timer.end()
        
        deadlocked_agents, all_rewards, = find_and_punish_deadlock(env, all_rewards, 0)

        # Render an episode at some interval
        #frame = env_renderer.render_env(show=False, show_observations=False, show_inactive_agents=False, show_predictions=False, return_image=True)
        #frames.append(frame)
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
                    policy.step(agent, agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent], agent_obs[agent], done[agent])
                    learn_timer.end()

                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                # Preprocess the new observations
                preproc_timer.start()
                if tree_observer:
                    agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth, observation_radius=observation_radius)
                else:
                    agent_obs[agent] = normalize_global_observation(obs[agent])
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
        if ((training_flag == 'training0') and (done[0] == True)) or \
            ((training_flag == 'training1') and (done[0] == True) and (done[1] == True)) or \
            ((training_flag == 'training1.1') and (done[0] == True) and (done[1] == True)):
            break
        
        if check_conflicts(env):
            
            num_of_conflict += 1
            
            break
        
    print()
    print('Episode Nr. {}\t Score = {}'.format(episode_idx, score))
    
    # metric near to 1 is great result
    for agent_handle in env.get_agent_handles():
        metric += env.calculate_metric_single_agent(timetable, agent_handle)
    
    metric = metric/len(env.get_agent_handles())
    
    if multi_agent:
        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        # Collect information about training
        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        tasks_deadlocked = sum(deadlocked_agents[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        deadlocked = tasks_deadlocked / max(1, env.get_num_agents())
        normalized_score = score / (max_steps * env.get_num_agents())
        action_probs = action_count / np.sum(action_count)
        action_count = [1] * action_size
        
        avg_num_of_conflict = num_of_conflict / (episode_idx + 1)
        
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
            '\t 💯 Done: {}%'
            ' Avg: {:.3f}%'
            '\t Num of conflicts: {}'
            ' Avg: {:.3f}'
            '\t 🎲 Epsilon: {:.3f} '
            '\t 🔀 Action Probs: {}'
            '\t Metric: {}'.format(
                episode_idx,
                normalized_score,
                smoothed_normalized_score,
                100 * completion,
                100 * smoothed_completion,
                num_of_conflict,
                avg_num_of_conflict,
                eps_start,
                format_action_prob(action_probs),
                metric
            ), end=" ")
        print()

    interruption = False
    
    writer.add_scalar("Reward", score, episode_idx)
    writer.add_scalar("Metric", metric, episode_idx)
    writer.add_scalar("Num_of_conflicts", num_of_conflict, episode_idx)
    writer.add_scalar("Avg_num_of_conflicts", avg_num_of_conflict, episode_idx)
    writer.flush()
   
    
#animation = display_episode(frames)
#plt.show()


# --------------------------------------- #
# --------------- TESTING --------------- #
# --------------------------------------- #

######################
##### TEST DEBUG ####
#####################
# Reset environment and get initial observations for all agents
env.reset()
# Reset the rendering system
env_renderer.reset()

frame_step = 0
frames = []
score = 0
num_of_tests = 10

for test_episode in range(num_of_tests + 1):
    # Reset environment and get initial observations for all agents
    env.reset()
    # Reset the rendering system
    env_renderer.reset()
    
    score = 0
    
    metric = 0
    
    for step in range(max_steps):
        for a in range(env.get_num_agents()):
            update_values[a] = True
            action = policy.act(a, agent_obs[a])

            action_count[action] += 1
            actions_taken.append(action)
            action_dict.update({a: action})
            score += all_rewards[a]
            
        next_obs, all_rewards, done, info = env.step(action_dict)

        frame = env_renderer.render_env(show=False, show_observations=False, show_inactive_agents=False, show_predictions=False, return_image=True)
        frames.append(frame)
        frame_step += 1
        
        if done['__all__']:
            break

        if check_conflicts(env):
            break
        
    # metric near to 1 is great result
    for agent_handle in env.get_agent_handles():
        metric += env.calculate_metric_single_agent(timetable, agent_handle)
        

    tasks_finished = sum(done[idx] for idx in env.get_agent_handles())

    print()
    print(  'Test 0 concluded:'
            '\t 🏆 Score: {:.3f}'
            '\t Agent completed {}'
            '\t Metric {}'.format(
                score,
                tasks_finished,
                metric
            ), end=" ")

animation = display_episode(frames)
plt.show()


"""#################
##### TEST 1 ####
#################
# Reset environment and get initial observations for all agents
env.reset()
# Reset the rendering system
env_renderer.reset()
# Change the position of the interrupted agents
if example_training == 'training0': 
    env.agents[1].initial_position = (6,10)
    env.agents[2].initial_position = (5,10)
else:
    env.agents[1].initial_position = (5,10)

frame_step = 0
frames = []
score = 0

for step in range(max_steps):
    # Broken agents
    if example_training == 'training0': 
        make_a_deterministic_interruption(env.agents[1], max_steps)
        make_a_deterministic_interruption(env.agents[2], max_steps)
    else:
        make_a_deterministic_interruption(env.agents[1], max_steps)
    update_values[0] = True
    action = policy.act(agent_obs[0], eps = 0.01)

    action_count[action] += 1
    actions_taken.append(action)
    action_dict.update({0: action})
    
    next_obs, all_rewards, done, info = env.step(action_dict)
    
    score += all_rewards[0]

    frame = env_renderer.render_env(show=False, show_observations=False, show_inactive_agents=False, show_predictions=False, return_image=True)
    frames.append(frame)
    frame_step += 1
    
    if done['__all__']:
        break
    #break if the first agent has done
    if ((training_flag == 'training0') and (done[0] == True)) or \
        ((training_flag == 'training1') and (done[0] == True) and (done[1] == True)) or \
        ((training_flag == 'training1.1') and (done[0] == True) and (done[1] == True)):
        break

    if check_conflicts(env):
        break
    
# metric most possible near to 0
metric = calculate_metric(env, timetable)

tasks_finished = sum(done[idx] for idx in env.get_agent_handles())

print()
print(  'Test 1 concluded:'
        '\t 🏆 Score: {:.3f}'
        '\t Agent completed {}'
        '\t Metric {}'.format(
            score,
            tasks_finished,
            metric
        ), end=" ")


animation = display_episode(frames)
plt.show()



if example_training == 'training0':
    #################
    ##### TEST 2 ####
    #################
    # Reset the rendering system
    env_renderer.reset()
    # Reset environment and get initial observations for all agents
    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
    # Change the position of the interrupted agents
    env.agents[1].initial_position = (6,8)
    env.agents[2].initial_position = (-1,0)

    frame_step = 0
    frames = []
    score = 0

    for step in range(max_steps):
        # Broken agents
        if training_flag == 'training0':
            make_a_deterministic_interruption(env.agents[1], max_steps)
            env.agents[2].malfunction_handler.malfunction_down_counter = max_steps
        if training_flag == 'training1':
            make_a_deterministic_interruption(env.agents[2], max_steps)
            make_a_deterministic_interruption(env.agents[3], max_steps)
        if training_flag == 'training1.1':
            make_a_deterministic_interruption(env.agents[2], max_steps)

        update_values[0] = True
        action = policy.act(agent_obs[0])

        action_count[action] += 1
        actions_taken.append(action)
        action_dict.update({0: action})
        
        next_obs, all_rewards, done, info = env.step(action_dict)
        
        score += all_rewards[0]

        frame = env_renderer.render_env(show=False, show_observations=False, show_inactive_agents=True, show_predictions=False, return_image=True)
        frames.append(frame)
        frame_step += 1
        
        if done['__all__']:
            break
        #break if the first agent has done
        if ((training_flag == 'training0') and (done[0] == True)) or \
            ((training_flag == 'training1') and (done[0] == True) and (done[1] == True)) or \
            ((training_flag == 'training1.1') and (done[0] == True) and (done[1] == True)):
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

    animation = display_episode(frames)
    plt.show()



    #################
    ##### TEST 3 ####
    #################
    # Reset the rendering system
    env_renderer.reset()
    # Reset environment and get initial observations for all agents
    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
    # Change the position of the interrupted agents
    env.agents[1].initial_position = (6,14)
    env.agents[2].initial_position = (5,14)

    frame_step = 0
    frames = []
    score = 0

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

        update_values[0] = True
        action = policy.act(agent_obs[0])

        action_count[action] += 1
        actions_taken.append(action)
        action_dict.update({0: action})
        
        next_obs, all_rewards, done, info = env.step(action_dict)
        
        score += all_rewards[0]

        frame = env_renderer.render_env(show=False, show_observations=False, show_inactive_agents=True, show_predictions=False, return_image=True)
        frames.append(frame)
        frame_step += 1
        
        if done['__all__']:
            break
        #break if the first agent has done
        if ((training_flag == 'training0') and (done[0] == True)) or \
            ((training_flag == 'training1') and (done[0] == True) and (done[1] == True)) or \
            ((training_flag == 'training1.1') and (done[0] == True) and (done[1] == True)):
            break

    # metric most possible near to 0
    metric = calculate_metric(env, timetable)

    tasks_finished = sum(done[idx] for idx in env.get_agent_handles())

    print(  'Test 3 concluded:'
            '\t 🏆 Score: {:.3f}'
            '\t Agent completed {}'
            '\t Metric {}'.format(
                score,
                tasks_finished,
                metric
            ), end=" ")

    animation = display_episode(frames)
    plt.show()


    #################
    ##### TEST 4 ####
    #################
    # Reset the rendering system
    env_renderer.reset()
    # Reset environment and get initial observations for all agents
    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
    # Change the position of the interrupted agents
    env.agents[1].initial_position = (6,14)
    env.agents[2].initial_position = (-1,0)

    frame_step = 0
    frames = []
    score = 0

    for step in range(max_steps):
        # Broken agents
        if training_flag == 'training0':
            make_a_deterministic_interruption(env.agents[1], max_steps)
            env.agents[2].malfunction_handler.malfunction_down_counter = max_steps
        if training_flag == 'training1':
            make_a_deterministic_interruption(env.agents[2], max_steps)
            make_a_deterministic_interruption(env.agents[3], max_steps)
        if training_flag == 'training1.1':
            make_a_deterministic_interruption(env.agents[2], max_steps)

        update_values[0] = True
        action = policy.act(agent_obs[0])

        action_count[action] += 1
        actions_taken.append(action)
        action_dict.update({0: action})
        
        next_obs, all_rewards, done, info = env.step(action_dict)
        
        score += all_rewards[0]

        frame = env_renderer.render_env(show=False, show_observations=False, show_inactive_agents=True, show_predictions=False, return_image=True)
        frames.append(frame)
        frame_step += 1
        
        if done['__all__']:
            break
        #break if the first agent has done
        if ((training_flag == 'training0') and (done[0] == True)) or \
            ((training_flag == 'training1') and (done[0] == True) and (done[1] == True)) or \
            ((training_flag == 'training1.1') and (done[0] == True) and (done[1] == True)):
            break


    # metric most possible near to 0
    metric = calculate_metric(env, timetable)

    tasks_finished = sum(done[idx] for idx in env.get_agent_handles())

    print(  'Test 4 concluded:'
            '\t 🏆 Score: {:.3f}'
            '\t Agent completed {}'
            '\t Metric {}'.format(
                score,
                tasks_finished,
                metric
            ), end=" ")

    animation = display_episode(frames)
    plt.show()




    #################
    ##### TEST 5 ####
    #################
    # Reset the rendering system
    env_renderer.reset()
    # Reset environment and get initial observations for all agents
    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
    # Change the position of the interrupted agents
    env.agents[2].initial_position = (5,10)
    env.agents[1].initial_position = (-1,0)

    frame_step = 0
    frames = []
    score = 0

    for step in range(max_steps):
        # Broken agents
        if training_flag == 'training0':
            make_a_deterministic_interruption(env.agents[2], max_steps)
            env.agents[1].malfunction_handler.malfunction_down_counter = max_steps
        if training_flag == 'training1':
            make_a_deterministic_interruption(env.agents[2], max_steps)
            make_a_deterministic_interruption(env.agents[3], max_steps)
        if training_flag == 'training1.1':
            make_a_deterministic_interruption(env.agents[2], max_steps)

        update_values[0] = True
        action = policy.act(agent_obs[0])

        action_count[action] += 1
        actions_taken.append(action)
        action_dict.update({0: action})
        
        next_obs, all_rewards, done, info = env.step(action_dict)
        
        score += all_rewards[0]

        frame = env_renderer.render_env(show=False, show_observations=False, show_inactive_agents=True, show_predictions=False, return_image=True)
        frames.append(frame)
        frame_step += 1
        
        if done['__all__']:
            break
        #break if the first agent has done
        if ((training_flag == 'training0') and (done[0] == True)) or \
            ((training_flag == 'training1') and (done[0] == True) and (done[1] == True)) or \
            ((training_flag == 'training1.1') and (done[0] == True) and (done[1] == True)):
            break

    # metric most possible near to 0
    metric = calculate_metric(env, timetable)

    tasks_finished = sum(done[idx] for idx in env.get_agent_handles())

    print(  'Test 5 concluded:'
            '\t 🏆 Score: {:.3f}'
            '\t Agent completed {}'
            '\t Metric {}'.format(
                score,
                tasks_finished,
                metric
            ), end=" ")

    animation = display_episode(frames)
    plt.show()




    #################
    ##### TEST 6 ####
    #################
    # Reset the rendering system
    env_renderer.reset()
    # Reset environment and get initial observations for all agents
    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)
    # Change the position of the interrupted agents
    env.agents[2].initial_position = (5,16)
    env.agents[1].initial_position = (-1,0)

    frame_step = 0
    frames = []
    score = 0

    for step in range(max_steps):
        # Broken agents
        if training_flag == 'training0':
            make_a_deterministic_interruption(env.agents[2], max_steps)
            env.agents[1].malfunction_handler.malfunction_down_counter = max_steps
        if training_flag == 'training1':
            make_a_deterministic_interruption(env.agents[2], max_steps)
            make_a_deterministic_interruption(env.agents[3], max_steps)
        if training_flag == 'training1.1':
            make_a_deterministic_interruption(env.agents[2], max_steps)

        update_values[0] = True
        action = policy.act(agent_obs[0])

        action_count[action] += 1
        actions_taken.append(action)
        action_dict.update({0: action})
        
        next_obs, all_rewards, done, info = env.step(action_dict)
        
        score += all_rewards[0]

        frame = env_renderer.render_env(show=False, show_observations=False, show_inactive_agents=True, show_predictions=False, return_image=True)
        frames.append(frame)
        frame_step += 1
        
        if done['__all__']:
            break
        #break if the first agent has done
        if ((training_flag == 'training0') and (done[0] == True)) or \
            ((training_flag == 'training1') and (done[0] == True) and (done[1] == True)) or \
            ((training_flag == 'training1.1') and (done[0] == True) and (done[1] == True)):
            break

    # metric most possible near to 0
    metric = calculate_metric(env, timetable)

    tasks_finished = sum(done[idx] for idx in env.get_agent_handles())

    print(  'Test 6 concluded:'
            '\t 🏆 Score: {:.3f}'
            '\t Agent completed {}'
            '\t Metric {}'.format(
                score,
                tasks_finished,
                metric
            ), end=" ")

    animation = display_episode(frames)
    plt.show()"""