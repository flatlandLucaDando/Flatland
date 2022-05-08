import time
import numpy as np
import os
from argparse import Namespace
from pprint import pprint
from random import *
from datetime import datetime
from statistics import mean

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
from flatland.envs.rail_env_utils import delay_a_train, make_interruption
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
from flatland.utils.observation_utils import normalize_global_observation, normalize_observation
from reinforcement_learning.dddqn_policy import DDDQNPolicy
# Import training and observation parameters
from parameters import training_params, obs_params


import matplotlib.pyplot as plt
import matplotlib.animation
plt.rcParams["animation.html"] = "jshtml"



###### TRAINING PARAMETERS #######
n_episodes = 15000
eps_start = 1
eps_end = 0.01
eps_decay = 0.99995
max_steps = 300          # 1440 one day
checkpoint_interval = 100

 # Unique ID for this training
now = datetime.now()
training_id = now.strftime('%y%m%d%H%M%S')


#########################################################
# Parameters that should change the results:

# eps_decay 
# step_maximum_penality
# % used in the sparse reward
#########################################################

render = True

######### FLAGS ##########
# Flag for the first training
training_flag = example_training
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
# flag to decide if save or not replay buffer
save_replay_buffer = True


####################################################################

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
        

def format_action_prob(action_probs):
    action_probs = np.round(action_probs, 3)
    actions = ["↻", "←", "↑", "→", "◼", "↓"]

    buffer = ""
    for action, action_prob in zip(actions, action_probs):
        buffer += action + " " + "{:.3f}".format(action_prob) + " "

    return buffer


def evaluate_policy(environment, environment_renderer, tree_observation, policy, train_params, obs_params):
    #####################
    ## Evaluate policy ##
    #####################
    
    num_of_tests = train_params.n_evaluation_episodes
    max_steps = environment._max_episode_steps
    
    # Reset environment and get initial observations for all agents
    environment.reset()
    # Reset the rendering system
    environment_renderer.reset()

    frame_step = 0
    frames = []
    score = 0
    action_dict = dict()
    num_of_tests = 1

    total_steps = 0

    # Flag to choose if save video of tests or not
    if episode_idx % 500 == 0:
        video_save = True
    else:
        video_save = False

    test_id = now.strftime('%y-%m-%d-%H,%M')
    
    os.makedirs("output/training_n_" + test_id, exist_ok=True)
    if video_save:
        os.makedirs("output/training_n_" + test_id + "/frames", exist_ok=True)
        os.makedirs("output/training_n_" + test_id + "/frames/_episode_n_" + str(episode_idx), exist_ok=True)

    test_directory_name = "output/training_n_" + test_id + "/frames/_episode_n_" + str(episode_idx) + "/"

    for test_episode in range(num_of_tests):
        
        # Reset environment and get initial observations for all agents
        environment.reset()
        # Reset the rendering system
        environment_renderer.reset()
        # Reset the scores and metrics
        score = 0.0
        metric = 0.0
        # Taking the first observation
        agent_obs, info = environment.reset(regenerate_rail=True, regenerate_schedule=True)
        final_step = 0
        counter_station = 0
        
        policy.start_episode(train=False)
        
        for step in range(max_steps):
            
            make_interruption((6,7), env)
            environment.interruption = True
             
            
            if video_save:
                # Check if this work with multiple tests in the same for loop
                environment_renderer.gl.save_image(test_directory_name + "flatland_episode_and_step_{:04d}.bmp".format(total_steps))
            
            policy.start_step(train=False)
            
            for a in range(environment.get_num_agents()):
                if environment.agents[a].position in environment.station_positions and environment.stop_station_time[a] > 0 \
                    and step > environment.agents[a].earliest_departure + 2:
                    # AGGIUNGERE CONTROLLI SUGLI ORARI ARRIVO - PARTENZA
                    action = RailEnvActions.STOP_MOVING
                    environment.stop_station_time[a] -= 1
                else:
                    agent_obs[a] = normalize_observation(agent_obs[agent], observation_tree_depth, observation_radius=observation_radius)
                    action = policy.act(a, agent_obs[a], eps=0.0)
                # Restart the stop_timer
                if environment.stop_station_time[a] == 0:
                    if counter_station >= 4:
                        environment.stop_station_time[a] = 2  # station.min_waiting_time # For now the timer is defaulted
                        counter_station = 0
                    counter_station += 1
                action_dict.update({a: action})
            policy.end_step(train=False)    
            agent_obs, all_rewards, done, info = environment.step(action_dict)
            
            for a in range(environment.get_num_agents()):
                score += all_rewards[a]
                if environment.agents[a].state == TrainState.DONE:
                    done[a] = True
            
            environment_renderer.render_env(show=False, show_observations=False, show_inactive_agents=False, show_predictions=False, return_image=True)
            
            total_steps += 1
            
            if done['__all__']:
                break
        
        policy.end_episode(train=False)    
        # metric near to 1 is great result
        for agent_handle in range(environment.number_of_agents):
            metric += environment.calculate_metric_single_agent(timetable, agent_handle)
            
        metric = metric/environment.number_of_agents
        
        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        
        string_with_rewards = '\n \n Episode number ' + str(episode_idx) + ' evaluation concluded:' + '\n Score: ' + str(score) + \
            '\n Agent completed: ' + str(tasks_finished) + '\n Metric: ' + str(metric)

        print(string_with_rewards)
        
        # Save the results in a txt file
        f = open("output/training_n_" + test_id + "/evaluation_policy.txt", "a+")
        f.write(string_with_rewards)
        f.close()

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

# divide_trains_in_station_rails(timetable, transition_map_example)  # WE HAVE A DOUBLE RAIL EVERYWHERE. WE DOESN'T CONSIDER CONFLICT. NON CI IMPORTA PIÙ

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

"""policy.load("checkpoints/colab/1600.pth")
policy.load_replay_buffer("replay_buffers/colab/1600.pkl")
policy.test()"""


os.makedirs("output/frames", exist_ok=True)

for episode_idx in range(n_episodes + 1):
    
    deterministic_interruption_activation = False

    reset_timer = Timer()
    policy_start_episode_timer = Timer()
    policy_start_step_timer = Timer()
    policy_act_timer = Timer()
    env_step_timer = Timer()
    policy_shape_reward_timer = Timer()
    policy_step_timer = Timer()
    policy_end_step_timer = Timer()
    policy_end_episode_timer = Timer()
    total_episode_timer = Timer()
    
    total_episode_timer.start()

    # Reset environment
    reset_timer.start()
    # Reset environment and get initial observations for all agents
    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)    
    reset_timer.end()
    for idx in range(env.get_num_agents()):
        tmp_agent = env.agents[idx]
        tmp_agent.speed_counter.speed = 1 / 10  # TODO rigestisci le velocità iniziali
    env_renderer.reset()
    policy.reset(env)
    reset_timer.end()

    if train_params.render:
        env_renderer.set_new_rail()

    score = 0
    nb_steps = 0
    metric = 0
    counter_station = 0
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

    policy_start_episode_timer.start()
    policy.start_episode(train=True)
    policy_start_episode_timer.end()
    
    # Run episode (one day long, 1 step is 1 minute) 1440
    for step in range(max_steps - 1):
        if video_save:
            env_renderer.gl.save_image("output/frames/flatland_frame_step_{:04d}.bmp".format(step))
            
    # Here define the actions to do
        # Broken agents
        # INTERRUPTION --------------------------------------------------------------------------------------------
        make_interruption((6,7), env)
        env.interruption = True

        # policy.start_step ---------------------------------------------------------------------------------------
        policy_start_step_timer.start()
        policy.start_step(train=True)
        policy_start_step_timer.end()
        
        # policy.act ----------------------------------------------------------------------------------------------
        policy_act_timer.start()
        action_dict = {}
        
        # Chose an action for each agent in the environment
        # If not interruption, the actions to do are stored in a matrix
        #       - each row of the matrix is a train
        #       - each column represent the action the train has to do at each time instant
        for a in range(env.get_num_agents()):
            # An action is not required if the train hasn't joined the railway network,
            # if it already reached its target, or if is currently malfunctioning.    
            if env.agents[a].state == TrainState.MALFUNCTION:
                action = 0
                action_dict.update({a: action})
                continue
            if env.agents[a].position in env.station_positions and env.stop_station_time[a] > 0 \
                    and step > env.agents[a].earliest_departure + 2:
                # AGGIUNGERE CONTROLLI SUGLI ORARI ARRIVO - PARTENZA
                action = RailEnvActions.STOP_MOVING
                env.stop_station_time[a] -= 1                 
            else:
                update_values[a] = True
                action = policy.act(a, agent_obs[a], eps=eps_start)
                action_count[action] += 1
                actions_taken.append(action)
                """ else:
                # An action is not required if the train hasn't joined the railway network,
                # if it already reached its target, or if is currently malfunctioning.
                update_values[a] = False
                action = 0        """
            # Restart the stop_timer
            if env.stop_station_time[a] == 0:
                if counter_station >= 4:
                    env.stop_station_time[a] = 2  # station.min_waiting_time # For now the timer is defaulted
                    counter_station = 0
                    env.reverse_once = True
                counter_station += 1

            action_dict.update({a: action})

        policy_act_timer.end()
        
        # policy.end_step -----------------------------------------------------------------------------------------
        policy_end_step_timer.start()
        policy.end_step(train=True)
        policy_end_step_timer.end()

        # Environment step which returns the observations for all agents, their corresponding
        # reward and whether their are done
        # Environment step ----------------------------------------------------------------------------------------
        env_step_timer.start()
        next_obs, all_rewards, done, info = env.step(action_dict)
        for agent_handle in env.get_agent_handles():
                done[agent_handle] = (env.agents[agent_handle].state == TrainState.DONE)
        env_step_timer.end()
        
        # policy.shape_reward -------------------------------------------------------------------------------------
        policy_shape_reward_timer.start()

        # The might requires a policy based transformation
        for agent_handle in env.get_agent_handles():
            all_rewards[agent_handle] = policy.shape_reward(agent_handle,
                                                            action_dict[agent_handle],
                                                            agent_obs[agent_handle],
                                                            all_rewards[agent_handle],
                                                            done[agent_handle])
        policy_shape_reward_timer.end()
        
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
                    policy_step_timer.start()
                    policy.step(agent_handle,
                            agent_prev_obs[agent_handle],
                            agent_prev_action[agent_handle],
                            all_rewards[agent_handle],
                            agent_obs[agent_handle],
                            done[agent_handle])
                    policy_step_timer.end()
                    
                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]
                score += all_rewards[agent]
                # Preprocess the new observations
                agent_obs[agent] = normalize_observation(next_obs[agent], observation_tree_depth, observation_radius=observation_radius)
                """if tree_observer:
                    agent_obs[agent] = normalize_observation(obs[agent], observation_tree_depth, observation_radius=observation_radius)
                else:
                    agent_obs[agent] = normalize_global_observation(obs[agent])"""

            nb_steps = step
        else:
            for a in range(env.get_num_agents()):
                controller.step((obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
                score += all_rewards[a]
                
        if done['__all__']:
            break
        #break if the first agent has done
        
        # Exit from the simulation if the first agent has ended
        if done[0]:
            break
        
        """if check_conflicts(env):
            break"""
       
    # policy.end_episode
    policy_end_episode_timer.start()
    policy.end_episode(train=True)
    policy_end_episode_timer.end()
    
    # Epsilon decay
    eps_start = max(eps_end, eps_decay * eps_start)
    
    total_episode_timer.end()
        
    print()
    print('Episode Nr. {}\t Score = {}'.format(episode_idx, score))
    
    # metric near to 1 is great result
    for agent_handle in range(env.number_of_agents):   # 1 is the interrupting agent...we don't consider it
        metric += env.calculate_metric_single_agent(timetable, agent_handle)
    
    metric = metric/env.number_of_agents               # TODO generalizza
    
    if multi_agent:
        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)

        # Collect information about training
        tasks_finished = sum(done[idx] for idx in env.get_agent_handles())
        completion = tasks_finished / max(1, env.get_num_agents())
        normalized_score = score / (max_steps * env.get_num_agents())
        action_probs = action_count / np.sum(action_count)
        action_count = [1] * action_size
        
        #avg_num_of_conflict = env.num_of_conflict / (episode_idx + 1)
        
        smoothing = 0.99
        smoothed_normalized_score = smoothed_normalized_score * smoothing + normalized_score * (1.0 - smoothing)
        smoothed_completion = smoothed_completion * smoothing + completion * (1.0 - smoothing)

        # Print logs
        if episode_idx % checkpoint_interval == 0:
            os.makedirs("checkpoints", exist_ok=True)
            os.makedirs("checkpoints/training_n_" + training_id, exist_ok=True)
            policy.save('checkpoints/training_n_' + training_id + '/' + str(episode_idx) + '.pth')
            if save_replay_buffer:
                os.makedirs("replay_buffers", exist_ok=True)
                os.makedirs("replay_buffers/training_n_" + training_id, exist_ok=True)
                policy.save_replay_buffer(
                'replay_buffers/training_n_' + training_id + '/' + str(episode_idx) + '.pkl')

            if train_params.render:
                env_renderer.close_window()

        print(
            '\r🚂 Episode {}'
            '\t 🏆 Score: {:.3f}'
            ' Avg: {:.3f}'
            '\t 💯 Done: {}%'
            ' Avg: {:.3f}%'
            #'\t Num of conflicts: {}'
            #' Avg: {:.3f}'
            '\t 🎲 Epsilon: {:.3f} '
            '\t 🔀 Action Probs: {}'
            '\t Metric: {}'.format(
                episode_idx,
                normalized_score,
                smoothed_normalized_score,
                100 * completion,
                100 * smoothed_completion,
                #env.num_of_conflict,
                #avg_num_of_conflict,
                eps_start,
                format_action_prob(action_probs),
                metric
            ), end=" ")
        print()

    interruption = False
    
    writer.add_scalar("Dense Reward", env.dense_score, episode_idx)
    writer.add_scalar("Sparse Reward", env.sparse_score, episode_idx)
    writer.add_scalar("Metric", metric, episode_idx)
    writer.flush()
     
    if episode_idx % 50 == 0 and episode_idx != 0:
        evaluate_policy(env, env_renderer, next_obs, policy, training_params, obs_params)
   
    
#animation = display_episode(frames)
#plt.show()

"""
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
num_of_tests = 1

total_steps = 0

# Flag to choose if save video of tests or not
video_save = True

test_id = now.strftime('%y-%m-%d - %H,%M')

if video_save:
    os.makedirs("output/frames/" + test_id, exist_ok=True)

test_directory_name = "output/frames/" + test_id + "/"

for test_episode in range(num_of_tests + 1):
    
    # Reset environment and get initial observations for all agents
    env.reset()
    # Reset the rendering system
    env_renderer.reset()
    
    score = 0
    
    metric = 0
    
    for step in range(max_steps):
        
        if video_save:
            # Check if this work with multiple tests in the same for loop
            env_renderer.gl.save_image(test_directory_name + "flatland_episode_and_step_{:04d}.bmp".format(total_steps))
        
        for a in range(env.get_num_agents()):
            update_values[a] = True
            action = policy.act(a, agent_obs[a], eps = 0)

            action_count[action] += 1
            actions_taken.append(action)
            action_dict.update({a: action})
            score += all_rewards[a]
            
        next_obs, all_rewards, done, info = env.step(action_dict)

        frame = env_renderer.render_env(show=False, show_observations=False, show_inactive_agents=False, show_predictions=False, return_image=True)
        frames.append(frame)
        frame_step += 1
        
        total_steps += 1
        
        if done['__all__']:
            break
        
    # metric near to 1 is great result
    for agent_handle in env.get_agent_handles():
        metric += env.calculate_metric_single_agent(timetable, agent_handle)
        
    metric = metric/env.number_of_agents
    
    tasks_finished = sum(done[idx] for idx in env.get_agent_handles())

    print()
    print(  'Test ' + str(test_episode) + ' concluded:'
            '\t 🏆 Score: {:.3f}'
            '\t Agent completed {}'
            '\t Metric {}'.format(
                score,
                tasks_finished,
                metric
            ), end=" ")

animation = display_episode(frames)
plt.show()"""
