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



###### TRAINING PARAMETERS #######
n_episodes = 12000
eps_start = 1
eps_end = 0.01
eps_decay = 0.99992

max_steps = 250           # 1440 one day
checkpoint_interval = 100

mean_tolerance = 1     # Tolerance to compare the mean of the two windows of episodes
                       # this is important to discover if a plateau is present in the rewards distribution over the episodes
tolerance_of_conflict = 0.35     # Threshold of maximum percentage of conflicts that we accept                       

num_of_plateau = 0

plateau_window = 60

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

render = True 

######### FLAGS ##########
# Flag for the first training
training_flag = example_training
# Flag active in case of interruptions
interruption = False
# Flag to select the agent ----> multi agent or external controller
multi_agent = True
# Flag to save the video or not
video_save = True
# Flag to applicate RL also in case of no interruptions
reinforcemente_learning = True
# Flag to output different things important for the debug
debug = False
# flag to select the tree observer
tree_observer = True
# flag to decide if save or not replay buffer
save_replay_buffer = True


####################################################################
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

"""actions_scheduled[4][44] = RailEnvActions.MOVE_RIGHT
actions_scheduled[4][36] = RailEnvActions.MOVE_FORWARD
actions_scheduled[4][37] = RailEnvActions.MOVE_FORWARD
actions_scheduled[4][38] = RailEnvActions.STOP_MOVING
actions_scheduled[4][39] = RailEnvActions.STOP_MOVING"""
actions_scheduled[2][68] = RailEnvActions.MOVE_FORWARD
actions_scheduled[3][24] = RailEnvActions.MOVE_FORWARD
actions_scheduled[3][25] = RailEnvActions.MOVE_FORWARD
actions_scheduled[3][86] = RailEnvActions.MOVE_FORWARD
actions_scheduled[3][87] = RailEnvActions.MOVE_FORWARD
actions_scheduled[3][88] = RailEnvActions.MOVE_FORWARD
actions_scheduled[3][89] = RailEnvActions.MOVE_FORWARD
actions_scheduled[5][48] = RailEnvActions.MOVE_FORWARD
for i in range(5):
    actions_scheduled[5][35+i] = RailEnvActions.MOVE_RIGHT

# DEBUG
if debug:
    for i in range(len(actions_scheduled)):
        print()
        print(actions_scheduled[i])
        print()

    time.sleep(3)

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



# Lets try to enter with all of these agents at the same time
action_dict = dict()

# Now that you have seen these novel concepts that were introduced you will realize that agents don't need to take
# an action at every time step as it will only change the outcome when actions are chosen at cell entry.
# Therefore the environment provides information about what agents need to provide an action in the next step.
# You can access this in the following way.

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
#avg_num_of_conflict = 0

score_mean = [0] * plateau_window


# policy.load("checkpoints/training_n_220224144516/8500.pth")
# policy.load_replay_buffer("replay_buffers/training_n_220224144516/8500.pkl")


os.makedirs("output/pippo", exist_ok=True)

for episode_idx in range(n_episodes + 1):
    
    env.agents[5].initial_position = (7,13)
    env.agents[1].initial_position = (13, 48)
    
    deterministic_interruption_activation = False
    
    if render:
        env_renderer.set_new_rail()

    # Run episode (one day long, 1 step is 1 minute) 1440
    for step in range(max_steps - 1):
        action_dict = {}
        
        # Chose an action for each agent in the environment
        # If not interruption, the actions to do are stored in a matrix
        #       - each row of the matrix is a train
        #       - each column represent the action the train has to do at each time instant
        for a in range(env.get_num_agents()):
            
            action = 0
            
            if env.agents[a].state == TrainState.DONE:
                done[a] = True
            done['__all__'] &= done[a]
            
            if step >= timetable[a][1][0]:
                # Normal plan to follow
                if (step - timetable[a][1][0]) < len(actions_scheduled[a]):
                    action = actions_scheduled[a][step - timetable[a][1][0]]
            action_dict.update({a: action})
            
        if done['__all__']:
            break
            
        next_obs, all_rewards, done, info = env.step(action_dict)

        # Render an episode at some interval
        #frame = env_renderer.render_env(show=False, show_observations=False, show_inactive_agents=False, show_predictions=False, return_image=True)
        #frames.append(frame)
        if render:
            env_renderer.render_env(
                    show=True, show_observations = False, frames = True, episode = True, step = True
                )   
        if video_save:
            env_renderer.gl.save_image("output/pippo/flatland_episode_and_step_{:04d}.bmp".format(step))  
            
    interruption = False