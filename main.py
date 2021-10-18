import time
import numpy as np
import os
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
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
# Import the schedule generators
from flatland.envs.custom_schedule_generator import custom_schedule_generator
from flatland.envs.timetables_utils import action_to_do, check_train_in_station, control_timetable
# Import the different structures needed
from configuration import railway_example, stations, timetable_example
# Import the agent class
from flatland.envs.agent import RandomAgent



# Flag active in case of interruptions
interruption = False

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

timetable = timetable_example

check_train_in_station(timetable)

print('=====================================================')
print('==================  TIMETABLE  ======================')
print('=====================================================')
for i in range(len(timetable)):
	print(timetable[i])

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

control_timetable(timetable,transition_map_example)

# We can now initiate the schedule generator with the given speed profiles
schedule_generator_custom = custom_schedule_generator(timetable = timetable)

actions_scheduled = action_to_do(timetable, transition_map_example)

# DEBUG
for i in range(len(actions_scheduled)):
	print()
	print(actions_scheduled[i])
	print()


TreeObservation = GlobalObsForRailEnv()

env = RailEnv(  width= widht,
				height= height,
				rail_generator = rail_custom,
				schedule_generator=schedule_generator_custom,
				number_of_agents= num_of_agents,
				obs_builder_object=TreeObservation,
				remove_agents_at_target=True,
				record_steps=True
				)


env.reset()

env_renderer = RenderTool(env,
						  screen_height=1080*2,
						  screen_width=1080*2)  # Adjust these parameters to fit your resolution

# This thing is importand for the RL part, initialize the agent with (state, action) dimension
# Initialize the agent with the parameters corresponding to the environment and observation_builder
controller = RandomAgent(218, env.action_space[0])

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

observations ,rewards, dones ,information = env.step(action_dict)

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

# How many episodes
n_trials = 3

os.makedirs("output/frames", exist_ok=True)

for trials in range(1, n_trials + 1):

	# Reset environment and get initial observations for all agents
	obs, info = env.reset()
	for idx in range(env.get_num_agents()):
		tmp_agent = env.agents[idx]
		tmp_agent.speed_data["speed"] = 1 / (idx + 1)
	env_renderer.reset()
	# Here you can also further enhance the provided observation by means of normalization
	# See training navigation example in the baseline repository

	score = 0
	# Run episode (one day long, 1 step is 1 minute)
	for step in range(1440):

		env_renderer.gl.save_image("output/frames/flatland_frame_step_{:04d}.bmp".format(step))

	# Here define the actions to do

		# Chose an action for each agent in the environment
		# If not interruption, the actions to do are stored in a matrix
		#       - each row of the matrix is a train
		#       - each column represent the action the train has to do at each time instant
		for a in range(env.get_num_agents()):
			if step >= timetable[a][1][0]:
				if not interruption and (step - timetable[a][1][0]) < len(actions_scheduled[a]):
					action = actions_scheduled[a][step - timetable[a][1][0]]
				# choose random from all the possible actions
				else:
					action = np.random.choice([RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT, RailEnvActions.MOVE_LEFT, 
						RailEnvActions.STOP_MOVING, RailEnvActions.REVERSE])
				action_dict.update({a: action})
		# Environment step which returns the observations for all agents, their corresponding
		# reward and whether their are done
		next_obs, all_rewards, done, _ = env.step(action_dict)
		env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

		# Update replay buffer and train agent
		for a in range(env.get_num_agents()):
			controller.step((obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
			score += all_rewards[a]
		obs = next_obs.copy()
		if done['__all__']:
			break
	print('Episode Nr. {}\t Score = {}'.format(trials, score))
