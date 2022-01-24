"""Schedule generators (railway undertaking, "EVU")."""
from typing import Tuple, List, Callable, Mapping, Optional, Any

from enum import IntEnum
from numpy.random.mtrand import RandomState
 
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.schedule_utils import Schedule 

from flatland.envs.line_generators import check_rail_road_direction

AgentPosition = Tuple[int, int]
ScheduleGenerator = Callable[[GridTransitionMap, int, Optional[Any], Optional[int]], Schedule]



class RailEnvActions(IntEnum):
	DO_NOTHING = 0  # implies change of direction in a dead-end!
	MOVE_LEFT = 1
	MOVE_FORWARD = 2
	MOVE_RIGHT = 3
	STOP_MOVING = 4
	REVERSE = 5

	@staticmethod
	def to_char(a: int):
		return {
			0: 'B',
			1: 'L',
			2: 'F',
			3: 'R',
			4: 'S',
			5: 'G'
		}[a]


def custom_schedule_generator(timetable, speed_ratio_map: Mapping[float, float] = None, seed: int = 1) -> ScheduleGenerator:

#class Custom_schedule_generator(BaseSchedGen):
	"""

	This is a custom schedule generator, create a schedule with the timetable, and the station where the trains should pass
	"""

	def generate_custom(rail: GridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0,
				  np_random: RandomState = None) -> Schedule:
		"""

		The generator that assigns tasks to all the agents
		:param rail: Rail infrastructure given by the rail_generator
		:param num_agents: Number of agents to include in the schedule
		:param hints: Hints provided by the rail_generator These include positions of start/target positions
		:param num_resets: How often the generator has been reset.
		:return: Returns the generator to the rail constructor
		"""

		train_stations = hints['train_stations']
		city_positions = hints['city_positions']
		city_orientation = hints['city_orientations']
		max_num_agents = hints['num_agents']

		if num_agents > max_num_agents:
			num_agents = max_num_agents
			warnings.warn("Too many agents! Changes number of agents.")
		# Place agents and targets within available train stations
		agents_position = []
		agents_target = []
		agents_direction = []

		# Define the agent positions
		# TODO capisci se così va bene o no
		for agent_i in range (len(timetable)):
			agents_position.append(timetable[agent_i][3][0])
			agents_target.append(timetable[agent_i][3][-1])


		# Define the direction of the trains based on the rail they occupy
		# Input --> the topology of the network, the position of the trains
		# Output --> an array with the directions of the trains
		# DIRECTIONS: 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT

		agents_direction = check_rail_road_direction(rail, timetable)


		_runtime_seed = seed + num_resets

		if speed_ratio_map:
			speeds = speed_initialization_helper(num_agents, speed_ratio_map, seed=_runtime_seed, np_random=np_random)
		else:
			speeds = [1.0] * len(agents_position)

		# We add multiply factors to the max number of time steps to simplify task in Flatland challenge.
		# These factors might change in the future.
		timedelay_factor = 4
		alpha = 2
		max_episode_steps = 1000

		#print(agents_position, agents_target, agents_direction)

		return Schedule(agent_positions=agents_position, agent_directions=agents_direction,
						agent_targets=agents_target, agent_speeds=speeds, agent_malfunction_rates=None,
						max_episode_steps=max_episode_steps)

	return generate_custom  #(station_to_traverse = [(21, 37), (15, 51)])