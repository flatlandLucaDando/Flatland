"""
Definition of the RailEnv environment.
"""
from hashlib import new
import random
from re import I, T
from tkinter import N

from typing import List, Optional, Dict, Tuple

import numpy as np
from gym.utils import seeding

from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4 import Grid4Transitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env_action import RailEnvActions
from flatland.core.grid.grid4_astar import a_star

from flatland.envs import malfunction_generators as mal_gen
from flatland.envs import rail_generators as rail_gen
from flatland.envs import line_generators as line_gen
from flatland.envs.timetable_generators import timetable_generator
from flatland.envs import persistence
from flatland.envs import agent_chains as ac

from flatland.envs.observations import GlobalObsForRailEnv

from flatland.envs.step_utils.states import TrainState, StateTransitionSignals
from flatland.envs.step_utils.transition_utils import check_valid_action, check_reverse_action
from flatland.envs.step_utils import action_preprocessing
from flatland.envs.step_utils import env_utils
from flatland.utils.fast_methods import fast_count_nonzero, fast_argmax, fast_isclose

from structures_rail import av_line

from configuration import example_training, timetable_example

# Penalities 
maximum_step_penalty = -0.08     # a step is time passing, so a penality for each step is needed
minimum_step_penalty = -0.008     # 

stop_penality = 0                 # penalty for stopping a moving agent
reverse_penality = 0          # penalty for reversing the march of an agent
skip_penality = 0                 # penalty for skipping a station
conflict_penalty = 0             # penalty when two agents have a conflict (deadlock)
target_not_reached_penalty = 0  # penalty for not reaching the final target (depot)
default_skip_penalty = 2000
not_spawning_penalty = -2

cancellation_factor = 1
cancellation_time_buffer = 0

reinforcement_learning = True

interval_to_calculate_step_reward = 5

metric_threshold = 0.7
maximum_reward = 15

target_reward = 0         # reward for an agent reaching his final target
station_passage_reward = 0 # reward for an agent reaching intermediate station, the reward is wheighted with the delay of the agent

# Flag for the training
training = example_training

class RailEnv(Environment):
    """
    RailEnv environment class.

    RailEnv is an environment inspired by a (simplified version of) a rail
    network, in which agents (trains) have to navigate to their target
    locations in the shortest time possible, while at the same time cooperating
    to avoid bottlenecks.

    The valid actions in the environment are:

     -   0: do nothing (continue moving or stay still)
     -   1: turn left at switch and move to the next cell; if the agent was not moving, movement is started
     -   2: move to the next cell in front of the agent; if the agent was not moving, movement is started
     -   3: turn right at switch and move to the next cell; if the agent was not moving, movement is started
     -   4: stop moving
     -   5: invert the direction of march

    Moving forward in a dead-end cell makes the agent turn 180 degrees and step
    to the cell it came from.

    The actions of the agents are executed in order of their handle to prevent
    deadlocks and to allow them to learn relative priorities.

    Reward Function:

    It costs each agent a step_penalty for every time-step taken in the environment. Independent of the movement
    of the agent. Currently all other penalties such as penalty for stopping, starting and invalid actions are set to 0.

    alpha = 1
    beta = 1
    Reward function parameters:

    - invalid_action_penalty = 0
    - step_penalty = -alpha
    - global_reward = beta
    - epsilon = avoid rounding errors
    - stop_penalty = 0  # penalty for stopping a moving agent
    - start_penalty = 0  # penalty for starting a stopped agent

    Stochastic malfunctioning of trains:
    Trains in RailEnv can malfunction if they are halted too often (either by their own choice or because an invalid
    action or cell is selected.

    Every time an agent stops, an agent has a certain probability of malfunctioning. Malfunctions of trains follow a
    poisson process with a certain rate. Not all trains will be affected by malfunctions during episodes to keep
    complexity managable.

    TODO: currently, the parameters that control the stochasticity of the environment are hard-coded in init().
    For Round 2, they will be passed to the constructor as arguments, to allow for more flexibility.

    """
    
    cancellation_factor = 1
    cancellation_time_buffer = 0

    def __init__(self,
                 width,
                 height,
                 max_episode_steps,
                 rail_generator=None,
                 line_generator=None,  # : line_gen.LineGenerator = line_gen.random_line_generator(),
                 number_of_agents=2,
                 obs_builder_object: ObservationBuilder = GlobalObsForRailEnv(),
                 malfunction_generator_and_process_data=None,  # mal_gen.no_malfunction_generator(),
                 malfunction_generator=None,
                 remove_agents_at_target=True,
                 random_seed=None,
                 record_steps=False,
                 close_following=True,
                 ):
        """
        Environment init.

        Parameters
        ----------
        rail_generator : function
            The rail_generator function is a function that takes the width,
            height and agents handles of a  rail environment, along with the number of times
            the env has been reset, and returns a GridTransitionMap object and a list of
            starting positions, targets, and initial orientations for agent handle.
            The rail_generator can pass a distance map in the hints or information for specific line_generators.
            Implementations can be found in flatland/envs/rail_generators.py
        line_generator : function
            The line_generator function is a function that takes the grid, the number of agents and optional hints
            and returns a list of starting positions, targets, initial orientations and speed for all agent handles.
            Implementations can be found in flatland/envs/line_generators.py
        width : int
            The width of the rail map. Potentially in the future,
            a range of widths to sample from.
        height : int
            The height of the rail map. Potentially in the future,
            a range of heights to sample from.
        number_of_agents : int
            Number of agents to spawn on the map. Potentially in the future,
            a range of number of agents to sample from.
        obs_builder_object: ObservationBuilder object
            ObservationBuilder-derived object that takes builds observation
            vectors for each agent.
        remove_agents_at_target : bool
            If remove_agents_at_target is set to true then the agents will be removed by placing to
            RailEnv.DEPOT_POSITION when the agent has reach it's target position.
        random_seed : int or None
            if None, then its ignored, else the random generators are seeded with this number to ensure
            that stochastic operations are replicable across multiple operations
        """
        super().__init__()

        if malfunction_generator_and_process_data is not None:
            print("DEPRECATED - RailEnv arg: malfunction_and_process_data - use malfunction_generator")
            self.malfunction_generator, self.malfunction_process_data = malfunction_generator_and_process_data
        elif malfunction_generator is not None:
            self.malfunction_generator = malfunction_generator
            # malfunction_process_data is not used
            # self.malfunction_generator, self.malfunction_process_data = malfunction_generator_and_process_data
            self.malfunction_process_data = self.malfunction_generator.get_process_data()
        # replace default values here because we can't use default args values because of cyclic imports
        else:
            self.malfunction_generator = mal_gen.NoMalfunctionGen()
            self.malfunction_process_data = self.malfunction_generator.get_process_data()
        
        self.number_of_agents = number_of_agents

        if rail_generator is None:
            rail_generator = rail_gen.sparse_rail_generator()
        self.rail_generator = rail_generator
        if line_generator is None:
            line_generator = line_gen.sparse_line_generator()
        self.line_generator = line_generator

        self.rail: Optional[GridTransitionMap] = None
        self.width = width
        self.height = height

        self.remove_agents_at_target = remove_agents_at_target

        self.obs_builder = obs_builder_object
        self.obs_builder.set_env(self)

        self._max_episode_steps: Optional[int] = max_episode_steps
        self._elapsed_steps = 0

        self.obs_dict = {}
        self.rewards_dict = {}
        self.dev_obs_dict = {}
        self.dev_pred_dict = {}
        
        self.cell_interrupted = []
        
        # Flag that tell the environment to increase the conflict penalty
        self.increase_conflict_penalty = False

        self.agents: List[EnvAgent] = []
        self.num_resets = 0
        self.distance_map = DistanceMap(self.agents, self.height, self.width)
        
        self.conflict_penalty = conflict_penalty   # penalty for deadlocks, if a certain number of deadlock is present the penalty should increase
        self.num_of_conflict = 0                   # total number of conflict from the start of the training

        self.action_space = [8]
        
        self.reverse_once = True
        
        self.interruption = False
        
        self.maximum_train_velocities = [0] * number_of_agents
        
        self.maximum_distance_from_target = [1] * number_of_agents
        
        self.previous_station = [[(-1,0)]] * number_of_agents
        
        self.dones_for_position = [False] * number_of_agents
        
        # Flag to check if an action is done or not
        self.elaborate_action = False
        
        self.dense_score = 0
        self.sparse_score = 0
        
        # List with all the station positions
        self.station_positions = []
        
        for i_agent in range(len(timetable_example)):
            for i_station in range(len(timetable_example[i_agent][0])):
                if not timetable_example[i_agent][0][i_station].position in self.station_positions:
                    self.station_positions.append(timetable_example[i_agent][0][i_station].position)
        # Array with time that the agents have to wait at stations 
        # TODO multiple runs
        self.stop_station_time = [2]*number_of_agents
        
        if self.get_num_agents() == 1:
            self.next_station_to_reach = [i[0] for i in timetable_example]
            self.next_station_to_reach = self.next_station_to_reach[0]
        else:
            self.next_station_to_reach = [i[0] for i in timetable_example]

        # If there is only one train run an alternative position of (-1,0) is chosen
        self.position_ending_run = [(-1,0)] * number_of_agents
        for i_agent in range(len(timetable_example)): 
            for i_station in range(1, len(timetable_example[i_agent][0])):
                if timetable_example[i_agent][0][i_station] == timetable_example[i_agent][0][i_station - 1]:
                    self.position_ending_run[i_agent] = timetable_example[i_agent][0][i_station].position
        
        self._seed()
        if random_seed:
            self._seed(seed=random_seed)

        self.agent_positions = None

        self.run_once = [0]*(self.number_of_agents)   # Flag to check when a train has started   

        # save episode timesteps ie agent positions, orientations.  (not yet actions / observations)
        self.record_steps = record_steps  # whether to save timesteps
        # save timesteps in here: [[[row, col, dir, malfunction],...nAgents], ...nSteps]
        self.cur_episode = []
        self.list_actions = []  # save actions in here

        self.motionCheck = ac.MotionCheck()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        self.random_seed = seed

        # Keep track of all the seeds in order
        if not hasattr(self, 'seed_history'):
            self.seed_history = [seed]
        if self.seed_history[-1] != seed:
            self.seed_history.append(seed)

        return [seed]

    def find_indices(self, array, index_to_find):
        indeces = []
        for i in range(len(array)):
            if array[i] == index_to_find:
                indeces.append(i)
        return (indeces)

    # no more agent_handles
    def get_agent_handles(self):
        return range(self.get_num_agents())
    
    def get_num_agents(self) -> int:
        return len(self.agents)

    def add_agent(self, agent):
        """ Add static info for a single agent.
            Returns the index of the new agent.
        """
        self.agents.append(agent)
        return len(self.agents) - 1

    def reset_agents(self):
        """ Reset the agents to their starting positions
        """
        for agent in self.agents:
            agent.reset()
        self.active_agents = [i for i in range(len(self.agents))]

    def action_required(self, agent):
        """
        Check if an agent needs to provide an action

        Parameters
        ----------
        agent: RailEnvAgent
        Agent we want to check

        Returns
        -------
        True: Agent needs to provide an action
        False: Agent cannot provide an action
        """
        return agent.state == TrainState.READY_TO_DEPART or \
               ( agent.state.is_on_map_state() and agent.speed_counter.is_cell_entry )

    def reset(self, regenerate_rail: bool = True, regenerate_schedule: bool = True, *,
              random_seed: int = None) -> Tuple[Dict, Dict]:
        """
        reset(regenerate_rail, regenerate_schedule, activate_agents, random_seed)

        The method resets the rail environment

        Parameters
        ----------
        regenerate_rail : bool, optional
            regenerate the 
        regenerate_schedule : bool, optional
            regenerate the schedule and the static agents
        random_seed : int, optional
            random seed for environment

        Returns
        -------
        observation_dict: Dict
            Dictionary with an observation for each agent
        info_dict: Dict with agent specific information

        """

        if random_seed:
            self._seed(random_seed)

        optionals = {}
        if regenerate_rail or self.rail is None:

            if "__call__" in dir(self.rail_generator):
                rail, optionals = self.rail_generator(
                    self.width, self.height, self.number_of_agents, self.num_resets, self.np_random)
            elif "generate" in dir(self.rail_generator):
                rail, optionals = self.rail_generator.generate(
                    self.width, self.height, self.number_of_agents, self.num_resets, self.np_random)
            else:
                raise ValueError("Could not invoke __call__ or generate on rail_generator")

            self.rail = rail
            self.height, self.width = self.rail.grid.shape

            # Do a new set_env call on the obs_builder to ensure
            # that obs_builder specific instantiations are made according to the
            # specifications of the current environment : like width, height, etc
            self.obs_builder.set_env(self)

        if optionals and 'distance_map' in optionals:
            self.distance_map.set(optionals['distance_map'])

        if regenerate_schedule or regenerate_rail or self.get_num_agents() == 0:
            agents_hints = None
            if optionals and 'agents_hints' in optionals:
                agents_hints = optionals['agents_hints']

            line = self.line_generator(self.rail, self.number_of_agents, agents_hints, 
                                               self.num_resets, self.np_random)
            self.agents = EnvAgent.from_line(line)

            # Reset distance map - basically initializing
            self.distance_map.reset(self.agents, self.rail)

            # NEW : Time Schedule Generation
            timetable = timetable_generator(timetable_example, self.agents, self.distance_map, 
                                               agents_hints, self.np_random)

            #self._max_episode_steps = 250

            for agent_i, agent in enumerate(self.agents):
                agent.earliest_departure = timetable.earliest_departures[agent_i]         
                agent.latest_arrival = timetable.latest_arrivals[agent_i]
        else:
            self.distance_map.reset(self.agents, self.rail)
        
        # Reset agents to initial states
        self.reset_agents()

        self.run_once = [0]*(self.number_of_agents)   # Flag to check when a train has started

        self.num_resets += 1
        self._elapsed_steps = 0
        
        self.maximum_distance_from_target = [1] * self.number_of_agents
        
        self.previous_station = [[(-1,0)]] * self.number_of_agents
        
        self.dones_for_position = [False] * self.number_of_agents
        
        self.interruption = False
        
        self.dense_score = 0
        self.sparse_score = 0
        
        # Flag to check if an action is done or not
        self.elaborate_action = False
        
        # List with all the station positions
        self.station_positions = []
        
        self.cell_interrupted = []
        
        for i_agent in range(len(timetable_example)):
            for i_station in range(len(timetable_example[i_agent][0])):
                if not timetable_example[i_agent][0][i_station].position in self.station_positions:
                    self.station_positions.append(timetable_example[i_agent][0][i_station].position)
        # Array with time that the agents have to wait at stations 
        # TODO multiple runs
        self.stop_station_time = [2]*self.number_of_agents
        
        # array that conteins the next stations to be reach
        if self.get_num_agents() == 1:
            self.next_station_to_reach = [i[0] for i in timetable_example]
            self.next_station_to_reach = self.next_station_to_reach[0]
        else:
            self.next_station_to_reach = [i[0] for i in timetable_example]  

        # Agent positions map
        self.agent_positions = np.zeros((self.height, self.width), dtype=int) - 1
        self._update_agent_positions_map(ignore_old_positions=False)

        self.dones = dict.fromkeys(list(range(self.get_num_agents())) + ["__all__"], False)

        # Reset the state of the observation builder with the new environment
        self.obs_builder.reset()

        # Empty the episode store of agent positions
        self.cur_episode = []

        info_dict = self.get_info_dict()
        # Return the new observation vectors for each agent
        observation_dict: Dict = self._get_observations()
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer = None
        return observation_dict, info_dict
    
    def check_station_from_rails(self, timetable, rail):
        for i in range(len(timetable)):
            for j in range(len(timetable[i][0])):
                if type(timetable[i][0][j].rails) == tuple:  
                    if rail == timetable[i][0][j].rails:
                        station = timetable[i][0][j]
                        return station
                else:
                    if rail in timetable[i][0][j].rails:
                        station = timetable[i][0][j]
                        return station
    
    
      
    def station_stop(self, timetable, handle):
        agent = self.agents[handle]
        agent_position = agent.position
        for i_station in range(len(timetable[handle][0])):
            if agent_position in timetable[handle][0][i_station].rails:
                agent.action_saver.saved_action = RailEnvActions.STOP_MOVING        
    
    
    def timetable_real_time(self, timetable, station_in_which_i_am, i_agent):
        current_time = self._elapsed_steps
        num_of_stations = len(timetable[i_agent][0])
        difference = 0
        temporary_stations = []
        if self.get_num_agents() != 1:
            self.next_station_to_reach[i_agent] = []
        else:
            self.next_station_to_reach = []
        min_difference = +np.inf
        for i_station in range(num_of_stations):
            if timetable[i_agent][0][i_station] == station_in_which_i_am:
                difference = abs(timetable[i_agent][1][i_station] - current_time)
                if difference < min_difference:
                    min_difference = difference
                    station_index = i_station
        # The agent is at the end of the train run
        if station_index < len(timetable[i_agent][0]) - 1 and reinforcement_learning:
            # Case One
            if timetable[i_agent][0][station_index] == timetable[i_agent][0][station_index + 1] and \
                    station_index != 0:
                if timetable[i_agent][1][station_index + 1] - self._elapsed_steps < 0:
                    self.stop_station_time[i_agent] = 2
                else:
                    self.stop_station_time[i_agent] = timetable[i_agent][1][station_index + 1] - self._elapsed_steps
                if self.reverse_once: 
                    self.agents[i_agent].direction = (self.agents[i_agent].direction + 2) % 4
                    self.reverse_once = False
            # Case two
            if timetable[i_agent][0][station_index] == timetable[i_agent][0][station_index - 1] and \
                    station_index != 0:
                if timetable[i_agent][1][station_index - 1] - self._elapsed_steps < 0:
                    self.stop_station_time[i_agent] = 2
                else:
                    self.stop_station_time[i_agent] = timetable[i_agent][1][station_index - 1] - self._elapsed_steps
                if self.reverse_once: 
                    self.agents[i_agent].direction = (self.agents[i_agent].direction + 2) % 4
                    self.reverse_once = False
                    
        for next_stations in range(1,num_of_stations - station_index):
            temporary_stations.append(timetable[i_agent][0][next_stations + station_index])
            
        if self.get_num_agents() > 1:
            self.next_station_to_reach[i_agent] = temporary_stations
        else:
            self.next_station_to_reach = temporary_stations
    
    
    # Check the maximum possible delay...180 not good for now
    def calculate_metric(self, timetable):
        positions = self.cur_episode
        prev_station = 0
        delta = 1200
        metric_result = []
        for i_agent in range(len(timetable)):
            station_vector = [delta] * len(timetable[i_agent][0])
            for i_station in range(len(timetable[i_agent][0])):
                for step in range(len(positions)):
                    if positions[step][i_agent] in timetable[i_agent][0][i_station].rails and positions[step][i_agent] != prev_station:
                        prev_station = positions[step][i_agent]
                        distance_delay = ((step - timetable[i_agent][1][i_station])**2)**(1/2)
                        station_vector[i_station] = distance_delay
            metric_result.append(station_vector)
            prev_station = 0
        metric_sum = sum(sum(x) for x in metric_result)
        dimension = 0
        for i in range(len(metric_result)):
            for j in range(len(metric_result[i])):
                dimension += 1
        metric_normalized = 1 - (metric_sum / (delta*dimension))
        
        return metric_normalized
    
             
    def calculate_metric_single_agent(self, timetable, i_agent):
        positions = self.cur_episode
        delta = 300
        passed = False
        metric_result = []
        difference = [delta]
        num_of_stations = len(timetable[i_agent][0])
        array_of_passed_stations = [False]*num_of_stations     
        # Elements for multiple runs
        station_ending_train_run = 0
        time_ending_train_run = np.inf
        for i_station in range(1, len(timetable[i_agent][0])): 
            if timetable[i_agent][0][i_station] == timetable[i_agent][0][i_station - 1]:
                station_ending_train_run = i_station
        if not self.agents[i_agent].state == TrainState.MALFUNCTION:
            station_vector = [delta] * len(timetable[i_agent][0])
            for i_station in range(num_of_stations):
                station_importance = timetable[i_agent][0][i_station].importance
                station_vector[i_station] = delta * station_importance
                for step in range(len(positions)):
                    # two different cases
                    # multiple rails or single rail in station
                    # SINGLE RAIL
                    if type(timetable[i_agent][0][i_station].rails) == tuple:
                        if positions[step][i_agent] == timetable[i_agent][0][i_station].rails: # and positions[step][i_agent] != prev_station:
                            if i_station == station_ending_train_run:
                                time_ending_train_run = step
                            # The last station of the run is more important
                            if i_station == (len(timetable[i_agent][0]) - 1):
                                if not array_of_passed_stations[i_station]:
                                    if i_station >= station_ending_train_run:
                                        if step >= time_ending_train_run:
                                            difference.append(((step - timetable[i_agent][1][i_station])**2)**(1/2) * station_importance)
                                    else:
                                        if step < time_ending_train_run:
                                            difference.append(((step - timetable[i_agent][1][i_station])**2)**(1/2) * station_importance)
                                    passed = True
                            else:
                                if not array_of_passed_stations[i_station]:
                                    if i_station >= station_ending_train_run:
                                        if step >= time_ending_train_run:
                                            difference.append(((step - timetable[i_agent][1][i_station])**2)**(1/2) * station_importance)
                                    else:
                                        if step < time_ending_train_run:
                                            difference.append(((step - timetable[i_agent][1][i_station])**2)**(1/2) * station_importance)
                                    passed = True
                    # MULTIPLE RAILS
                    else:
                        if positions[step][i_agent] in timetable[i_agent][0][i_station].rails: # and positions[step][i_agent] != prev_station:
                            if i_station == station_ending_train_run:
                                time_ending_train_run = step
                            # The last station of the run is more important
                            if i_station == (len(timetable[i_agent][0]) - 1):
                                if not array_of_passed_stations[i_station]:
                                    difference.append(((step - timetable[i_agent][1][i_station])**2)**(1/2) * station_importance)
                                    passed = True
                            else:
                                if not array_of_passed_stations[i_station]:
                                    if i_station >= station_ending_train_run:
                                        if step >= time_ending_train_run:
                                            difference.append(((step - timetable[i_agent][1][i_station])**2)**(1/2) * station_importance)
                                    else:
                                        if step < time_ending_train_run:
                                            difference.append(((step - timetable[i_agent][1][i_station])**2)**(1/2) * station_importance)
                                    passed = True
                                    
                # If the agent has passed from a station choose the minimum time passage and then reset all the things
                if passed:
                    station_vector[i_station] = min(difference)
                    array_of_passed_stations[i_station] = True
                    difference = [delta]
                    passed = False
                    
            metric_result.append(station_vector)
        metric_sum = sum(sum(x) for x in metric_result)
        
        # Maximum value for the metric
        maximum_metric_value = 0
        
        # Remember to give more importance at the end stations
        for i_station in range(len(timetable[i_agent][0])):
            station_importance = timetable[i_agent][0][i_station].importance
            maximum_metric_value += delta * station_importance
                
        metric_normalized = 1 - (metric_sum / maximum_metric_value)
        
        return metric_normalized
    
    def calculate_sparse_reward(self, metric, threshold, maximum_reward):
        
        
        #     /              maximum value
        #    /
        #   /
        #--/-------------   threshold value
        # /
        #/
        reward = - maximum_reward * (metric - 1)/(threshold-1) + maximum_reward
        
        return reward

    def _update_agent_positions_map(self, ignore_old_positions=True):
        """ Update the agent_positions array for agents that changed positions """
        for agent in self.agents:
            if not ignore_old_positions or agent.old_position != agent.position:
                if agent.position is not None:
                    self.agent_positions[agent.position] = agent.handle
                if agent.old_position is not None:
                    self.agent_positions[agent.old_position] = -1
                 
                    
    def check_station_in_timetable(self, timetable, target_position, i_agent):
        time_difference = np.inf
        station_in_which_i_am = 0
        for i in range(len(timetable[i_agent][0])): # for all the stations
            for j in range(timetable[i_agent][0][i].capacity):   # for all the rails
                if target_position == tuple(timetable[i_agent][0][i].rails[j]):  
                    # we have multiple train run in the timetable, so is important to calculate the real station 
                    station_in_which_i_can_be = i  
                    if self._elapsed_steps - timetable[i_agent][1][station_in_which_i_can_be] < time_difference:
                        time_difference = self._elapsed_steps - timetable[i_agent][1][station_in_which_i_can_be]
                        station_in_which_i_am = i   # index of the station in which the agent is
        return station_in_which_i_am
                    
                
    
    def generate_state_transition_signals(self, timetable, agent, preprocessed_action, movement_allowed):
        """ Generate State Transitions Signals used in the state machine """        
        st_signals = StateTransitionSignals()
        
        # Malfunction starts when in_malfunction is set to true
        st_signals.in_malfunction = agent.malfunction_handler.in_malfunction

        # Malfunction counter complete - Malfunction ends next timestep
        st_signals.malfunction_counter_complete = agent.malfunction_handler.malfunction_counter_complete

        # Earliest departure reached - Train is allowed to move now
        st_signals.earliest_departure_reached = self._elapsed_steps >= agent.earliest_departure

        # Stop Action Given
        st_signals.stop_action_given = (preprocessed_action == RailEnvActions.STOP_MOVING)

        # Valid Movement action Given
        st_signals.valid_movement_action_given = preprocessed_action.is_moving_action() and movement_allowed
      
        station_target = self.check_station_from_rails(timetable, agent.target)
        
        agent_past_positions = [row[agent.handle] for row in self.cur_episode]
        
        if type(station_target.rails) == tuple:
            if agent.position == station_target.rails and self._elapsed_steps >= agent.latest_arrival/2 \
                and self.position_ending_run[agent.handle] in agent_past_positions:
                st_signals.target_reached = True
        else:
            if agent.position in station_target.rails and self._elapsed_steps >= agent.latest_arrival/2\
                and self.position_ending_run[agent.handle] in agent_past_positions:
                st_signals.target_reached = True

        # Movement conflict - Multiple trains trying to move into same cell
        # If speed counter is not in cell exit, the train can enter the cell
        st_signals.movement_conflict = (not movement_allowed) and agent.speed_counter.is_cell_exit
        
        """if st_signals.movement_conflict:
            if self.increase_conflict_penalty:
                self.conflict_penalty += -1
                self.rewards_dict[agent.handle] += self.conflict_penalty 
                self.increase_conflict_penalty = False
            else:
                self.rewards_dict[agent.handle] += self.conflict_penalty """
        
        return st_signals

    def _handle_end_reward(self, agent: EnvAgent, timetable) -> int:
        '''
        Handles end-of-episode reward for a particular agent.

        Parameters
        ----------
        agent : EnvAgent
        '''
        i_agent = agent.handle

        if training == 'training0' and i_agent != 0:
            reward = 0
            return reward
        if (training == 'training1' or training == 'training1.1') and i_agent > 1:
            reward = 0
            return reward

        reward = 0

        # Reached intermediated stations?
        #reward = self.intermediate_station_reward(i_agent, timetable)

        # agent done? (arrival_time is not None)
        
        if agent.state == TrainState.DONE:
            self.dones[i_agent] = True
        
        return reward
        
        """if agent.state == TrainState.DONE:
            # if agent arrived earlier or on time = 0
            # if agent arrived later = -ve reward based on how late
            reward += target_reward
            self.dones[i_agent] = True
            # DELAY (scheduled time - real arrival time)
            delay = min(agent.latest_arrival - agent.arrival_time, 0)  
            
            delay_penalty = delay * 0.7 * 0.7/3 # FORMULA DATA DAL PROF DA AGGIUSTARE
            reward += delay_penalty
            #reward = min(agent.latest_arrival - agent.arrival_time, 0)

        # Agents not done (arrival_time is None)
        else:
            # CANCELLED check (never departed)
            if (agent.state.is_off_map_state()):
                reward += -1 * self.cancellation_factor * \
                    (agent.get_travel_time_on_shortest_path(self.distance_map) + self.cancellation_time_buffer)

            # Departed but never reached
            if (agent.state.is_on_map_state()) and not agent.state == TrainState.MALFUNCTION:
                reward += target_not_reached_penalty
                
                pasted_agent_positions = self.cur_episode
                if pasted_agent_positions == []:
                    reward += default_skip_penalty
                    return reward
                
                stations_to_pass = timetable[i_agent][0]
                
                for positions in range (len(pasted_agent_positions)):
                    # If the agent is passed in at least one station (different from the starting one) no problems
                    if pasted_agent_positions[positions][i_agent] in stations_to_pass[1:]:
                        return reward
                    
                # else, I have to give a skip penalty for all the skipped stations, for now simplified
                # I give an high penalty
                reward += - default_skip_penalty
                return reward
        
        return reward"""

    def preprocess_action(self, action, agent):
        """
        Preprocess the provided action
            * Change to DO_NOTHING if illegal action
            * Block all actions when in waiting state
            * Check MOVE_LEFT/MOVE_RIGHT actions on current position else try MOVE_FORWARD
        """
        action = action_preprocessing.preprocess_raw_action(action, agent.state, agent.action_saver.saved_action)
        action = action_preprocessing.preprocess_action_when_waiting(action, agent.state)

        # Try moving actions on current position
        current_position, current_direction = agent.position, agent.direction
        if current_position is None: # Agent not added on map yet
            current_position, current_direction = agent.initial_position, agent.initial_direction
        
        action = action_preprocessing.preprocess_moving_action(action, self.rail, current_position, current_direction)
        
        if action == RailEnvActions.REVERSE:
            _ , reverse_active = check_reverse_action(current_direction, self.interruption)
            if reverse_active:
                action = RailEnvActions.REVERSE
            else:
                action = RailEnvActions.DO_NOTHING

        # Check transitions, bounts for executing the action in the given position and directon
        elif action.is_moving_action() and not check_valid_action(action, self.rail, current_position, current_direction):
            action = RailEnvActions.STOP_MOVING
            
        return action
    
    def clear_rewards_dict(self):
        """ Reset the rewards dictionary """
        self.rewards_dict = {i_agent: 0 for i_agent in range(len(self.agents))}

    def get_info_dict(self):
        """ 
        Returns dictionary of infos for all agents 
        dict_keys : action_required - 
                    malfunction - Counter value for malfunction > 0 means train is in malfunction
                    speed - Speed of the train
                    state - State from the trains's state machine
        """
        info_dict = {
            'action_required': {i: self.action_required(agent) for i, agent in enumerate(self.agents)},
            'malfunction': {
                i: agent.malfunction_handler.malfunction_down_counter for i, agent in enumerate(self.agents)
            },
            'speed': {i: agent.speed_counter.speed for i, agent in enumerate(self.agents)},
            'state': {i: agent.state for i, agent in enumerate(self.agents)}
        }
        return info_dict
    
    def calculate_step_reward(self, i_agent):
        
        reward = 0
        stations_distance = 0
        stations_previous_distance = 0
        
        agent = self.agents[i_agent]
        agent_speed = agent.speed_counter.speed
        
        agent_position = agent.position
        agent_prev_position = self.cur_episode[self._elapsed_steps - int(agent_speed**(-1)) - 1][i_agent] # Checking the previous position based on the speed of the agent
                
        if self.get_num_agents() != 1:
            stations_to_reach = self.next_station_to_reach[i_agent]
        else:
            stations_to_reach = self.next_station_to_reach
        
        for i_station in range(len(stations_to_reach)):
            stations_distance += len(a_star(self.rail, agent_position, \
                stations_to_reach[i_station].position, respect_rail_directions = False))
            stations_previous_distance += len(a_star(self.rail, agent_prev_position, \
                stations_to_reach[i_station].position, respect_rail_directions = False))
        
        stations_previous_distance_normalized = stations_previous_distance/len(stations_to_reach)
        
        decrement = stations_previous_distance - stations_distance
        
        if decrement  > 0.05 * stations_previous_distance_normalized:  # 5% of the previous distance
            reward += minimum_step_penalty 
        else:
            reward += (decrement + minimum_step_penalty - stations_previous_distance_normalized*0.05)/30
        return reward
    
    def update_step_rewards(self, i_agent):
        """
        Update the rewards dict for agent id i_agent for every timestep
        """
        
        agent = self.agents[i_agent]
        action = agent.action_saver.saved_action
        
        if agent.state == TrainState.DONE:
            return

        reward = 0

        if action == RailEnvActions.REVERSE:
            reward += reverse_penality
        
        if agent.position != None:
            reward += self.calculate_step_reward(i_agent)
        
        # If an agent decided not to spawn after 10 minutes with respect to the scheduled starting time we punish it
        if self._elapsed_steps > agent.earliest_departure + 2:
            if agent.position == None:
                reward += not_spawning_penalty

        self.dense_score += reward
        
        self.rewards_dict[i_agent] += reward
        
    def calculate_train_run(self, timetable, i_agent, specific_station_index):
        """[Function to calculate a specific train run for a specific agent]

        Args:
            timetable ([list]): [Timetable]
            i_agent ([int]): [i_agent]
            specific_station_index ([int]): [Index of the station I want to calculate the train run in which is]
        """
        all_train_run = timetable[i_agent][0]
        all_times = timetable[i_agent][1]

        previous_station = all_train_run[specific_station_index]
        
        specific_train_run_stations = []
        specific_train_run_times = []
        specific_train_run = []
        train_run_initial_index = 0
        
        # Starting finding the first station
        if specific_station_index != 0:
            for station_reversed in range(specific_station_index + 1):                    
                if all_train_run[specific_station_index - station_reversed - 1] == previous_station:
                    initial_train_run_station = previous_station
                    if station_reversed == 0:
                        train_run_initial_index = specific_station_index
                        initial_time = all_times[train_run_initial_index]
                    else:
                        train_run_initial_index = specific_station_index - station_reversed
                        initial_time = all_times[train_run_initial_index]
                    break
                elif station_reversed == specific_station_index:
                    initial_train_run_station = previous_station
                    train_run_initial_index = specific_station_index - station_reversed
                    initial_time = all_times[train_run_initial_index]
                    break
                else:
                    previous_station = all_train_run[specific_station_index - station_reversed - 1] 
        # If I'm the first station...
        else:
            initial_train_run_station = previous_station
            initial_time = all_times[0]
        
        previous_station = initial_train_run_station
        previous_time = initial_time   
        
        for stations in range(1 , len(all_train_run)):
            if previous_station == all_train_run[train_run_initial_index + stations]:
                ending_train_run_station = previous_station
                specific_train_run_stations.append(ending_train_run_station)
                specific_train_run_times.append(all_times[train_run_initial_index + stations])
                break
            elif train_run_initial_index + stations == len(all_train_run):
                ending_train_run_station = previous_station
                specific_train_run_stations.append(ending_train_run_station)
                specific_train_run_times.append(all_times[train_run_initial_index + stations])
                break
            else:
                specific_train_run_stations.append(previous_station)
                specific_train_run_times.append(previous_time)
                previous_station = all_train_run[train_run_initial_index + stations]
                previous_time = all_times[train_run_initial_index + stations]
                if train_run_initial_index + stations == len(all_train_run) - 1:
                    specific_train_run_stations.append(previous_station)
                    specific_train_run_times.append(all_times[train_run_initial_index + stations])
                    break
        specific_train_run.append(specific_train_run_stations)
        specific_train_run.append(specific_train_run_times)
                
        return specific_train_run
    
    def calculate_skip_penalty(self, timetable, index_of_station_skipped, station_skipped, time_scheduled,
                               i_agent, station, train_type, specific_train_run):
        
        train_importance = train_type
        station_importance = 1
        #station_importance = station.importance   # TODO modificare !!!!
        number_of_station_to_pass = len(specific_train_run)
        
        # array that have to contein all the possible passages from the skipped station for other agents (convoys)
        possible_train_passage = []
        
        for i in range(len(timetable)):
            if i != i_agent:
                num_of_stations = len(timetable[i_agent][0])
                for stations in range(num_of_stations - 1):
                    if timetable[i][0][stations] == station_skipped:        # same station
                        if timetable[i][1][stations] > time_scheduled:      # greater time, so the successive agent
                            if timetable[i][0][stations + 1] == timetable[i_agent][0][index_of_station_skipped + 1]:   # same direction
                                possible_train_passage.append(timetable[i][1][stations] - time_scheduled)
        
        if possible_train_passage == []:
            penalty = - default_skip_penalty
            return penalty
            
        else:
            for i in range(len(possible_train_passage)):
                delay = min(possible_train_passage)
                
            
        penalty = - (delay*train_importance*station_importance)/number_of_station_to_pass
        
        return penalty
    
    def calculate_delay_penalty(self, delay, train_run, station, train_type):
        """[Calculate the penalty of the agent based on the delay, and weighted on the type of train, on the importance of the station and 
        on the number of stations reached by the train]

        Args:
            delay ([int]): [delay of the train in a specific station]
            train_run ([list]): [specific train run i'm doing with my train]
            station ([Station]): [station reached]
            train_type ([convoy.type]): [type of the convoy (regional, intercity, high velocity)]
        """
        number_of_station_to_pass = len(train_run) + 1     
        
        train_importance = train_type
        #station_importance = station.importance
        station_importance = station.importance / 10
        
        penalty = - (delay*train_importance*station_importance)/number_of_station_to_pass
        
        if delay < 0:
            penalty = station_passage_reward
        
        return penalty

    def end_of_episode_update(self, have_all_agents_ended, timetable):
        """ 
        Updates made when episode ends
        Parameters: have_all_agents_ended - Indicates if all agents have reached done state
        """
        # When there is a conflict the episode is concluded [failed :'-(]
        agent_conflict = False
        
        # HERE ACTIVATE THE END OF EPISODE IN CASE OF CONFLICT !!!!!!!!!
        # THIS IS IMPORTANT IF WE WANT TO "ACTIVATE" SINGLE RAIL SECTIONS
        """for i in range(self.number_of_agents):
            agent = self.agents[i]
            if agent.state_machine.st_signals.movement_conflict:
                agent_conflict = True
                self.num_of_conflict += 1
                break"""
        
        if have_all_agents_ended or \
           ( (self._max_episode_steps is not None) and (self._elapsed_steps >= self._max_episode_steps)) or agent_conflict:

            """for i_agent, agent in enumerate(self.agents):
                
                reward = self._handle_end_reward(agent, timetable)
                self.rewards_dict[i_agent] += reward
                
                #self.dones[i_agent] = True"""
                
            reward = 0
            
            for i_agent in self.get_agent_handles():
                
                if self.agents[i_agent].state != TrainState.MALFUNCTION:
                    metric = self.calculate_metric_single_agent(timetable, i_agent)
                    
                    reward = self.calculate_sparse_reward(metric, metric_threshold, maximum_reward)
                    
                    self.rewards_dict[i_agent] += reward
                    
                    self.sparse_score += reward  #RICORDA DI NORMALIZZARE (?)

            self.dones["__all__"] = True

    def handle_done_state(self, agent):
        """ Any updates to agent to be made in Done state """
        if agent.state == TrainState.DONE and agent.arrival_time is None and self._elapsed_steps >= agent.latest_arrival/2:
            agent.arrival_time = self._elapsed_steps
            if self.remove_agents_at_target:
                agent.position = None    

    def check_intermediate_station_passage(self, step, i_agent, timetable):
            from operator import itemgetter
            positions = self.cur_episode
            if positions == []:
                return
            reward = 0
                        
            rails_to_pass = []
            for i in range(len(timetable[i_agent][0])):
                for j in range(len(timetable[i_agent][0][i].rails)):
                    rails_to_pass.append(tuple(timetable[i_agent][0][i].rails[j]))

            if self.agents[i_agent].position in rails_to_pass and self.agents[i_agent].position not in self.previous_station[i_agent]:

                if self.agents[i_agent].state == TrainState.DONE:
                    for j in range(len(timetable[i_agent][0][-1].rails)):
                        self.previous_station[i_agent] = (tuple(timetable[i_agent][0][-1].rails[j]))  # Final position reached by the agent
                else:
                    station = self.check_station_from_rails(timetable, positions[step - 1][i_agent])
                    for j in range(len(timetable[i_agent][0][-1].rails)):
                        self.previous_station[i_agent] = station.rails
                
                reward += station_passage_reward

                station_in_which_i_am = self.check_station_from_rails(timetable, self.agents[i_agent].position)

                index = self.find_indices(timetable[i_agent][0], station_in_which_i_am)
                difference = []
                difference_abs = []
                for num_station in range(len(index)):
                    difference.append((step - timetable[i_agent][1][index[num_station]])) 
                    difference_abs.append(abs(step - timetable[i_agent][1][index[num_station]]))
                # Use the absolute values to calculate the index of the min 
                index_of_min, value_of_min = min(enumerate(difference_abs), key=itemgetter(1))
                # Use the real difference (with informations about dealy or advance) to calculate the delay or advance
                value_of_min = difference[index_of_min]
                index_of_my_station = index[index_of_min]
                
                station = timetable[i_agent][0][index_of_my_station]
                train_type = timetable[i_agent][2].maximum_velocity
                
                specific_train_run = self.calculate_train_run(timetable, i_agent, index_of_my_station)
                
                index_of_my_station_for_my_train_run = specific_train_run[0].index(station)
                
                reward += self.calculate_delay_penalty(value_of_min, specific_train_run, station, train_type)
                # Check if I have skipped a previous station
                if index_of_my_station > 1:
                    previous_stations = specific_train_run[0][0:index_of_my_station_for_my_train_run]
                    flag_station_passed = [False] * (len(previous_stations) - 1)
                    for past_positions in range(len(positions)):
                        for i_previous_station in range(1, index_of_my_station):
                            if positions[past_positions][i_agent] == timetable[i_agent][0][i_previous_station].rails:
                                self.rewards_dict[i_agent] += 0 
                                flag_station_passed[i_previous_station - 1] = True
                    
                    for i_previous_station in range(len(flag_station_passed)):
                        if not flag_station_passed[i_previous_station]: 
                            station_skipped = previous_stations[i_previous_station]
                            time_scheduled_for_station = specific_train_run[1][i_previous_station] 
                            reward += self.calculate_skip_penalty(timetable, index_of_my_station, station_skipped, 
                                                             time_scheduled_for_station, i_agent, station, train_type, specific_train_run)
                    self.rewards_dict[i_agent] += reward
                    return
                
                self.rewards_dict[i_agent] += reward


    def step(self, action_dict_: Dict[int, RailEnvActions]):
        """
        Updates rewards for the agents at a step.
        """
        self._elapsed_steps += 1

        # Not allowed to step further once done
        if self.dones["__all__"]:
            raise Exception("Episode is done, cannot call step()")

        self.clear_rewards_dict()

        have_all_agents_ended = True # Boolean flag to check if all agents are done

        self.motionCheck = ac.MotionCheck()  # reset the motion check

        temp_transition_data = {}
        
        for agent in self.agents:
            i_agent = agent.handle

            # Build info dict
            rail, optionals = self.rail_generator(
                self.width, self.height, self.number_of_agents, self.num_resets, self.np_random)
            
            timetable = optionals['agents_hints']['timetable']

            # Starting time of the agent
            starting_time = timetable[i_agent][1][0]
            ending_time = timetable[i_agent][1][-1]
            
            # Calculate velocities that the agents have to mantein
            if self._elapsed_steps == starting_time:
                agent.speed_counter.speed = timetable[i_agent][2].maximum_velocity
                self.maximum_train_velocities[i_agent] = timetable[i_agent][2].maximum_velocity
                
            agent.earliest_departure = starting_time
            agent.latest_arrival = ending_time
            
            agent_speed = agent.speed_counter.speed
            
            # Control if the agent has to make an action or not...In this case ok, if not I jump at the end of this cycle
            """if not agent.speed_counter.is_cell_exit and not agent.speed_counter.is_cell_entry and \
                agent.action_saver.is_action_saved and agent.state.is_off_map_state() and agent.state == TrainState.READY_TO_DEPART:
                new_position, new_direction = agent.initial_position, agent.initial_position
                continue
            
            elif not agent.speed_counter.is_cell_exit and not agent.speed_counter.is_cell_entry and \
                agent.action_saver.is_action_saved and agent.state.is_on_map_state():
                new_position, new_direction = agent.position, agent.position
                self.elaborate_action = True
                continue"""
            
            self.elaborate_action = False
            agent.old_position = agent.position
            agent.old_direction = agent.direction
            # Generate malfunction
            agent.malfunction_handler.generate_malfunction(self.malfunction_generator, self.np_random)

            # Get action for the agent
            action = action_dict_.get(i_agent, RailEnvActions.DO_NOTHING)

            preprocessed_action = self.preprocess_action(action, agent)

            # Save moving actions in not already saved
            agent.action_saver.save_action_if_allowed(preprocessed_action, agent.state)

            # Train's next position can change if current stopped in a fractional speed or train is at cell's exit

            position_update_allowed = agent.speed_counter.is_cell_exit and \
                        not agent.malfunction_handler.malfunction_down_counter > 0 and \
                        not preprocessed_action == RailEnvActions.STOP_MOVING                     

            #position_update_allowed = (agent.speed_counter.is_cell_exit or agent.state == TrainState.STOPPED)

            # Calculate new position
            # Keep agent in same place if already done
            if agent.state == TrainState.DONE:
                new_position, new_direction = agent.position, agent.direction
            elif agent.state == TrainState.MALFUNCTION:
                new_position, new_direction = agent.position, agent.direction
            # Add agent to the map if not on it yet
            elif agent.position is None and agent.action_saver.is_action_saved:
                new_position = agent.initial_position
                new_direction = agent.initial_direction       
            elif agent.action_saver.saved_action == RailEnvActions.REVERSE and position_update_allowed:
                new_direction, _ = check_reverse_action(agent.direction, self.interruption)
                new_position = get_new_position(agent.position, new_direction)
            # If the agent has to accelerate or decelerate
            elif agent.action_saver.is_action_saved and agent.action_saver.saved_action.is_action_speed():
                agent_velocity = agent.speed_counter.speed
                if agent.action_saver.saved_action == RailEnvActions.ACCELERATE:
                    if agent_velocity < self.maximum_train_velocities[i_agent]:
                        new_speed = (int(1 / agent_velocity)) - 1
                        agent.speed_counter.speed = 1 / new_speed
                    new_position = agent.position
                    new_direction = agent.direction
                elif agent.action_saver.saved_action == RailEnvActions.DECELERATE:
                    new_speed = (int(1 / agent_velocity)) + 1
                    agent.speed_counter.speed = 1 / new_speed
                    new_position = agent.position
                    new_direction = agent.direction
            # If movement is allowed apply saved action independent of other agents
            elif agent.action_saver.is_action_saved and position_update_allowed:
                saved_action = agent.action_saver.saved_action
                # Apply action independent of other agents and get temporary new position and direction
                new_position, new_direction  = env_utils.apply_action_independent(saved_action, 
                                                                             self.rail, 
                                                                             agent.position, 
                                                                             agent.direction)
                preprocessed_action = saved_action
            else:
                new_position, new_direction = agent.position, agent.direction

            temp_transition_data[i_agent] = env_utils.AgentTransitionData(position=new_position,
                                                                direction=new_direction,
                                                                preprocessed_action=preprocessed_action)
            
            # This is for storing and later checking for conflicts of agents trying to occupy same cell                                                    
            self.motionCheck.addAgent(i_agent, agent.position, new_position)

        # Find conflicts between trains trying to occupy same cell
        self.motionCheck.find_conflicts()
        
        for agent in self.agents:
            
            i_agent = agent.handle
 
            # Control if the agent has to make an action or not...In this case ok, if not I jump at the end of this cycle
            """if not agent.speed_counter.is_cell_exit and not agent.speed_counter.is_cell_entry and \
                agent.action_saver.is_action_saved and agent.state.is_off_map_state():
                new_position, new_direction = agent.initial_position, agent.initial_position
                continue
            elif not agent.speed_counter.is_cell_exit and not agent.speed_counter.is_cell_entry and \
                agent.action_saver.is_action_saved:
                new_position, new_direction = agent.position, agent.position
                continue"""

            ## Update positions
            if agent.malfunction_handler.in_malfunction:
                movement_allowed = False
            elif new_position in self.cell_interrupted:
                movement_allowed = False
            else:
                movement_allowed = self.motionCheck.check_motion(i_agent, agent.position) 

            movement_inside_cell = agent.state == TrainState.STOPPED and not agent.speed_counter.is_cell_exit
            movement_allowed = movement_allowed or movement_inside_cell

            # Fetch the saved transition data
            agent_transition_data = temp_transition_data[i_agent]
            preprocessed_action = agent_transition_data.preprocessed_action

            ## Update states
            state_transition_signals = self.generate_state_transition_signals(timetable_example, agent, preprocessed_action, movement_allowed)
            agent.state_machine.set_transition_signals(state_transition_signals)
            agent.state_machine.step()

            # Needed when not removing agents at target
            movement_allowed = movement_allowed and agent.state != TrainState.DONE

            # Agent is being added to map
            if agent.state.is_on_map_state():
                if agent.state_machine.previous_state.is_off_map_state():
                    agent.position = agent.initial_position
                    agent.direction = agent.initial_direction
            # Speed counter completes
                elif movement_allowed and (agent.speed_counter.is_cell_exit):
                    agent.position = agent_transition_data.position
                    agent.direction = agent_transition_data.direction
                    if state_transition_signals.target_reached and self._elapsed_steps >= agent.latest_arrival/2:
                        agent.state_machine.update_if_reached(agent.position, agent.target)

            # Off map or on map state and position should match
            env_utils.state_position_sync_check(agent.state, agent.position, agent.handle)
                
        self._update_agent_positions_map()
        if self.record_steps:
            self.record_timestep(action_dict_)
            
        for agent in self.agents:
            
            i_agent = agent.handle

            ## Update rewards 
            if agent.state != TrainState.MALFUNCTION and agent.state != TrainState.DONE and not agent.position in self.station_positions and \
                self._elapsed_steps % interval_to_calculate_step_reward == 0:                           
                self.update_step_rewards(i_agent)

            """ # The if condition is important to avoid multiple penalties due to malfunctions occurred in stations
            if agent.state.is_on_map_state() or agent.state == TrainState.DONE:
                if agent.state != TrainState.MALFUNCTION: 
                    self.check_intermediate_station_passage(self._elapsed_steps, i_agent, optionals['agents_hints']['timetable'])"""
                
            # Handle done state actions, optionally remove agents
            self.handle_done_state(agent)
            
            if training == 'training0':
                if i_agent == 0:
                    have_all_agents_ended &= (agent.state == TrainState.DONE)

            elif training == 'training1' or training == 'training1.1':
                if i_agent < 2:
                    have_all_agents_ended &= (agent.state == TrainState.DONE)

            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # to do Aggiusta in modo che sia più generale
            else:
                if i_agent == 0:
                    have_all_agents_ended &= (agent.state == TrainState.DONE)

            # If an action is chosen and the agent have to do (not done yet) the state is moving
            if self.elaborate_action and not agent.state == TrainState.DONE and self._elapsed_steps >= agent.earliest_departure:
                agent.state = TrainState.MOVING
            ## Update counters (malfunction and speed)            
            agent.speed_counter.update_counter(agent.state, agent.old_position)
                                            #    agent.state_machine.previous_state)
            agent.malfunction_handler.update_counter()

            # Clear old action when starting in new cell
            if agent.speed_counter.is_cell_entry and agent.position is not None:
                agent.action_saver.clear_saved_action()
               
            # Checking if the agent has reached one of the scheduled station
            array_of_positions_of_stations = []   # array with the position of the stations
            if self.get_num_agents() != 1:
                for i_station in range(len(self.next_station_to_reach[i_agent])):
                    if not self.next_station_to_reach[i_agent][i_station].rails in array_of_positions_of_stations:
                        array_of_positions_of_stations.append(self.next_station_to_reach[i_agent][i_station].rails)
            else:
                for i_station in range(len(self.next_station_to_reach)):
                    if not self.next_station_to_reach[i_station].rails in array_of_positions_of_stations:
                        array_of_positions_of_stations.append(self.next_station_to_reach[i_station].rails)
                
            if any(type(elements) == list for elements in array_of_positions_of_stations):
                array_of_positions_of_stations = [item for sublist in array_of_positions_of_stations for item in sublist]
            
            # If the agent has reached a scheduled station update the next station to be reached
            if agent.position in array_of_positions_of_stations:
                # HERE ADDING THE STOP AT STATION
                station_in_which_i_am = self.check_station_from_rails(timetable_example, agent.position)
                self.timetable_real_time(timetable_example, station_in_which_i_am, agent.handle)
        
        # Check if episode has ended and update rewards and dones
        self.end_of_episode_update(have_all_agents_ended, optionals['agents_hints']['timetable'])

        return self._get_observations(), self.rewards_dict, self.dones, self.get_info_dict() 
    


    '''def record_timestep(self, dActions):
                    """ 
                    Record the positions and orientations of all agents in memory, in the cur_episode
                    """
                    list_agents_state = []
                    for i_agent in range(self.get_num_agents()):
                        agent = self.agents[i_agent]
                        # the int cast is to avoid numpy types which may cause problems with msgpack
                        # in env v2, agents may have position None, before starting
                        if agent.position is None:
                            pos = (0, 0)
                        else:
                            pos = (int(agent.position[0]), int(agent.position[1]))
                        # print("pos:", pos, type(pos[0]))
                        list_agents_state.append([
                                *pos, int(agent.direction), 
                                agent.malfunction_handler.malfunction_down_counter,  
                                int(agent.state),
                                int(agent.position in self.motionCheck.svDeadlocked)
                                ])
            
                    self.cur_episode.append(list_agents_state)
                    self.list_actions.append(dActions)'''

    def record_timestep(self, dActions):
        """ 
        Record the positions and orientations of all agents in memory, in the cur_episode
        """
        list_agents_state = []
        for i_agent in range(self.get_num_agents()):
            agent = self.agents[i_agent]
            # the int cast is to avoid numpy types which may cause problems with msgpack
            # in env v2, agents may have position None, before starting
            if agent.state == TrainState.DONE and not self.dones_for_position[i_agent]:
                pos = agent.target
                self.dones_for_position[i_agent] = True
                list_agents_state.append(pos)
                continue
                
            elif agent.state.is_off_map_state():
                pos = (-1, 0) 
                
            elif agent.position == None: 
                pos = (-1, 0)
            else:
                pos = (int(agent.position[0]), int(agent.position[1]))
                
            # print("pos:", pos, type(pos[0]))
            list_agents_state.append(pos)

        self.cur_episode.append(list_agents_state)
        self.list_actions.append(dActions)

    def _get_observations(self):
        """
        Utility which returns the dictionary of observations for an agent with respect to environment
        """
        # print(f"_get_obs - num agents: {self.get_num_agents()} {list(range(self.get_num_agents()))}")
        self.obs_dict = self.obs_builder.get_many(list(range(self.get_num_agents())))
        return self.obs_dict

    def get_valid_directions_on_grid(self, row: int, col: int) -> List[int]:
        """
        Returns directions in which the agent can move
        """
        return Grid4Transitions.get_entry_directions(self.rail.get_full_transitions(row, col))

    def _exp_distirbution_synced(self, rate: float) -> float:
        """
        Generates sample from exponential distribution
        We need this to guarantee synchronity between different instances with same seed.
        :param rate:
        :return:
        """
        u = self.np_random.rand()
        x = - np.log(1 - u) * rate
        return x

    def _is_agent_ok(self, agent: EnvAgent) -> bool:
        """
        Check if an agent is ok, meaning it can move and is not malfuncitoinig
        Parameters
        ----------
        agent

        Returns
        -------
        True if agent is ok, False otherwise

        """
        return agent.malfunction_handler.in_malfunction
        

    def save(self, filename):
        print("DEPRECATED call to env.save() - pls call RailEnvPersister.save()")
        persistence.RailEnvPersister.save(self, filename)

    def render(self, mode="rgb_array", gl="PGL", agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
            show_debug=False, clear_debug_text=True, show=False,
            screen_height=600, screen_width=800,
            show_observations=False, show_predictions=False,
            show_rowcols=False, return_image=True):
        """
        This methods provides the option to render the
        environment's behavior as an image or to a window.
        Parameters
        ----------
        mode

        Returns
        -------
        Image if mode is rgb_array, opens a window otherwise
        """
        if not hasattr(self, "renderer") or self.renderer is None:
            self.initialize_renderer(mode=mode, gl=gl,  # gl="TKPILSVG",
                                    agent_render_variant=agent_render_variant,
                                    show_debug=show_debug,
                                    clear_debug_text=clear_debug_text,
                                    show=show,
                                    screen_height=screen_height,  # Adjust these parameters to fit your resolution
                                    screen_width=screen_width)
        return self.update_renderer(mode=mode, show=show, show_observations=show_observations,
                                    show_predictions=show_predictions,
                                    show_rowcols=show_rowcols, return_image=return_image)

    def initialize_renderer(self, mode, gl,
                agent_render_variant,
                show_debug,
                clear_debug_text,
                show,
                screen_height,
                screen_width):
        # Initiate the renderer
        self.renderer = RenderTool(self, gl=gl,  # gl="TKPILSVG",
                                agent_render_variant=agent_render_variant,
                                show_debug=show_debug,
                                clear_debug_text=clear_debug_text,
                                screen_height=screen_height,  # Adjust these parameters to fit your resolution
                                screen_width=screen_width)  # Adjust these parameters to fit your resolution
        self.renderer.show = show
        self.renderer.reset()

    def update_renderer(self, mode, show, show_observations, show_predictions,
                    show_rowcols, return_image):
        """
        This method updates the render.
        Parameters
        ----------
        mode

        Returns
        -------
        Image if mode is rgb_array, None otherwise
        """
        image = self.renderer.render_env(show=show, show_observations=show_observations,
                                show_predictions=show_predictions,
                                show_rowcols=show_rowcols, return_image=return_image)
        if mode == 'rgb_array':
            return image[:, :, :3]

    def close(self):
        """
        This methods closes any renderer window.
        """
        if hasattr(self, "renderer") and self.renderer is not None:
            try:
                if self.renderer.show:
                    self.renderer.close_window()
            except Exception as e:
                print("Could Not close window due to:",e)
            self.renderer = None


    def check_speed(self, agents_hints):

    # Velocity depending on the train type and on the line (Take the minimum between the two possible velocities)
        train_velocities = [0]*self.number_of_agents

        # Check for all the agents
        for i_agent, agent in enumerate(self.agents):

            # the i_agent
            agent = self.agents[i_agent]

            # Check if the agent is in the environment or not
            if agent.position != None:

                # If the agent is in the line i the max velocity is x

                # High velocity line case
                if (agent.position in av_line):  
                    train_velocities[i_agent] = min(1, agents_hints['timetable'][i_agent][2].maximum_velocity)                
                # Regional line case
                else:
                    train_velocities[i_agent] = min(1/2, agents_hints['timetable'][i_agent][2].maximum_velocity)

            # If agent is not in the environment deafault velocity is 1/2
            else:
                train_velocities[i_agent] = 1/2

        return train_velocities
