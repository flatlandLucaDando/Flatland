"""
Definition of the RailEnv environment.
"""
import random

from typing import List, Optional, Dict, Tuple

import numpy as np
from gym.utils import seeding

from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4 import Grid4Transitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env_action import RailEnvActions

from flatland.envs import malfunction_generators as mal_gen
from flatland.envs import rail_generators as rail_gen
from flatland.envs import line_generators as line_gen
from flatland.envs.timetable_generators import timetable_generator
from flatland.envs import persistence
from flatland.envs import agent_chains as ac

from flatland.envs.observations import GlobalObsForRailEnv

from flatland.envs.timetable_generators import timetable_generator
from flatland.envs.step_utils.states import TrainState, StateTransitionSignals
from flatland.envs.step_utils.transition_utils import check_valid_action
from flatland.envs.step_utils import action_preprocessing
from flatland.envs.step_utils import env_utils

from structures_rail import av_line

from configuration import example_training

# Penalities 
step_penality = - 0.01               # a step is time passing, so a penality for each step is needed
stop_penality = 0                # penalty for stopping a moving agent
reverse_penality = 0             # penalty for reversing the march of an agent
skip_penality = 0                   # penalty for skipping a station
target_not_reached_penalty = -1.5     # penalty for not reaching the final target (depot)
default_skip_penalty = 10000
cancellation_factor = 1
cancellation_time_buffer = 0

target_reward = 5         # reward for an agent reaching his final target
station_passage_reward = 3 # reward for an agent reaching intermediate station, the reward is wheighted with the delay of the agent

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

        self.agents: List[EnvAgent] = []
        self.num_resets = 0
        self.distance_map = DistanceMap(self.agents, self.height, self.width)

        self.action_space = [6]
        
        self.previous_station = [0] * number_of_agents
        
        self.dones_for_position = [False] * number_of_agents
        

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
            regenerate the rails
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
            timetable = timetable_generator(self.agents, self.distance_map, 
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
        
        self.previous_station = [0] * self.number_of_agents
        
        self.dones_for_position = [False] * self.number_of_agents
        

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


    def _update_agent_positions_map(self, ignore_old_positions=True):
        """ Update the agent_positions array for agents that changed positions """
        for agent in self.agents:
            if not ignore_old_positions or agent.old_position != agent.position:
                if agent.position is not None:
                    self.agent_positions[agent.position] = agent.handle
                if agent.old_position is not None:
                    self.agent_positions[agent.old_position] = -1
    
    def generate_state_transition_signals(self, agent, preprocessed_action, movement_allowed, target_time):
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

        # Target Reached
        if self._elapsed_steps >= target_time:
            # agent.target = [agent.target[0] - 1, agent.target[1]]
            st_signals.target_reached = env_utils.fast_position_equal(agent.position, agent.target)
        else:
            st_signals.target_reached = False

        # Movement conflict - Multiple trains trying to move into same cell
        # If speed counter is not in cell exit, the train can enter the cell
        st_signals.movement_conflict = (not movement_allowed) and agent.speed_counter.is_cell_exit

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
            # if agent arrived earlier or on time = 0
            # if agent arrived later = -ve reward based on how late
            reward += target_reward
            i_agent = agent.handle
            self.dones[i_agent] = True
            # DELAY
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
            if (agent.state.is_on_map_state()):
                reward += target_not_reached_penalty
                # DELAY
                #reward = agent.get_current_delay(self._elapsed_steps, self.distance_map)
        
        return reward

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

        # Check transitions, bounts for executing the action in the given position and directon
        if action.is_moving_action() and not check_valid_action(action, self.rail, current_position, current_direction):
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
    
    def update_step_rewards(self, i_agent):
        """
        Update the rewards dict for agent id i_agent for every timestep
        """

        """
        action = self.agents[i_agent].action_saver.saved_action
        moving = self.agents[i_agent].moving
        state = self.agents[i_agent].state
        """
        reward = None

        reward = step_penality
        """
        if action == RailEnvActions.REVERSE:
            reward += reverse_penality
        if not moving or state == TrainState.STOPPED:
            reward += stop_penality
        """
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
        station_importance = 0.7
        #station_importance = station.importance   # TODO modificare !!!!
        number_of_station_to_pass = len(specific_train_run)
        
        # array that have to contein all the possible passages from the skipped station for other agents (convoys)
        possible_train_passage = []
        
        for i in range(len(timetable)):
            if i != i_agent:
                num_of_stations = len(timetable[i_agent][0])
                for stations in range(num_of_stations):
                    if timetable[i][0][stations] == station_skipped:        # same station
                        if timetable[i][1][stations] > time_scheduled:      # greater time, so the successive agent
                            if timetable[i][0][stations + 1] == timetable[i_agent][0][index_of_station_skipped + 1]:   # same direction
                                possible_train_passage.append(timetable[i][1][stations] - time_scheduled)
        
        if possible_train_passage == []:
            penalty = default_skip_penalty
            
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
        number_of_station_to_pass = len(train_run)      
        
        train_importance = train_type
        #station_importance = station.importance
        station_importance = 0.7  # TODO modificalo !!!!!
        
        penalty = - (delay*train_importance*station_importance)/number_of_station_to_pass
        
        return penalty

    def end_of_episode_update(self, have_all_agents_ended, timetable):
        """ 
        Updates made when episode ends
        Parameters: have_all_agents_ended - Indicates if all agents have reached done state
        """
        if have_all_agents_ended or \
           ( (self._max_episode_steps is not None) and (self._elapsed_steps >= self._max_episode_steps)):

            for i_agent, agent in enumerate(self.agents):
                
                reward = self._handle_end_reward(agent, timetable)
                self.rewards_dict[i_agent] += reward
                
                #self.dones[i_agent] = True

            #self.dones["__all__"] = True

    def handle_done_state(self, agent):
        """ Any updates to agent to be made in Done state """
        if agent.state == TrainState.DONE and agent.arrival_time is None:
            agent.arrival_time = self._elapsed_steps
            if self.remove_agents_at_target:
                agent.position = None

    def check_intermediate_station_passage(self, step, i_agent, timetable):
            from operator import itemgetter
            positions = self.cur_episode
            if positions == []:
                return
            reward = 0
            step += - 2
            stations_to_pass = timetable[i_agent][0]

            if positions[step][i_agent] in stations_to_pass and positions[step][i_agent] != self.previous_station[i_agent]:
                
                self.previous_station[i_agent] = positions[step][i_agent]
                
                reward += station_passage_reward

                station_in_which_i_am = positions[step][i_agent]

                index = self.find_indices(timetable[i_agent][0], positions[step][i_agent])
                difference = []
                for num_station in range(len(index)):
                    difference.append(step - timetable[i_agent][1][index[num_station]])

                index_of_min, value_of_min = min(enumerate(difference), key=itemgetter(1))
                index_of_my_station = index[index_of_min]
                
                station = timetable[i_agent][0][index_of_my_station]
                train_type = timetable[i_agent][2]
                
                specific_train_run = self.calculate_train_run(timetable, i_agent, index_of_my_station)
                
                index_of_my_station_for_my_train_run = specific_train_run[0].index(station)
                
                reward += self.calculate_delay_penalty(value_of_min, specific_train_run, station, train_type)

                if index_of_my_station > 1:
                    previous_stations = specific_train_run[0][0:index_of_my_station_for_my_train_run]
                    previous_times = specific_train_run[1][0:index_of_my_station_for_my_train_run]
                    flag_station_passed = [False] * len(previous_stations)
                    for past_positions in range(len(positions)):
                        for i_previous_station in range(1, index_of_my_station):
                            if positions[past_positions][i_agent] == timetable[i_agent][0][index_of_my_station - i_previous_station]:
                                self.rewards_dict[i_agent] += 0 
                                flag_station_passed[index_of_my_station - i_previous_station] = True
                    
                    for i_previous_station in range(len(flag_station_passed)):
                        if not flag_station_passed[i_previous_station]: 
                            station_skipped = previous_stations[0][i_previous_station]
                            time_scheduled_for_station = previous_stations[1][i_previous_station] 
                            reward += self.calculate_skip_penalty(timetable, index_of_my_station, station_skipped, 
                                                             time_scheduled_for_station, i_agent, station, train_type, specific_train_run)
                    self.rewards_dict[i_agent] += reward
                    return
    """
    def intermediate_station_reward(self, convoy_i, timetable):
        reward = 0
        positions = self.cur_episode
        passed_positions_convoy_i = [row[convoy_i] for row in positions]
        i = 0
        initial_station = timetable[convoy_i][0][0]
        for station_i in timetable[convoy_i][0]:
            if station_i == initial_station:
                continue
            for positions in passed_positions_convoy_i:
                if station_i == positions:
                    index = self.find_indices(timetable[convoy_i][0], station_i)
                    for time_index in index: 
                        time_scheduled = timetable[convoy_i][1][time_index]
                        time_difference = (time_scheduled - i)**2
                        if time_difference == 0:
                            time_difference = 1
                        reward += station_passage_reward/time_difference
                i += 1
        return reward      """ 

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

            agent = self.agents[i_agent]

            # Calculate velocities that the agents have to mantein
            velocities = self.check_speed(optionals['agents_hints'])   # TODO variare velocità in base alla stazione da raggiungere     

            agent.speed_counter.speed = velocities[i_agent]

            # Starting time of the agent
            starting_time = optionals['agents_hints']['timetable'][i_agent][1][0]
            ending_time = optionals['agents_hints']['timetable'][i_agent][1][-1]

            agent.earliest_departure = starting_time
            agent.latest_arrival = ending_time

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

        # Find conflicts between trains trying to occupy same cell  TODO controlla i bug
        self.motionCheck.find_conflicts()
        
        for agent in self.agents:
            i_agent = agent.handle

            ## Update positions
            if agent.malfunction_handler.in_malfunction:
                movement_allowed = False
            else:
                # TODO check how the check motion is gestito, fai si che una reverse action sia sempre 
                # possibile ma attenzione quando c'è un treno vicino
                movement_allowed = self.motionCheck.check_motion(i_agent, agent.position) 


            movement_inside_cell = agent.state == TrainState.STOPPED and not agent.speed_counter.is_cell_exit
            movement_allowed = movement_allowed or movement_inside_cell

            # Fetch the saved transition data
            agent_transition_data = temp_transition_data[i_agent]
            preprocessed_action = agent_transition_data.preprocessed_action

            ## Update states
            state_transition_signals = self.generate_state_transition_signals(agent, preprocessed_action, movement_allowed, ending_time)
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
                    agent.state_machine.update_if_reached(agent.position, agent.target)

            # Off map or on map state and position should match
            env_utils.state_position_sync_check(agent.state, agent.position, agent.handle)

            # Handle done state actions, optionally remove agents
            self.handle_done_state(agent)
            
            if training == 'training0':
                if i_agent == 0:
                    have_all_agents_ended &= (agent.state == TrainState.DONE)

            elif training == 'training1' or training == 'training1.1':
                if i_agent < 2:
                    have_all_agents_ended &= (agent.state == TrainState.DONE)

            else:
                have_all_agents_ended &= (agent.state == TrainState.DONE)

            ## Update rewards 
            if agent.state != TrainState.MALFUNCTION and agent.state.is_on_map_state:                           
                self.update_step_rewards(i_agent)

            # The if condition is important to avoid multiple penalties due to malfunctions occurred in stations
            if agent.state != TrainState.MALFUNCTION and agent.speed_counter.is_cell_entry and agent.state.is_on_map_state:
                self.check_intermediate_station_passage(self._elapsed_steps, i_agent, optionals['agents_hints']['timetable'])

            ## Update counters (malfunction and speed)
            agent.speed_counter.update_counter(agent.state, agent.old_position)
                                            #    agent.state_machine.previous_state)
            agent.malfunction_handler.update_counter()

            # Clear old action when starting in new cell
            if agent.speed_counter.is_cell_entry and agent.position is not None:
                agent.action_saver.clear_saved_action()
        
        # Check if episode has ended and update rewards and dones
        self.end_of_episode_update(have_all_agents_ended, optionals['agents_hints']['timetable'])

        self._update_agent_positions_map()
        if self.record_steps:
            self.record_timestep(action_dict_)

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
                    train_velocities[i_agent] = min(1, agents_hints['timetable'][i_agent][2])                
                # Regional line case
                else:
                    train_velocities[i_agent] = min(1/2, agents_hints['timetable'][i_agent][2])

            # If agent is not in the environment deafault velocity is 1/2
            else:
                train_velocities[i_agent] = 1/2

        return train_velocities
