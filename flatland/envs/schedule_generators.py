"""Schedule generators (railway undertaking, "EVU")."""
import warnings
from typing import Tuple, List, Callable, Mapping, Optional, Any

import numpy as np
from enum import IntEnum
from numpy.random.mtrand import RandomState

from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.schedule_utils import Schedule
from flatland.envs import persistence

# This is to test if the timetable is valid or not
from flatland.core.grid.grid4_astar import a_star

from structures import railway_example, stations, timetable_example, av_line

AgentPosition = Tuple[int, int]
ScheduleGenerator = Callable[[GridTransitionMap, int, Optional[Any], Optional[int]], Schedule]

class RailEnvActions(IntEnum):
    DO_NOTHING = 0  # implies change of direction in a dead-end!
    MOVE_LEFT = 1
    MOVE_FORWARD = 2
    MOVE_RIGHT = 3
    STOP_MOVING = 4

    @staticmethod
    def to_char(a: int):
        return {
            0: 'B',
            1: 'L',
            2: 'F',
            3: 'R',
            4: 'S',
        }[a]

def speed_initialization_helper(nb_agents: int, speed_ratio_map: Mapping[float, float] = None,
                                seed: int = None, np_random: RandomState = None) -> List[float]:
    """
    Parameters
    ----------
    nb_agents : int
        The number of agents to generate a speed for
    speed_ratio_map : Mapping[float,float]
        A map of speeds mappint to their ratio of appearance. The ratios must sum up to 1.

    Returns
    -------
    List[float]
        A list of size nb_agents of speeds with the corresponding probabilistic ratios.
    """
    if speed_ratio_map is None:
        return [1.0] * nb_agents

    nb_classes = len(speed_ratio_map.keys())
    speed_ratio_map_as_list: List[Tuple[float, float]] = list(speed_ratio_map.items())
    speed_ratios = list(map(lambda t: t[1], speed_ratio_map_as_list))
    speeds = list(map(lambda t: t[0], speed_ratio_map_as_list))
    print (nb_classes, nb_agents, speed_ratios)
    return list(map(lambda index: speeds[index], np_random.choice(nb_classes, nb_agents, p=speed_ratios)))


# Check if the timetable is feaseble or not
def control_timetable(timetable, railway_topology):
    # Check for all the trains
    for trains in range (len(timetable)):       
        # Check for all the stations
        # Calculate the difference of two different times, so i don't need the last term to cycle          
        for stations in range (len(timetable[trains][1]) - 1):   
            if (timetable[trains][1][stations] - timetable[trains][1][stations + 1]) >= 0:
                print('===================================================================================================================================')
                print('Attention!!! The agent number', trains, 'has a problem in the timetable, times to reach stations', stations, 'and', (stations+1), 'are not right')
                print('The time to reach the successive station SHOULD BE > 0, pay attenction to the timetable')
                # Function that check if the time to reach a station defined by the timetable are possible or not,
                # Return the time minimum time to reach two different stations depending on the distance and on the line type (high velocity, regional...)
            time_to_next_station = time_to_reach_next_station(timetable[trains][0][stations], timetable[trains][0][stations + 1], railway_topology, timetable_example, trains)
            # Control if the time to reach the next station is possible (considering maximum velocities of lines and the distances between two stations)
            if time_to_next_station > (timetable[trains][1][stations+1]- timetable[trains][1][stations]):
                print('===================================================================================================================================')
                print('Attention!!! Agent number', trains, 'has a problem in the timetable, times to reach stations', stations, 'and', (stations+1), 'are not right')
                print('The time to reach the next station SHOULD BE HIGHER. The minimum time to reach the station should be:', time_to_next_station)
    return

# TODO make action to do for each station!!!

# Define the scheduled actions the agents have to do
def action_to_do(timetable, railway_topology):
    # Path to do to arrive to the right station
    path_result = []
    # Calculate the path for all the trains
    for train_i in range (len(timetable)):
        path_result.append(a_star(railway_topology,timetable[train_i][0][0],timetable[train_i][0][-1]))

    # DEBUG
    for train_i in range(len(timetable)):
        print()
        print(path_result[train_i])
        print()
        
    # Calculate the actions that have to be done
    actions_to_do = []
    for train_i in range (len(timetable)):
        # Flag that tells me that the next step is particular
        next = False
        # Each train occupy a row in the action_to_do matrix 
        actions_single_train = []
        for step in range (len(path_result[train_i])):
            # Calculate the direction of the trains at each step
            if step == 0:
                difference_y = path_result[train_i][step][0] - path_result[train_i][step + 1][0]
                difference_x = path_result[train_i][step][1] - path_result[train_i][step + 1][1]
                if difference_y == 1:
                    direction = 0
                if difference_x ==  -1:
                    direction = 1
                if difference_y == -1:
                    direction = 2
                if difference_x == 1:
                    direction = 3 
            else:
                difference_y = path_result[train_i][step - 1][0] - path_result[train_i][step][0]
                difference_x = path_result[train_i][step - 1][1] - path_result[train_i][step][1]
                if difference_y == 1:
                    direction = 0
                if difference_x ==  -1:
                    direction = 1
                if difference_y == -1:
                    direction = 2
                if difference_x == 1:
                    direction = 3 
            # Variable to count the number of possible path at each cell, is an int with the number of possible path
            if not step == 0:
                # Specific case, a train is at the boarder of two different lines, 
                # if this appen I have to consider the previous transition at the next time stamp due to the fact the velocity changes
                if next:
                    multiple_path = railway_topology.get_transitions(path_result[train_i][step-1][0],path_result[train_i][step-1][1],prev_direction).count(1)
                    next = False
                elif (path_result[train_i][step] in av_line) and not (path_result[train_i][step - 1] in av_line):
                    multiple_path = railway_topology.get_transitions(path_result[train_i][step][0],path_result[train_i][step][1],prev_direction).count(1)
                    next = True
                else:
                    multiple_path = railway_topology.get_transitions(path_result[train_i][step-1][0],path_result[train_i][step-1][1],prev_direction).count(1)
            # Starting with a move forward direction for the train
            if step == 0:
                #actions_single_train.append(RailEnvActions.MOVE_FORWARD)
                prev_direction = direction
            # If I'm not at the start of the train 
            else:
                # The direction doesn't change
                if direction - prev_direction == 0:
                    # If I'm in an hig velocity line velocity is define only by the type of train
                    if path_result[train_i][step - 1] in av_line:
                        velocity = timetable[train_i][2]
                    # If I'm in other line velocity is the minimum between 1/2 (the velocity of the line) and the type of train velocity
                    else:
                        velocity = min(timetable[train_i][2], 1/2)
                    for i in range(int(pow(velocity, -1))):
                        #print('Test per capire come varia',i, 'Treno numero', train_i)
                        actions_single_train.append(RailEnvActions.MOVE_FORWARD)
                    prev_direction = direction
                # I have to move to left 
                # and I have more then one possible path, so I go left at the deviation
                # Depending on the direction of march the results can be -1 or -3
                elif ((direction - prev_direction == -1) and (multiple_path > 1)) or ((direction - prev_direction == +3)):
                    # If I'm in an hig velocity line velocity is define only by the type of train
                    if path_result[train_i][step - 1] in av_line:
                        velocity = timetable[train_i][2]
                    # If I'm in other line velocity is the minimum between 1/2 (the velocity of the line) and the type of train velocity
                    else:
                        velocity = min(timetable[train_i][2], 1/2)
                    for i in range(int(pow(velocity, -1))):
                        actions_single_train.append(RailEnvActions.MOVE_LEFT)
                    prev_direction = direction
                # I have to move right 
                # and I have more then one possible path, so I go left at the deviation 
                # Depending on the direction of march the results can be +1 or -3
                elif ((direction - prev_direction == 1) and (multiple_path > 1)) or ((direction - prev_direction == -3) ):
                    # If I'm in an hig velocity line velocity is define only by the type of train
                    if path_result[train_i][step - 1] in av_line:
                        velocity = timetable[train_i][2]
                    # If I'm in other line velocity is the minimum between 1/2 (the velocity of the line) and the type of train velocity
                    else:
                        velocity = min(timetable[train_i][2], 1/2)
                    for i in range(int(pow(velocity, -1))):
                        actions_single_train.append(RailEnvActions.MOVE_RIGHT)
                    prev_direction = direction
                else:
                    if path_result[train_i][step - 1] in av_line:
                        velocity = timetable[train_i][2]
                    else:
                        velocity = min(timetable[train_i][2], 1/2)
                    for i in range(int(pow(velocity, -1))):
                        actions_single_train.append(RailEnvActions.MOVE_FORWARD)
                    prev_direction = direction
        #print(len(actions_single_train))
        actions_to_do.append(actions_single_train)
    return actions_to_do



# Calculate the time to reach the stations to understand if timetable is right
def time_to_reach_next_station(departure_station_position, arrival_station_position, railway_topology, schedule, train_number):
    # First thing check the distance between two stations 
    result = a_star(railway_topology, departure_station_position, arrival_station_position)
    # Maximum velocity a train can achieve
    train_velocity = schedule[train_number][2]

    lenght_path = len(result)  # distance between stations

    # Array when I put at each step the time needed to make the path
    # The total time is the sum of the numbers
    time_array = []
    # Check the at each step which train i am and which line im in
    for step in range(lenght_path):
        if (result[step]) in av_line:
            time_array.append(pow(train_velocity,-1))
        else:
            time_array.append(pow(min(train_velocity, 1/2), -1))
    time_needed = sum(time_array)

    #print((time_needed + int(time_needed/10))) DEBUG

    # Adding to the time a 10% to face with problems in case it's neaded
    return (time_needed + int(time_needed/10))


class BaseSchedGen(object):
    def __init__(self, speed_ratio_map: Mapping[float, float] = None, seed: int = 1):
        self.speed_ratio_map = speed_ratio_map
        self.seed = seed

    def generate(self, rail: GridTransitionMap, num_agents: int, hints: Any=None, num_resets: int = 0,
        np_random: RandomState = None) -> Schedule:
        pass

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

def custom_schedule_generator(speed_ratio_map: Mapping[float, float] = None, seed: int = 1) -> ScheduleGenerator:

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
        for agent_i in range (len(timetable_example)):
            agents_position.append(timetable_example[agent_i][0][0])
            agents_target.append(timetable_example[agent_i][0][-1])


        # Define the direction of the trains based on the rail they occupy
        # Input --> the topology of the network, the position of the trains
        # Output --> an array with the directions of the trains
        # DIRECTIONS: 0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT

        agents_direction = check_rail_road_direction(rail, timetable_example)


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

def complex_schedule_generator(speed_ratio_map: Mapping[float, float] = None, seed: int = 1) -> ScheduleGenerator:
    """

    Generator used to generate the levels of Round 1 in the Flatland Challenge. It can only be used together
    with complex_rail_generator. It places agents at end and start points provided by the rail generator.
    It assigns speeds to the different agents according to the speed_ratio_map
    :param speed_ratio_map: Speed ratios of all agents. They are probabilities of all different speeds and have to
            add up to 1.
    :param seed: Initiate random seed generator
    :return:
    """

    def generator(rail: GridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0,
                  np_random: RandomState = None) -> Schedule:
        """

        The generator that assigns tasks to all the agents
        :param rail: Rail infrastructure given by the rail_generator
        :param num_agents: Number of agents to include in the schedule
        :param hints: Hints provided by the rail_generator These include positions of start/target positions
        :param num_resets: How often the generator has been reset.
        :return: Returns the generator to the rail constructor
        """
        # Todo: Remove parameters and variables not used for next version, Issue: <https://gitlab.aicrowd.com/flatland/flatland/issues/305>
        _runtime_seed = seed + num_resets

        start_goal = hints['start_goal']
        start_dir = hints['start_dir']
        #print(start_goal[:num_agents])
        agents_position = [sg[0] for sg in start_goal[:num_agents]]
        agents_target = [sg[1] for sg in start_goal[:num_agents]]
        agents_direction = start_dir[:num_agents]

        if speed_ratio_map:
            speeds = speed_initialization_helper(num_agents, speed_ratio_map, seed=_runtime_seed, np_random=np_random)
        else:
            speeds = [1.0] * len(agents_position)
        # Compute max number of steps with given schedule
        extra_time_factor = 1.5  # Factor to allow for more then minimal time
        max_episode_steps = 1000 #int(extra_time_factor * rail.height * rail.width)

        return Schedule(agent_positions=agents_position, agent_directions=agents_direction,
                        agent_targets=agents_target, agent_speeds=speeds, agent_malfunction_rates=None,
                        max_episode_steps=max_episode_steps)

    return generator


def sparse_schedule_generator(speed_ratio_map: Mapping[float, float] = None, seed: int = 1) -> ScheduleGenerator:
    return SparseSchedGen(speed_ratio_map, seed)


class SparseSchedGen(BaseSchedGen):
    """

    This is the schedule generator which is used for Round 2 of the Flatland challenge. It produces schedules
    to railway networks provided by sparse_rail_generator.
    :param speed_ratio_map: Speed ratios of all agents. They are probabilities of all different speeds and have to
            add up to 1.
    :param seed: Initiate random seed generator
    """

    def generate(self, rail: GridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0,
                  np_random: RandomState = None) -> Schedule:
        """

        The generator that assigns tasks to all the agents
        :param rail: Rail infrastructure given by the rail_generator
        :param num_agents: Number of agents to include in the schedule
        :param hints: Hints provided by the rail_generator These include positions of start/target positions
        :param num_resets: How often the generator has been reset.
        :return: Returns the generator to the rail constructor
        """

        _runtime_seed = self.seed + num_resets

        train_stations = hints['train_stations']
        city_positions = hints['city_positions']
        city_orientation = hints['city_orientations']
        max_num_agents = hints['num_agents']
        city_orientations = hints['city_orientations']
        if num_agents > max_num_agents:
            num_agents = max_num_agents
            warnings.warn("Too many agents! Changes number of agents.")
        # Place agents and targets within available train stations
        agents_position = []
        agents_target = []
        agents_direction = []

        for agent_idx in range(num_agents):
            infeasible_agent = True
            tries = 0
            while infeasible_agent:
                tries += 1
                infeasible_agent = False
                # Set target for agent
                city_idx = np_random.choice(len(city_positions), 2, replace=False)
                start_city = city_idx[0]
                target_city = city_idx[1]

                start_idx = np_random.choice(np.arange(len(train_stations[start_city])))
                target_idx = np_random.choice(np.arange(len(train_stations[target_city])))
                start = train_stations[start_city][start_idx]
                target = train_stations[target_city][target_idx]

                while start[1] % 2 != 0:
                    start_idx = np_random.choice(np.arange(len(train_stations[start_city])))
                    start = train_stations[start_city][start_idx]
                while target[1] % 2 != 1:
                    target_idx = np_random.choice(np.arange(len(train_stations[target_city])))
                    target = train_stations[target_city][target_idx]
                possible_orientations = [city_orientation[start_city],
                                         (city_orientation[start_city] + 2) % 4]
                agent_orientation = np_random.choice(possible_orientations)
                if not rail.check_path_exists(start[0], agent_orientation, target[0]):
                    agent_orientation = (agent_orientation + 2) % 4
                if not (rail.check_path_exists(start[0], agent_orientation, target[0])):
                    infeasible_agent = True
                if tries >= 100:
                    warnings.warn("Did not find any possible path, check your parameters!!!")
                    break
            agents_position.append((start[0][0], start[0][1]))
            agents_target.append((target[0][0], target[0][1]))

            agents_direction.append(agent_orientation)
            # Orient the agent correctly

        if self.speed_ratio_map:
            speeds = speed_initialization_helper(num_agents, self.speed_ratio_map, seed=_runtime_seed, np_random=np_random)
        else:
            speeds = [1.0] * len(agents_position)

        # We add multiply factors to the max number of time steps to simplify task in Flatland challenge.
        # These factors might change in the future.
        timedelay_factor = 4
        alpha = 2
        max_episode_steps = int(
            timedelay_factor * alpha * (rail.width + rail.height + num_agents / len(city_positions)))

        return Schedule(agent_positions=agents_position, agent_directions=agents_direction,
                        agent_targets=agents_target, agent_speeds=speeds, agent_malfunction_rates=None,
                        max_episode_steps=max_episode_steps)


def random_schedule_generator(speed_ratio_map: Mapping[float, float] = None, seed: int = 1) -> ScheduleGenerator:
    return RandomSchedGen(speed_ratio_map, seed)


class RandomSchedGen(BaseSchedGen):
    
    """
    Given a `rail` GridTransitionMap, return a random placement of agents (initial position, direction and target).

    Parameters
    ----------
        speed_ratio_map : Optional[Mapping[float, float]]
            A map of speeds mapping to their ratio of appearance. The ratios must sum up to 1.

    Returns
    -------
        Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]], List[float]]
            initial positions, directions, targets speeds
    """

    def generate(self, rail: GridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0,
                  np_random: RandomState = None) -> Schedule:
        _runtime_seed = self.seed + num_resets

        #print(num_agents)

        valid_positions = []
        for r in range(rail.height):
            for c in range(rail.width):
                if rail.get_full_transitions(r, c) > 0:
                    valid_positions.append((r, c))
        if len(valid_positions) == 0:
            return Schedule(agent_positions=[], agent_directions=[],
                            agent_targets=[], agent_speeds=[], agent_malfunction_rates=None, max_episode_steps=0)

        if len(valid_positions) < num_agents:
            warnings.warn("schedule_generators: len(valid_positions) < num_agents")
            return Schedule(agent_positions=[], agent_directions=[],
                            agent_targets=[], agent_speeds=[], agent_malfunction_rates=None, max_episode_steps=0)

        agents_position_idx = [i for i in np_random.choice(len(valid_positions), num_agents, replace=False)]
        agents_position = [valid_positions[agents_position_idx[i]] for i in range(num_agents)]
        agents_target_idx = [i for i in np_random.choice(len(valid_positions), num_agents, replace=False)]
        agents_target = [valid_positions[agents_target_idx[i]] for i in range(num_agents)]
        update_agents = np.zeros(num_agents)

        re_generate = True
        cnt = 0
        while re_generate:
            cnt += 1
            if cnt > 1:
                print("re_generate cnt={}".format(cnt))
            if cnt > 1000:
                raise Exception("After 1000 re_generates still not success, giving up.")
            # update position
            for i in range(num_agents):
                if update_agents[i] == 1:
                    x = np.setdiff1d(np.arange(len(valid_positions)), agents_position_idx)
                    agents_position_idx[i] = np_random.choice(x)
                    agents_position[i] = valid_positions[agents_position_idx[i]]
                    x = np.setdiff1d(np.arange(len(valid_positions)), agents_target_idx)
                    agents_target_idx[i] = np_random.choice(x)
                    agents_target[i] = valid_positions[agents_target_idx[i]]
            update_agents = np.zeros(num_agents)

            # agents_direction must be a direction for which a solution is
            # guaranteed.
            agents_direction = [0] * num_agents
            re_generate = False
            for i in range(num_agents):
                valid_movements = []
                for direction in range(4):
                    position = agents_position[i]
                    moves = rail.get_transitions(position[0], position[1], direction)
                    for move_index in range(4):
                        if moves[move_index]:
                            valid_movements.append((direction, move_index))

                valid_starting_directions = []
                for m in valid_movements:
                    new_position = get_new_position(agents_position[i], m[1])
                    if m[0] not in valid_starting_directions and rail.check_path_exists(new_position, m[1],
                                                                                        agents_target[i]):
                        valid_starting_directions.append(m[0])

                if len(valid_starting_directions) == 0:
                    update_agents[i] = 1
                    warnings.warn(
                        "reset position for agent[{}]: {} -> {}".format(i, agents_position[i], agents_target[i]))
                    re_generate = True
                    break
                else:
                    agents_direction[i] = valid_starting_directions[
                        np_random.choice(len(valid_starting_directions), 1)[0]]

        agents_speed = speed_initialization_helper(num_agents, self.speed_ratio_map, seed=_runtime_seed, 
            np_random=np_random)

        # Compute max number of steps with given schedule
        extra_time_factor = 1.5  # Factor to allow for more then minimal time
        max_episode_steps = int(extra_time_factor * rail.height * rail.width)

        return Schedule(agent_positions=agents_position, agent_directions=agents_direction,
                        agent_targets=agents_target, agent_speeds=agents_speed, agent_malfunction_rates=None,
                        max_episode_steps=max_episode_steps)



def schedule_from_file(filename, load_from_package=None) -> ScheduleGenerator:
    """
    Utility to load pickle file

    Parameters
    ----------
    input_file : Pickle file generated by env.save() or editor

    Returns
    -------
    Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]], List[float]]
        initial positions, directions, targets speeds
    """

    def generator(rail: GridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0,
                  np_random: RandomState = None) -> Schedule:

        env_dict = persistence.RailEnvPersister.load_env_dict(filename, load_from_package=load_from_package)

        max_episode_steps = env_dict.get("max_episode_steps", 0)
        if (max_episode_steps==0):
            print("This env file has no max_episode_steps (deprecated) - setting to 100")
            max_episode_steps = 100
            
        agents = env_dict["agents"]

        #print("schedule generator from_file - agents: ", agents)

        # setup with loaded data
        agents_position = [a.initial_position for a in agents]

        # this logic is wrong - we should really load the initial_direction as the direction.
        #agents_direction = [a.direction for a in agents]
        agents_direction = [a.initial_direction for a in agents]
        agents_target = [a.target for a in agents]
        agents_speed = [a.speed_data['speed'] for a in agents]

        # Malfunctions from here are not used.  They have their own generator.
        #agents_malfunction = [a.malfunction_data['malfunction_rate'] for a in agents]

        return Schedule(agent_positions=agents_position, agent_directions=agents_direction,
                        agent_targets=agents_target, agent_speeds=agents_speed, 
                        agent_malfunction_rates=None,
                        max_episode_steps=max_episode_steps)

    return generator



def check_rail_road_direction(rail: GridTransitionMap, timetable):
    # To establish the direction of trains in the railroas I define a simple law, as for the cars, each trains has to 
    # go the direction that let them to have the right free

    agents_direction = [0]*len(timetable)
    path_result = [0]*len(timetable)

    #print(rail.grid[7 ,13],rail.grid[7 ,12],rail.grid[7 ,11],rail.grid[7 ,10],rail.grid[7 ,9],rail.grid[7 ,8],rail.grid[7 ,7],rail.grid[7 ,6], rail.grid[7 ,5])

    for i in range (len(timetable)):
        # Consider the a_star result to calculate the direction
        path_result[i] = (a_star(rail,timetable[i][0][0],timetable[i][0][-1]))

        difference_x = path_result[i][0][1] - path_result[i][1][1]
        difference_y = path_result[i][0][0] - path_result[i][1][0]
        if difference_y == 1:
            agents_direction[i] = 0
        if difference_x ==  -1:
            agents_direction[i] = 1
        if difference_y == -1:
            agents_direction[i] = 2
        if difference_x == 1:
            agents_direction[i] = 3

    return agents_direction
