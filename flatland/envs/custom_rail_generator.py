"""Rail generators (infrastructure manager, "Infrastrukturbetreiber")."""
from typing import Callable, Tuple, Optional, Dict, List

from numpy.random.mtrand import RandomState

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap


RailGeneratorProduct = Tuple[GridTransitionMap, Optional[Dict]]
""" A rail generator returns a RailGenerator Product, which is just
    a GridTransitionMap followed by an (optional) dict/
"""

RailGenerator = Callable[[int, int, int, int], RailGeneratorProduct]

# Number of agents are given by timetables (the num of the rows), target stations are also given by timetable 
def rail_custom_generator(rail_spec, train_stations_position: list = None, timetable: list = None):

    """
    Utility to convert a rail given by manual specification as a map of tuples
    (cell_type, rotation), to a transition map with the correct 16-bit
    transitions specifications.

    Parameters
    ----------
    rail_spec : list of list of tuples
        List (rows) of lists (columns) of tuples, each specifying a rail_spec_of_cell for
        the RailEnv environment as (cell_type, rotation), with rotation being
        clock-wise and in [0, 90, 180, 270].

    Returns
    -------
    function
        Generator function that always returns a GridTransitionMap object with
        the matrix of correct 16-bit bitmaps for each rail_spec_of_cell.

    New features:
    -------------
        Train station poition: to define the position of the station
        Target station: Define the target of the different agents
        Timetable: timetable contein the intermediate station and the time at which pass across them
    """

    def custom_generator(width: int, height: int, num_agents: int, num_resets: int = 0,
                  np_random: RandomState = None) -> RailGenerator:

        # All the cities are oriented in the same way in my model
        city_orientations = 1

        # Taking the number of agents from timetable
        num_of_agents = len(timetable)

        # Taking the target stations from timetable
        target_stations = []

        for agent in range(num_agents):
            target_stations.append(timetable[agent][-1])
      
        rail_env_transitions = RailEnvTransitions()

        height = len(rail_spec)
        width = len(rail_spec[0])
        rail = GridTransitionMap(width=width, height=height, transitions=rail_env_transitions)

        for r in range(height):
            for c in range(width):
                rail_spec_of_cell = rail_spec[r][c]
                index_basic_type_of_cell_ = rail_spec_of_cell[0]
                rotation_cell_ = rail_spec_of_cell[1]
                if index_basic_type_of_cell_ < 0 or index_basic_type_of_cell_ >= len(rail_env_transitions.transitions):
                    print("ERROR - invalid rail_spec_of_cell type=", index_basic_type_of_cell_)
                    return []
                basic_type_of_cell_ = rail_env_transitions.transitions[index_basic_type_of_cell_]
                effective_transition_cell = rail_env_transitions.rotate_transition(basic_type_of_cell_, rotation_cell_)
                rail.set_transitions((r, c), effective_transition_cell)

        return rail,  {'agents_hints': {
            'num_agents': num_of_agents,            
            'city_positions': train_stations_position,
            'train_stations': train_stations_position,
            'city_orientations': city_orientations,
            'targets' : target_stations,
            'timetable' : timetable 
        }}

    return custom_generator
