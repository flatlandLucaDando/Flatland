from typing import List, NamedTuple

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid_utils import IntVector2DArray

Schedule = NamedTuple('Schedule', [('agent_positions', IntVector2DArray),
                                   ('agent_directions', List[Grid4TransitionsEnum]),
                                   ('agent_targets', IntVector2DArray),
                                   ('agent_speeds', List[float]),
                                   ('agent_malfunction_rates', List[int]),
                                   ('max_episode_steps', int)])

# The custom schedule add the intermediate station in which the train should pass at the right time to respect the timetable
Schedule_custom = NamedTuple ('Schedule_custom', [('agent_positions', IntVector2DArray),
                                                  ('agent_directions', List[Grid4TransitionsEnum]),
                                                  ('agent_targets', IntVector2DArray),
                                                  ('intermediate_stations', List[int]),  #intermediate station to pass with the time at which the train need to pass
                                                  ('agent_speeds', List[float]),
                                                  ('agent_malfunction_rates', List[int]),
                                                  ('max_episode_steps', int)])
