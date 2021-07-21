import time
import numpy as np
import os

# In Flatland you can use custom observation builders and predicitors
# Observation builders generate the observation needed by the controller
# Preditctors can be used to do short time prediction which can help in avoiding conflicts in the network
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters, ParamMalfunctionGen

from flatland.envs.observations import GlobalObsForRailEnv
# First of all we import the Flatland rail environment
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_original import RailEnvOriginal
from flatland.envs.rail_env import RailEnvActions

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_manual_specifications_generator, rail_custom_generator
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.utils.rendertools import RenderTool
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

from flatland.envs.schedule_generators import random_schedule_generator, custom_schedule_generator, complex_schedule_generator, control_timetable

# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)

specs = [[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],                          #1
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (8, 0), (1, 90), (1, 90), (1, 90),(2, 270), (10, 90), (1, 90), (10, 90), (2, 270), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (8, 90), (0, 0), (0, 0), (0, 0)],    #2
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (8, 0), (1, 90), (1, 90),(2, 90), (10, 270),  (1, 90), (10, 270),(2, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (8, 90), (1, 0), (0, 0), (0, 0), (0, 0)],      #3
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                          #4
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                          #5   
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                          #6
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),  (0,0),    (0,0),   (0,0),    (0,0),   (0,0),   (0,0),   (0,0),   (0,0),    (0,0),    (0,0),   (0,0),   (0,0),   (0,0),   (0,0),   (0,0),   (0,0),   (0,0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                          #7
		 [(0, 0) , (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (8, 0), (1, 90), (1, 90), (2, 270), (10, 90), (1, 90), (10, 90), (2, 270), (1, 90), (1, 90),(1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (2, 270), (10, 90), (1, 90), (1, 90), (10, 90), (2, 270), (1, 90), (1, 90),(8, 180), (1, 0),(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0),(0, 0)],                   #8
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0), (8, 0),  (1, 90), (2, 90), (10, 270), (1, 90),  (10, 270),(2, 90), (1, 90), (1, 90),(1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (2, 90), (10, 270), (10, 90), (2, 270), (10, 270),(2, 90), (1, 90), (1, 90), (1, 90),(8, 180), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0),(0, 0)],                #9
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0), (1, 0), (0, 0), (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0),  (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                        #10
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0), (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0),  (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                       #11
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0), (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0),  (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],  #12
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0), (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],   #13
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0), (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (2, 180), (2, 0), (0, 0), (0, 0), (0, 0)],   #14
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0), (0, 0), (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (10, 0), (10, 180), (0, 0), (0, 0), (0, 0)],    #15
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (6, 270), (1, 90), (1, 90), (7, 90)],  #16
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (10, 0),(10, 180), (0, 0), (0, 0), (0, 0)],  #17
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (10, 0),(10, 180),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (10, 0),(10, 180), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (2, 180), (2, 0), (0, 0), (0, 0), (0, 0)],  #18
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (2, 180), (2, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (2, 180), (2, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)], #19
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (1, 0),  (1, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],    #20
		 [(0, 0),  (8, 0), (1, 90),(1, 90),(1, 90),(10, 90),(2, 270),(2, 90),(10, 270),(2, 270), (10, 90), (1, 90),  (1, 90),  (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (2, 270), (10, 90), (2, 90),(10, 270), (10, 90), (2, 270), (1, 90), (1, 90),(1, 90),(1, 90), (1, 90),(1, 90), (1, 90), (2, 270), (10, 90), (1, 90), (10, 90), (2, 270), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (8, 180), (1, 0), (0, 0), (0, 0),(0, 0)],    #21
		 [(7, 270),(2, 90),(1, 90),(1, 90),(1, 90),(10, 270),(2, 90),(1, 90),(1, 90),(2, 90), (10, 270), (1, 90),  (1, 90),  (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (2, 90), (10, 270), (1, 90), (1, 90), (10, 270),(2, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90),(1, 90), (1, 90), (2, 90), (10, 270),  (1, 90), (10, 270),(2, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (8, 180), (0, 0), (0, 0),(0, 0)],    #22
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],   #23
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],   #24
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],   #25
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],   #26
		 [(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]]   #27

number_of_trains = 5

station_a = (21, 0)
station_b = (21, 36)
station_c = (8, 12)
stationo_d = (1, 35)
station_e = (15, 51)

# station_position = [[22, 0], [22, 20], [8, 12], [1, 32], [16, 49]]   #debug
station_position = [station_a, station_b, station_c, stationo_d, station_e]

target_stations = [station_c, station_a, station_b, station_e, stationo_d]


# GESTISCI ORARI IN MODO CHE ABBIANO SENSO CON L'ambiente
# TODO: controlli sulla fattibilità della timetable 

# Timetable conteins the station where the train should pass, from starting station to aim, and conteins the time at which
# each train has to pass in the station, the last number represent the velocity of train (high velocity, intercity or regional)
# Each row represent a different train

timetable = [[(station_a, station_b, station_c), (4 ,10, 20), (1.)],   		 # agent 0   high velocity
			 [(station_b, station_a),(0, 5), (1. / 2.)],                     # agent 1   Intercity
             [(station_c, station_b), (12, 15), (1. / 2.)],                  # agent 2   Intercity
             [(stationo_d, station_e), (20, 25), (1. / 4.)],                 # agent 3   Regional
             [(station_e, stationo_d), (14, 20), (1. / 4.)]]                 # agent 4   Regional


# Check if the timetable is feaseble or not, the function is in schedule_generators
# A timetable is feaseble if the difference of times between two stations is positive and let the trains to reach the successive station
# if two stations are very distant from each other the difference of times can't be very small
control_timetable(timetable,timetable,timetable)

# Different velocities in different lines.

speed_ration_map_lines = [1.       ,  # High velocity lines
						  1. / 2.  ]  # Regional lines

# Lines are represented by a rectangle between two stations
line_a_b = [[20, 0],
			[21, 36]] # high velocity

line_b_e = [[15, 36],
			[21, 51]] # high velocity

line_a_c = [[8, 0],
			[21, 12]] # regional

line_c_d = [[1, 12],
			[8, 35]] # regional

line_d_e = [[1, 35],
			[15, 51]] # regional


seed = 2

# Generating the railway topology, with stations
# Arguments of the generator (specs of the railway, position of stations, timetable)

rail_custom = rail_custom_generator(specs, station_position, timetable)

# We can now initiate the schedule generator with the given speed profiles
schedule_generator_custom = custom_schedule_generator()

TreeObservation = GlobalObsForRailEnv()

env = RailEnv(  width=40,
				height=30,
				rail_generator = rail_custom,
				schedule_generator=schedule_generator_custom,
				number_of_agents=5,
				obs_builder_object=TreeObservation,
				remove_agents_at_target=True,
				record_steps=True
				)

env.reset()

env_renderer = RenderTool(env,
						  screen_height=1080*2,
						  screen_width=1080*2)  # Adjust these parameters to fit your resolution



# Import your own Agent or use RLlib to train agents on Flatland
# As an example we use a random agent instead
class RandomAgent:

	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size

	def act(self, state):
		"""

		:param state: input is the observation of the agent
		:return: returns an action
		"""

		return(RailEnvActions.MOVE_FORWARD) # DEBUG, only move forward action for now
		
		#return np.random.choice([RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT, RailEnvActions.MOVE_LEFT, RailEnvActions.STOP_MOVING])

	def step(self, memories):
		"""

		Step function to improve agent by adjusting policy given the observations

		:param memories: SARS Tuple to be
		:return:
		"""

		return

	def save(self, filename):
		# Store the current policy
		return

	def load(self, filename):
		# Load a policy
		return



# Initialize the agent with the parameters corresponding to the environment and observation_builder
controller = RandomAgent(218, env.action_space[0])

# We start by looking at the information of each agent
# We can see the task assigned to the agent by looking at
print("\n Agents in the environment have to solve the following tasks: \n")
for agent_idx, agent in enumerate(env.agents):
	print(
		"The agent with index {} has the task to go from its initial position {}, facing in the direction {} to its target at {}.".format(
			agent_idx, agent.initial_position, agent.direction, agent.target))

# The agent will always have a status indicating if it is currently present in the environment or done or active
# For example we see that agent with index 0 is currently not active
print("\n Their current statuses are:")
print("============================")

for agent_idx, agent in enumerate(env.agents):
	print("Agent {} status is: {} with its current position being {}".format(agent_idx, str(agent.status),
																			 str(agent.position)))

# The agent needs to take any action [1,2,3] except do_nothing or stop to enter the level
# If the starting cell is free they will enter the level
# If multiple agents want to enter the same cell at the same time the lower index agent will enter first.

# Let's check if there are any agents with the same start location
agents_with_same_start = set()
print("\n The following agents have the same initial position:")
print("=====================================================")
for agent_idx, agent in enumerate(env.agents):
	for agent_2_idx, agent2 in enumerate(env.agents):
		if agent_idx != agent_2_idx and agent.initial_position == agent2.initial_position:
			print("Agent {} as the same initial position as agent {}".format(agent_idx, agent_2_idx))
			agents_with_same_start.add(agent_idx)

# Lets try to enter with all of these agents at the same time
action_dict = dict()

for agent_id in agents_with_same_start:
	action_dict[agent_id] = 1  # Try to move with the agents

# Do a step in the environment to see what agents entered:
env.step(action_dict)

# Current state and position of the agents after all agents with same start position tried to move
print("\n This happened when all tried to enter at the same time:")
print("========================================================")
for agent_id in agents_with_same_start:
	print(
		"Agent {} status is: {} with the current position being {}.".format(
			agent_id, str(env.agents[agent_id].status),
			str(env.agents[agent_id].position)))

# As you see only the agents with lower indexes moved. As soon as the cell is free again the agents can attempt
# to start again.

# You will also notice, that the agents move at different speeds once they are on the rail.
# The agents will always move at full speed when moving, never a speed inbetween.
# The fastest an agent can go is 1, meaning that it moves to the next cell at every time step
# All slower speeds indicate the fraction of a cell that is moved at each time step
# Lets look at the current speed data of the agents:

print("\n The speed information of the agents are:")
print("=========================================")

for agent_idx, agent in enumerate(env.agents):
	print(
		"Agent {} speed is: {:.2f} with the current fractional position being {}".format(
			agent_idx, agent.speed_data['speed'], agent.speed_data['position_fraction']))

# New the agents can also have stochastic malfunctions happening which will lead to them being unable to move
# for a certain amount of time steps. The malfunction data of the agents can easily be accessed as follows
print("\n The malfunction data of the agents are:")
print("========================================")

for agent_idx, agent in enumerate(env.agents):
	print(
		"Agent {} is OK = {}".format(
			agent_idx, agent.malfunction_data['malfunction'] < 1))

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

# Let us now look at an episode playing out with random actions performed

print("\nStart episode...")

# Reset the rendering system
env_renderer.reset()

# Here you can also further enhance the provided observation by means of normalization
# See training navigation example in the baseline repository


score = 0
# Run episode
frame_step = 0

# How many episodes
n_trials = 10

os.makedirs("tmp/frames", exist_ok=True)

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
	# Run episode
	for step in range(500):
		# Chose an action for each agent in the environment
		for a in range(env.get_num_agents()):
			action = controller.act(obs[a])
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


"""
class RandomAgent:

	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size

	def act(self, state):
		"""
"""
		:param state: input is the observation of the agent
		:return: returns an action
		"""
"""
		return np.random.choice(np.arange(self.action_size))
	def step(self, memories):
		"""
"""
		Step function to improve agent by adjusting policy given the observations

		:param memories: SARS Tuple to be
		:return:
		"""
"""
		return

	def save(self, filename):
		# Store the current policy
		return

	def load(self, filename):
		# Load a policy
		return


# Initialize the agent with the parameters corresponding to the environment and observation_builder
agent = RandomAgent(256, 5)
n_trials = 100

# Empty dictionary for all agent action
action_dict = dict()
print("Starting Training...")

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
	# Run episode
	for step in range(500):
		# Chose an action for each agent in the environment
		for a in range(env.get_num_agents()):
			action = agent.act(obs[a])
			action_dict.update({a: action})
		# Environment step which returns the observations for all agents, their corresponding
		# reward and whether their are done
		next_obs, all_rewards, done, _ = env.step(action_dict)
		env_renderer.render_env(show=True, show_observations=True, show_predictions=False)

		# Update replay buffer and train agent
		for a in range(env.get_num_agents()):
			agent.step((obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
			score += all_rewards[a]
		obs = next_obs.copy()
		if done['__all__']:
			break
	print('Episode Nr. {}\t Score = {}'.format(trials, score))

# uncomment to keep the renderer open
input("Press Enter to continue...")
"""

