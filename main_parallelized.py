import time
import numpy as np
import os
from argparse import Namespace
from pprint import pprint
from random import *
from datetime import datetime
from statistics import mean
import multiprocessing

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
n_episodes = 20000
eps_start = 1
eps_end = 0.01
eps_decay = 0.99992

max_steps = 300           # 1440 one day
checkpoint_interval = 100

mean_tolerance = 1     # Tolerance to compare the mean of the two windows of episodes
                       # this is important to discover if a plateau is present in the rewards distribution over the episodes
tolerance_of_conflict = 0.35     # Threshold of maximum percentage of conflicts that we accept                       

num_of_plateau = 0

plateau_window = 60

 # Unique ID for this training
now = datetime.now()
training_id = now.strftime('%y%m%d%H%M%S')


if __name__ == '__main__':
    training()