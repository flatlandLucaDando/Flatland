from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.line_generators import line_from_file
from flatland.envs.rail_env import RailEnvActions


def load_flatland_environment_from_file(file_name: str,
                                        load_from_package: str = None,
                                        obs_builder_object: ObservationBuilder = None,
                                        record_steps = False,
                                        ) -> RailEnv:
    """
    Parameters
    ----------
    file_name : str
        The pickle file.
    load_from_package : str
        The python module to import from. Example: 'env_data.tests'
        This requires that there are `__init__.py` files in the folder structure we load the file from.
    obs_builder_object: ObservationBuilder
        The obs builder for the `RailEnv` that is created.


    Returns
    -------
    RailEnv
        The environment loaded from the pickle file.
    """
    if obs_builder_object is None:
        obs_builder_object = TreeObsForRailEnv(
            max_depth=2,
            predictor=ShortestPathPredictorForRailEnv(max_depth=10))
    environment = RailEnv(width=1, height=1, rail_generator=rail_from_file(file_name, load_from_package),
                          line_generator=line_from_file(file_name, load_from_package),
                          number_of_agents=1,
                          obs_builder_object=obs_builder_object,
                          record_steps=record_steps,
                          )
    return environment

# TODO riferisciti a un agente (EnvAgent)
# Usa le azioni dell'agente specifico
def delay_a_train(delay: int, delay_time: int, time_of_train_generation: int, actions, train):
    
    i_agent = train.handle
    train_velocity = train.speed_counter.speed

    
    actions_scheduled = [0] * (len(actions[i_agent]) + delay)
    
    # Copy the actions scheduled for the train before the delay
    for i in range(delay_time - time_of_train_generation):
        actions_scheduled[i] = actions[i_agent][i]
    # Delay the train (STOP)  
    for i in range(delay):
        if (i + delay_time - time_of_train_generation) < 0 or (delay_time - time_of_train_generation) > len(actions[i_agent]):
            raise ImportError('The train is not present in the environment, check the delay time')
        actions_scheduled[i + delay_time - time_of_train_generation] = RailEnvActions.STOP_MOVING
    # Copy the actions scheduled for the train after the delay
    for i in range(len(actions[i_agent]) - (delay_time - time_of_train_generation)):
        actions_scheduled[i + delay_time - time_of_train_generation + delay] = actions[i_agent][i + delay_time - time_of_train_generation]
    
    actions[i_agent] = actions_scheduled
    return 


# Function to convert decimal number to base number
def actions_decimal_to_base(base, number_to_convert, num_agents):
    division = number_to_convert
    result = []
    while division != 0:
        result.append(division % base)
        division = int(division / base)
    actions_to_perform = result[::-1]
    if len(actions_to_perform) < num_agents:
        zero_to_add = num_agents - len(actions_to_perform)
        for i in range(zero_to_add):
            actions_to_perform.insert(0,0)
    return actions_to_perform
