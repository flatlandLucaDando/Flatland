import numpy as np
from flatland.envs.observations import TreeObsForRailEnv

def max_lt(seq, val):
    """
    Return greatest item in seq for which item < val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    max = 0
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] < val and seq[idx] >= 0 and seq[idx] > max:
            max = seq[idx]
        idx -= 1
    return max


def min_gt(seq, val):
    """
    Return smallest item in seq for which item > val applies.
    None is returned if seq was empty or all items in seq were >= val.
    """
    min = np.inf
    idx = len(seq) - 1
    while idx >= 0:
        if seq[idx] >= val and seq[idx] < min:
            min = seq[idx]
        idx -= 1
    return min

def tree_normalization(obs):
    
    normalized_value = []
    maximum_value = max(obs)
    minimum_value = min(obs)
    normalized_value.append(np.clip(obs, minimum_value, maximum_value))
        
    return normalized_value   

def norm_obs_clip(obs, clip_min=-1, clip_max=1, fixed_radius=0, normalize_to_range=False):
    """
    This function returns the difference between min and max value of an observation
    :param obs: Observation that should be normalized
    :param clip_min: min value where observation will be clipped
    :param clip_max: max value where observation will be clipped
    :return: returnes normalized and clipped observatoin
    """
    if fixed_radius > 0:
        max_obs = fixed_radius
    else:
        max_obs = max(1, max_lt(obs, 1000)) + 1

    min_obs = 0  # min(max_obs, min_gt(obs, 0))
    if normalize_to_range:
        min_obs = min_gt(obs, 0)
    if min_obs > max_obs:
        min_obs = max_obs
    if max_obs == min_obs:
        return np.clip(np.array(obs) / max_obs, clip_min, clip_max)
    norm = np.abs(max_obs - min_obs)
    return np.clip((np.array(obs) - min_obs) / norm, clip_min, clip_max)


def _split_node_into_feature_groups(node):
    data = np.zeros(6)
    distance = np.zeros(1)
    agent_data = np.zeros(4)

    data[0] = node.dist_other_agent_encountered
    data[1] = node.dist_other_target_encountered
    data[2] = node.dist_other_agent_encountered
    data[3] = node.dist_potential_conflict
    data[4] = node.dist_unusable_switch
    data[5] = node.dist_to_next_branch

    distance[0] = node.dist_min_to_target

    agent_data[0] = node.num_agents_same_direction
    agent_data[1] = node.num_agents_opposite_direction
    agent_data[2] = node.num_agents_malfunctioning
    agent_data[3] = node.speed_min_fractional

    return data, distance, agent_data

def _split_node_into_feature_groups_new(node):
    data = np.zeros(10)
    agent_data = np.zeros(5)
    
    data[0] = node.dist_other_agent_encountered
    data[1] = node.dist_potential_conflict
    data[2] = node.dist_unusable_switch
    data[3] = node.dist_to_next_branch
    
    data[4] = node.station_positions
    data[5] = node.station_index
    data[6] = node.time_at_which_reach_station
    data[7] = node.station_positions_other_agent_0
    data[8] = node.station_index_other_agent_0
    data[9] = node.time_at_which_reach_station_other_agent_0

    agent_data[0] = node.num_agents_same_direction
    agent_data[1] = node.num_agents_opposite_direction
    agent_data[2] = node.num_agents_malfunctioning
    agent_data[3] = node.num_agents_ready_to_depart
    agent_data[4] = node.speed_min_fractional

    return data, agent_data


def _split_subtree_into_feature_groups(node, current_tree_depth: int, max_tree_depth: int):
    if node == -np.inf:
        remaining_depth = max_tree_depth - current_tree_depth
        # reference: https://stackoverflow.com/questions/515214/total-number-of-nodes-in-a-tree-data-structure
        num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
        return [-np.inf] * num_remaining_nodes * 10, [-np.inf] * num_remaining_nodes * 5

    data, agent_data = _split_node_into_feature_groups_new(node)

    if not node.childs:
        return data, agent_data

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_agent_data = _split_subtree_into_feature_groups(node.childs[direction], current_tree_depth + 1, max_tree_depth)
        data = np.concatenate((data, sub_data))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, agent_data


def split_tree_into_feature_groups(tree, max_tree_depth: int):
    """
    This function splits the tree into three difference arrays of values
    """
    data, agent_data = _split_node_into_feature_groups_new(tree)

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_agent_data = _split_subtree_into_feature_groups(tree.childs[direction], 1, max_tree_depth)
        data = np.concatenate((data, sub_data))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, agent_data


def normalize_observation(observation, tree_depth: int, observation_radius=0):
    """
    This function normalizes the observation used by the RL algorithm
    """
    data, agent_data = split_tree_into_feature_groups(observation, tree_depth)
    
    data = norm_obs_clip(data, fixed_radius=observation_radius)
    """data = tree_normalization(data)
    distance = tree_normalization(distance)"""
    agent_data = np.clip(agent_data, -1, 1)
    #normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data))
    normalized_obs = np.zeros(len(data)+ len(agent_data))
    for i in range(len(data)):
        normalized_obs[i] = data[i]
    for k in range(len(agent_data)):
        normalized_obs[k + len(data)] = agent_data[k]
    return normalized_obs

def normalize_global_observation(observation):
    observation_to_modify = observation
    number_of_features = len(observation_to_modify[0,0])
    for i in range(number_of_features):
        array_to_be_normalized = observation_to_modify[:,:,i]
        normalize_value = np.max(array_to_be_normalized)
        if normalize_value == 0:
            normalize_value = 0.0001
        array_normalized = np.zeros((len(array_to_be_normalized),len(array_to_be_normalized[0])))
        for j in range(len(array_to_be_normalized)):
            for k in range(len(array_to_be_normalized[j])):
                array_normalized[j][k] = array_to_be_normalized[j][k]/normalize_value
        observation_to_modify[:,:,i] = array_normalized
        
    observation_normalized = observation_to_modify.flatten()
    
    return observation_normalized
        
        