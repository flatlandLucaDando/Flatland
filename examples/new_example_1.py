from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap

'''
###############################################################
######################   EXAMPLE 1  #########################
###############################################################
'''

# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)

# Example 1 simple railway with 2 stations and one rail
railway_example = [[(0,0)]*12,                                                      						 					# 0
				   [(0,0)]*12,                                                      						 					# 1
				   [(0,0)] + [(7,270)] + [(1,90)]*2 + [(8,90)] + [(0,0)]*2 + [(8,0)] + [(1,90)]*2 + [(7,90)] + [(0,0)],         # 2
				   [(0,0)] + [(7,270)] + [(1,90)]*2 + [(10,270)] + [(1,90)]*2 + [(2,90)] + [(1,90)]*2 + [(7,90)] + [(0,0)],     # 3
				   [(0,0)]*12,                                                              				 					# 4
				   [(0,0)]*12]                                                       						 					# 5

# wheight and height of the grid
height = len(railway_example)
width = len(railway_example[0])

# creating the transition map
rail_env_transitions = RailEnvTransitions()
rail = GridTransitionMap(width=width, height=height, transitions=rail_env_transitions)

for r in range(height):
	for c in range(width):
		rail_spec_of_cell = railway_example[r][c]
		index_basic_type_of_cell_ = rail_spec_of_cell[0]
		rotation_cell_ = rail_spec_of_cell[1]
		if index_basic_type_of_cell_ < 0 or index_basic_type_of_cell_ >= len(rail_env_transitions.transitions):
			print("ERROR - invalid rail_spec_of_cell type=", index_basic_type_of_cell_)
		basic_type_of_cell_ = rail_env_transitions.transitions[index_basic_type_of_cell_]
		effective_transition_cell = rail_env_transitions.rotate_transition(basic_type_of_cell_, rotation_cell_)
		rail.set_transitions((r, c), effective_transition_cell)

# One rail, so no right or left rails  
right_rails = [(0,0)]
left_rails = [(0,0)]
down_rails = [(0,0)]
up_rails = [(0,0)]

# No high velocity lines, so make a (0,0) position
av_line = (0,0)
