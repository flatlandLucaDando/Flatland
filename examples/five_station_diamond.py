from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap

'''
###############################################################
##################   5 stations diamond  ######################
###############################################################
'''

# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)

# Example 2 simple railway with 2 stations and two rails and multiple rails in the stations
railway_example = [[(0,0)] * 25,
				   [(0,0)] * 5 + [(8,0)] + [(1,90)]*13 + [(8,90)] + [(0,0)]*5,
				   [(0,0)] * 5 + [(1,0)] + [(0, 0)]*13 + [(1,0)] + [(0,0)] * 5,
				   [(0,0)] * 5 + [(1, 0)] + [(0, 0)]*13 + [(1,0)] + [(0,0)] * 5,
				   [(7, 270)] + [(1,90)] * 3 + [(1,90)] + [(12, 180)] + [(1,90)] + [(12, 0)] + [(1,90)]*9 + [(12,0)] + [(1,90)] + [(12, 180)] + [(1,90)] + [(1,90)] * 3 + [(7, 90)],
				   [(0,0)] * 7  + [(1,0)] + [(0,0)] * 9 + [(1,0)] + [(0,0)] * 7,
				   [(0,0)] * 7  + [(1,0)] + [(0,0)] * 9 + [(1,0)] + [(0,0)] * 7,
                   [(0,0)] * 7 + [(8,270)] + [(1,90)]*9 + [(8,180)] + [(0,0)] * 7,
                   [(0,0)] * 25]

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

# No high velocity lines, so make a (0,0) position
	av_line = (0,0)
	
	# TODO sistema i binari destra e sinistra e su e giù
	# Rails where the direction is right
	right_rails = [(0,0)]
	# Rails where the direction is left
	left_rails = [(0,0)]
	down_rails = [(0,0)]
	up_rails = [(0,0)]
