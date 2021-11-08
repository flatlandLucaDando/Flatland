from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap

'''
###############################################################
######################   EXAMPLE 2  #########################
###############################################################
'''

# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)

# Example 2 simple railway with 2 stations and two rails and multiple rails in the stations
railway_example = [[(0,0)] * 20,
				   [(0,0)] * 20,
				   [(7, 270)] + [(1,90)] * 3 + [(8,90)] + [(0,0)] * 10 + [(8,0)] + [(1,90)]*3 + [(7, 90)],
				   [(7, 270)] + [(1,90)] * 3 + [(2, 0)] + [(0,0)] * 10 + [(10, 0)] + [(1,90)] * 3 + [(7, 90)],
				   [(7, 270)] + [(1,90)] * 3 + [(2, 0)] + [(0,0)] * 10 + [(10, 0)] + [(1,90)] * 3 + [(7, 90)],
				   [(7, 270)] + [(1,90)] * 3 + [(5, 270)] + [(2,270)] + [(1,90)] * 8 + [(10,90)] + [(5, 180)] + [(1,90)] * 3 + [(7, 90)],
				   [(7, 270)] + [(1,90)] * 3 + [(10, 270)] + [(2,90)] + [(1,90)] *  8 + [(10,270)] + [(2,90)] + [(1,90)] * 3 + [(7, 90)],
				   [(0,0)] * 20,
				   [(0,0)] * 20]

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

	# Rails where the direction is right
	right_rails = []
	for i in range(10):
		right_rails.append((5,i+5))
	# Rails where the direction is left
	left_rails = []
	for i in range(10):
		left_rails.append((6,i+5))

	down_rails = [(0,0)]
	up_rails = [(0,0)]
