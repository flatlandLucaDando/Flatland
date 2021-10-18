i = 2
# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)

# Example 1 simple railway with 2 stations and one rail
railway_example = [[(0,0)]*12,                                                      						 # 0
				   [(0,0)]*12,                                                      						 # 1
				   [(8,0)] + [(1,90)]*3 + [(8,90)] + [(0,0)]*2 + [(8,0)] + [(1,90)]*3 + [(8,90)],            # 2
				   [(8,270)] + [(1,90)]*3 + [(10,270)] + [(1,90)]*2 + [(2,90)] + [(1,90)]*3 + [(8,180)],     # 3
				   [(0,0)]*12,                                                              				 # 4
				   [(0,0)]*12]                                                       						 # 5

# wheight and height of the grid
height = len(railway_example)
width = len(railway_example[0])
if i == 1:
	# One rail, so no right or left rails  
	right_rails = [(0,0)]
	left_rails = [(0,0)]
	down_rails = [(0,0)]
	up_rails = [(0,0)]

if i ==2:
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

	av_line = [(0,0)]
