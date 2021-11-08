from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d

# Station
class Station:

	def __init__(self, name, position, capacity, min_wait_time, additional_wait_percent, importance, railway_topology):
		# Name of the station (e.g. Milano, Torino etc etc)
		self.name = name
		# Position (y,x) of the station in the railway
		self.position = position
		# Capacity of the station, num of rails in the station
		self.capacity = capacity
		# Minimum wait time for the trains, a train can't stop less than the min wait time
		self.min_wait_time = min_wait_time
		# Additional wait percent is the percent [0-1] of the minWaitTime that is added to the minWait at each stop, if the train is on schedule. 
		# In general, the actual (runtime) stopping time is computed as minWaitTime + max(0, (minWaitTime*additionalWaitTimePercent)-actual_delay).
		self.additional_wait_percent = additional_wait_percent
		# Stations have different importance depending on how much they are big and how much people they transport depending on the time
		self.importance = importance
		# Rails of the station
		if capacity == 1:
			self.rails = None
		else:
			self.rails = self.calculate_rails(railway_topology)
		return


	def time_in_station(self, train_velocity):
		# The len of the rails is given by the station
		len_rails = len(self.rails_in_station[0])
		# The time needed is given by the formula (len * 1/velocity + waiting time + 10% of time)
		time_needed =  len_rails * int(pow(train_velocity, -1)) + self.min_wait_time[0]
		time_needed += int(time_needed/10)
		return time_needed

	def calculate_rails(self, railway_topology):
		# Number of rails of the station
		num_of_rails = self.capacity
		center_of_station = self.position 
		rail_shape = railway_topology.grid.shape

		print(num_of_rails)
		print(center_of_station)
		print(rail_shape)

		#Flag
		right = False  # Flag to understand where to go right or left
		north = True    # Flag to understand where to go right or left

		# Indicating the incrementing number north or sud (in case of horizontal station), east ovest (in case of vertical stations)
		difference_from_original = 0 

		# Counter
		counter_of_rails = 0
		# Rails of the station, has the position of the rails. Each row is a rail
		self.rails = []
		# Contein the single rail positions
		single_rail_in_station = []

		# Counter to check the station is well positioned, to avoid the while goes for eternity
		counter = 0

		# Starting position the center of station
		current_position = (self.position[0], self.position[1])

		while counter_of_rails < num_of_rails:

			counter += 1

			if counter > 500:
				raise ImportError('The position of the station, or the capacity should be different, check for the right position or capacity, cant calculate the rails')

			# Horizontal rail
			if railway_topology.grid[current_position] == 1025:
				# The starting rail is a rail of the station
				single_rail_in_station.append(current_position)
				# Going to left
				if not right:
					new_pos = (0,-1)
				# Going to right
				if right:
					new_pos = (0, 1)
				# Update the position to left or right
				current_position = Vec2d.add(current_position, new_pos)

				# is position inside the grid?
				if current_position[0] >= rail_shape[0] or current_position[0] < 0 or current_position[1] >= rail_shape[1] or current_position[1] < 0:
					continue 
				# Current position != 0 and going left?
				# Starting going to left
				if railway_topology.grid[current_position] != 0 and not right:
					# If not 0 the rail is in the station
					single_rail_in_station.append(current_position)
					# I'm arrived at the end of the rail in station
					if railway_topology.grid[current_position] != 1025:
						# The entrance is on left
						if current_position[0] == self.position[0]:
							self.station_entrance = current_position
						# Restart the position to the original once
						# If I'm north or sud the new original position is north or sud with respect to the real original
						if difference_from_original == 0:
							current_position = (self.position[0], self.position[1])
						else:
							current_position = (self.position[0] + difference_from_original, self.position[1])
						# Then going to right
						right = True
						continue
					continue

				# Current position != 0 and going right?	            		
				if railway_topology.grid[current_position] != 0 and right:
					# If not 0 the rail is in the station
					single_rail_in_station.append(current_position)
					# I'm arrived at the end of the rail in station
					if railway_topology.grid[current_position] != 1025:
						# The exit is on right
						if current_position[0] == self.position[0]:
							self.station_exit = current_position
						self.rails.append(single_rail_in_station)
						# Restart the position from original once
						current_position = (self.position[0], self.position[1])
						# A rail is ended
						counter_of_rails += 1
						# Resetting the flag
						right = False
						if north:
							# Going to north, checking if at north there is a rail or not 
							new_pos = (-1, 0)
							position = Vec2d.add(current_position, new_pos)
							if railway_topology.grid[position] != 0:
								# Checking if there is a rail
								current_position = Vec2d.add(current_position, new_pos)
								# Resetting the single rail
								single_rail_in_station = []
								# I'm one step north, so I'm one row up in the matrix
								difference_from_original -= 1
								continue
							else:
								# No rails upper, so I have to go down
								north = False
								# Restart the position from original once
								current_position = (self.position[0], self.position[1])
								# I'm at the original position, so difference is 0
								difference_from_original = 0
								continue
						else:
							# Going south
							new_pos = (+1, 0)
							position = Vec2d.add(current_position, new_pos)
							# Check if south I have a rail or not
							if railway_topology.grid[position] != 0:
								current_position = Vec2d.add(current_position, new_pos)
								# Resetting the single rail
								single_rail_in_station = []
								# I'm one step south, so I'm one row down in the matrix
								difference_from_original += 1
								continue
							else:
								# No rails down, so I have to go up
								north = True
								# Restart the position from original once
								current_position = (self.position[0], self.position[1])
								# I'm at the original position, so difference is 0
								difference_from_original = 0
								continue

			# Vertical rail same as for horizontal, but starting from south and north, and then left and right
			elif railway_topology.grid[current_position] == 32800:
				# The starting rail is a rail of the station
				single_rail_in_station.append(current_position)
				# Going down
				if not north:
					new_pos = (+1,0)
				# Going to right
				if north:
					new_pos = (-1, 0)
				# Update the position to left or right
				current_position = Vec2d.add(current_position, new_pos)

				# is position inside the grid?
				if current_position[0] >= rail_shape[0] or current_position[0] < 0 or current_position[1] >= rail_shape[1] or current_position[1] < 0:
					continue 
				# Current position != 0 and going left?
				# Starting going to left
				if railway_topology.grid[current_position] != 0 and not north:
					# If not 0 the rail is in the station
					single_rail_in_station.append(current_position)
					# I'm arrived at the end of the rail in station
					if railway_topology.grid[current_position] != 32800:
						# The entrance is on left
						if current_position[1] == self.position[1]:
							self.station_entrance = current_position
						# Restart the position to the original once
						# If I'm north or sud the new original position is north or sud with respect to the real original
						if difference_from_original == 0:
							current_position = (self.position[0], self.position[1])
						else:
							current_position = (self.position[0], self.position[1] + difference_from_original)
						# Then going to right
						north = True
						continue
					continue

				# Current position != 0 and going right?	            		
				if railway_topology.grid[current_position] != 0 and north:
					# If not 0 the rail is in the station
					single_rail_in_station.append(current_position)
					# I'm arrived at the end of the rail in station
					if railway_topology.grid[current_position] != 32800:
						# The exit is on right
						if current_position[1] == self.position[1]:
							self.station_exit = current_position
						self.rails.append(single_rail_in_station)
						# Restart the position from original once
						current_position = (self.position[0], self.position[1])
						# A rail is ended
						counter_of_rails += 1
						# Resetting the flag
						north = False
						if not right:
							# Going to left, checking if at north there is a rail or not 
							new_pos = (0, -1)
							position = Vec2d.add(current_position, new_pos)
							if railway_topology.grid[position] != 0:
								# Checking if there is a rail
								current_position = Vec2d.add(current_position, new_pos)
								# Resetting the single rail
								single_rail_in_station = []
								# I'm one step north, so I'm one row up in the matrix
								difference_from_original -= 1
								continue
							else:
								# No rails upper, so I have to go down
								right = True
								# Restart the position from original once
								current_position = (self.position[0], self.position[1])
								# I'm at the original position, so difference is 0
								difference_from_original = 0
								continue
						else:
							# Going south
							new_pos = (0, +1)
							position = Vec2d.add(current_position, new_pos)
							# Check if south I have a rail or not
							if railway_topology.grid[position] != 0:
								current_position = Vec2d.add(current_position, new_pos)
								# Resetting the single rail
								single_rail_in_station = []
								# I'm one step south, so I'm one row down in the matrix
								difference_from_original += 1
								continue
							else:
								# No rails down, so I have to go up
								right = False
								# Restart the position from original once
								current_position = (self.position[0], self.position[1])
								# I'm at the original position, so difference is 0
								difference_from_original = 0
								continue

		# Eliminate duplicates
		rails_position = []
		rails_position_single = []
		for i in range(len(self.rails)):
			for j in range(len(self.rails[i])):
				if self.rails[i][j] not in rails_position_single:
					rails_position_single.append(self.rails[i][j])
			rails_position.append(rails_position_single)
			rails_position_single = []
		self.rails = rails_position
