from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d

# Station
class Station:

	# TODO metti valori precisi per min wait time (lis)
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
		self.calculate_rails(railway_topology)
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

		#Flag
		left = True  # Flag to understand where to go right or left
		north = True    # Flag to understand where to go right or left

		# Indicating the incrementing number north or sud (in case of horizontal station), east ovest (in case of vertical stations)
		difference_from_original = 0 

		# Counter
		counter_of_rails = 1
		# Rails of the station, has the position of the rails. Each row is a rail
		self.rails = []
  
		# Counter to check the station is well positioned, to avoid the while goes for eternity
		counter = 0

		# Starting position the center of station
		current_position = (self.position[0], self.position[1])
  
		# The first rail coincide with the position of the station
		self.rails.append(tuple(current_position))

		while counter_of_rails < num_of_rails:

			counter += 1

			if counter > 500:
				raise ImportError('The position of the station, or the capacity should be different, check for the right position or capacity, cant calculate the rails')

			# Horizontal rail
			# is position inside the grid?
			elif current_position[0] >= rail_shape[0] or current_position[0] < 0 or current_position[1] >= rail_shape[1] or current_position[1] < 0:
				continue 
			
			# Starting going up to check the rails
			elif railway_topology.grid[current_position] == 1025 and north:
				# Going to up
				new_pos = (-1, 0)
				# Update the position to left or right
				current_position = Vec2d.add(current_position, new_pos)
				# If not 0 the rail is in the station
				if railway_topology.grid[current_position] != 0:
					self.rails.append(tuple(current_position))
					counter_of_rails += 1
					continue
				else:
					current_position = (self.position[0], self.position[1])
					north = False
					continue			

			elif railway_topology.grid[current_position] == 1025 and not north:
				# Going to up
				new_pos = (1, 0)
				# Update the position to left or right
				current_position = Vec2d.add(current_position, new_pos)
				# If not 0 the rail is in the station
				if railway_topology.grid[current_position] != 0:
					self.rails.append(tuple(current_position))
					counter_of_rails += 1
					continue
				else:
					continue
 
			# Vertical rail same as for horizontal, but starting from south and north, and then left and right
			elif railway_topology.grid[current_position] == 32800 and left:
				# Going to up
				new_pos = (0, -1)
				# Update the position to left or right
				current_position = Vec2d.add(current_position, new_pos)
				# If not 0 the rail is in the station
				if railway_topology.grid[current_position] != 0:
					self.rails.append(tuple(current_position))
					counter_of_rails += 1
					continue
				else:
					current_position = (self.position[0], self.position[1])
					left = False
					continue			

			elif railway_topology.grid[current_position] == 32800 and not left:
				# Going to up
				new_pos = (0, 1)
				# Update the position to left or right
				current_position = Vec2d.add(current_position, new_pos)
				# If not 0 the rail is in the station
				if railway_topology.grid[current_position] != 0:
					self.rails.append(tuple(current_position))
					counter_of_rails += 1
					continue
				else:
					continue

		# Eliminate duplicates
		rails_position = []
		for i in range(len(self.rails)):
			if self.rails[i] not in rails_position:
				rails_position.append((self.rails[i]))
		self.rails = rails_position