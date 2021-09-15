import numpy as np
from flatland.core.grid.grid4_astar import a_star
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap


example_num = 4 


# A convoy is a locomotive + wagons.
class Convoy:

	def __init__(self, identifier, train_type, schedule):
		# identifier of the train
		self.id = identifier
		# type of train (High velocity, Intercity, regional)
		self.train_type = train_type
		# schedule of the train
		self.schedule = schedule

	# Discover the starting run of a certain run
	def starting_time(self, run_number):
		return self.schedule[run_number][0]

	def schedule_verification(self, schedule, num_trains_run):
		if type(schedule) == list:
			row_num = len(schedule)
			column_num = len(schedule[0])
			if row_num >  1:
				for num_of_runs in range(num_trains_run - 1):
					for steps in range(len(schedule) - 1):
						if schedule[num_of_runs][step + 1] <= schedule[num_of_runs][step]:
							print('==========================================================')
							print('The time to connect stations',step,'and',step + 1,'have to be > 0')

			else:
				for steps in range(len(schedule) - 1):
					if schedule[num_of_runs][step + 1] <= schedule[num_of_runs][step]:
						print('==========================================================')
						print('The time to connect stations',step,'and',step + 1,'have to be > 0')
		else:
			print('A schedule should comprend different stations')
		return True

# Type of a convoy, can be high velocity or regional
class Type_of_convoy:

	def __init__(self, type_name, max_speed):
		# High velocity, Intercity, regional
		self.type_name = type_name
		# Maximum possible speed for the train based on its type
		self.max_speed = max_speed

# A physical connection between two stations
class Rail_connection:

	def __init__(self, identifier, station_a, station_b, rail_connection_type, additional_runtime_percent):
		# Each rail section have an identifier
		self.id = identifier
		# Station A and B are the two connected stations
		self.station_a = station_a
		self.station_b = station_b
		# A connection can be: High velocity or normal
		self.rail_connection_type = rail_connection_type
		# Additional Runtime Percent is the percent [0-1] of the min run time that is added to the min run time, if the train is on schedule.
		# In general, he actual run time is computed as min run time + max(0, (min run time*additionalRuntimePercent)-actual_delay).
		self.additional_runtime_percent = additional_runtime_percent

	def calculate_runtime(self, station_a, station_b, transition_map):
		time_to_run = 0
		return time_to_run

	def calculate_rails(self, station_a, station_b, transition_map):
		return True

# A connection can be: High velocity or normal
class Connection_type:

	def __init__(self, name, max_speed_possible):
		# High velocity or normal
		self.name = name
		# max_speed_possible is the velocity the train can go depending in the type of connection
		self.max_speed_possible = max_speed_possible

# Station
class Station:

	def __init__(self, name, position, capacity, min_wait_time, additional_wait_percent, importance):
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

# A train run is the run on a line for a train (e.g. Genova-Milano --- Milano-Genova)
# The train run consider each intermediate station the train has to pass between the two "principal" stations
class Train_run:

	def __init__(self, identifier, line_id, starting_time, from_depot: bool, to_depot: bool):
		# each train_run has an identifier
		self.identifier = identifier
		# Id of the line of the run
		self.line_id = line_id
		# Starting time of the run
		self.starting_time = starting_time
		# FromDepot indicates whether this run starts from a depot. Similar for ToDepot.
		self.from_depot = from_depot
		self.to_depot = to_depot

# A line is considered from a city to another, tipically joining several cities
class Line:

	def __init__(self, identifier, type_line, stations, stops, direction):
		# ID of the line
		self.identifier = identifier
		# type of line (High velocity or regional)
		self.type_line = type_line
		# Stations where the line pass from
		self.stations = stations
		# Stops are stations where the train have to stop
		# Is an array with 0 where not stop and 1 where train stops
		self.stops = stops
		# Direction of the line (e.g. MILANO - ROMA ---> direction 1, ROMA - MILANO ----> direction -1)
		self.direction = direction

		if type(stations) == int or type(stops) == int:
			print('The stations of a line should be more than one, and the dimension of the stops should be the same of the stations')
		else:
			if(len(stations)) != (len(stops)):
				print('Stations and Stops have to be the same lenght')


	def inversion_of_line(self):
		self.direction = self.direction * -1


def single_train_run_generator(convoy, starting_time_of_the_run, stations_to_stop, railway_topology):
	stations_to_stop_position = []
	for i in range(len(stations_to_stop)):
		stations_to_stop_position.append(stations_to_stop[i].position)
	schedule = []
	schedule.append(starting_time_of_the_run)
	for stations in range( len(stations_to_stop) - 1 ):
		departure_station_position = stations_to_stop_position[stations]
		arrival_station_position = stations_to_stop_position[stations + 1]
		# First thing check the distance between two stations
		result = a_star(railway_topology, departure_station_position, arrival_station_position)
		# Maximum velocity a train can achieve
		train_velocity = convoy.train_type.max_speed

		lenght_path = len(result)  # distance between stations

		# Array when I put at each step the time needed to make the path
		# The total time is the sum of the numbers
		time_array = []

		# Check the at each step which train i am and which line im in
		# Train should be in the middle of two line type
		for step in range(lenght_path):
			if (result[step]) in av_line:
				time_array.append(pow(train_velocity,-1))
			else:
				time_array.append(pow(min(train_velocity, 1/2), -1))
		time_needed = sum(time_array)
		# Adding to the time a 10% to face with problems in case it's neaded
		time_needed = time_needed + int(time_needed/10)

		# Adding the precedence time 
		if stations != (len(stations_to_stop) - 1):
			schedule.append(int(time_needed + schedule[stations] + stations_to_stop[stations].min_wait_time))
		
	return schedule


'''
###############################################################
######################   EXAMPLE 1  ###########################
###############################################################
'''

# Environment 1
# Schedule example 1

if example_num == 1:
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

	# Define the type of convoy
	regional_train = Type_of_convoy('Regional', 1/3)
	intercity_train = Type_of_convoy('Intercity', 1/2)
	high_velocity_train = Type_of_convoy('High velocity', 1)

	# Define the type of connection
	regional = Connection_type(0, 1/2)

	# No high velocity lines, so make a (0,0) position
	av_line = (0,0)

	# Define the stations
	station_a = Station('Quarto', (3, 2), 2, 2, 0, 1)
	station_b = Station('Quinto', (3, 9), 2, 2, 0, 1)

	# Define the rail connection beetween the two stations
	connection_a_b = Rail_connection(0, station_a, station_b, regional, 0)

	# Define the line
	line_a_b = Line(0, regional, (station_a, station_b), (1, 1), 1)

	stations = []
	stations.append([station_a.position, line_a_b.type_line.max_speed_possible])
	stations.append([station_b.position, line_a_b.type_line.max_speed_possible])

	# Define the train runs
	train_run_0 = Train_run(0, 0, 3, True, True)
	train_run_1 = Train_run(1, 0, 25, True, True)

	# Define the convoys
	convoy_0 = Convoy(0, intercity_train, 0)
	convoy_1 = Convoy(1, intercity_train, 1)

	schedule_0 = []

	schedule_1 = []

	# Calculate the schedule for the first convoy 
	# for the first run
	schedule_0.append(single_train_run_generator(convoy_0, train_run_0.starting_time, line_a_b.stations, rail))

	convoy_0.schedule = schedule_0

	# Calculate the schedule for the second convoy
	# for the first run (inverted)
	schedule_1.append(single_train_run_generator(convoy_1, train_run_1.starting_time, line_a_b.stations[::-1], rail))

	convoy_1.schedule = schedule_1

	# Timetable has the stations positions, the schedule times, and the velocity
	timetable_example = []

	timetable_example.append([(station_a.position, station_b.position),schedule_0[0],0.5])
	timetable_example.append([((station_b.position[0] - 1, station_b.position[1]), (station_a.position[0] - 1, station_a.position[1])),schedule_1[0],0.5])


	# Different velocities in different lines.
	speed_ration_map_lines = [1.       ,  # High velocity lines
							  1. / 2.  ]  # Regional lines


'''
###############################################################
######################   EXAMPLE 2  ###########################
###############################################################
'''

# Environment 1
# Schedule example 2

if example_num == 2:
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

	# Define the type of convoy
	regional_train = Type_of_convoy('Regional', 1/3)
	intercity_train = Type_of_convoy('Intercity', 1/2)
	high_velocity_train = Type_of_convoy('High velocity', 1)

	# Define the type of connection
	regional = Connection_type(0, 1/2)

	# No high velocity lines, so make a (0,0) position
	av_line = (0,0)

	# Define the stations
	station_a = Station('Quarto', (3, 2), 2, 2, 0, 1)
	station_b = Station('Quinto', (3, 9), 2, 2, 0, 1)

	# Define the rail connection beetween the two stations
	connection_a_b = Rail_connection(0, station_a, station_b, regional, 0)

	# Define the line
	line_a_b = Line(0, regional, (station_a, station_b), (1, 1), 1)

	stations = []
	stations.append([station_a.position, line_a_b.type_line.max_speed_possible])
	stations.append([station_b.position, line_a_b.type_line.max_speed_possible])

	# Define the train runs
	train_run_0 = Train_run(0, 0, 3, True, True)
	train_run_1 = Train_run(1, 0, 44, True, True)
	train_run_2 = Train_run(2, 0, 85, True, True)
	train_run_3 = Train_run(3, 0, 25, True, True)
	train_run_4 = Train_run(4, 0, 65, True, True)
	train_run_5 = Train_run(5, 0, 110, True, True)

	# Define the convoys
	convoy_0 = Convoy(0, intercity_train, 0)
	convoy_1 = Convoy(1, intercity_train, 1)

	schedule_0 = []

	schedule_1 = []

	# Calculate the schedule for the first convoy 
	# for the first run
	schedule_0.append(single_train_run_generator(convoy_0, train_run_0.starting_time, line_a_b.stations, rail))
	# for the second run (inverted)
	schedule_0.append(single_train_run_generator(convoy_0, train_run_1.starting_time, line_a_b.stations[::-1], rail))
	# for the third run
	schedule_0.append(single_train_run_generator(convoy_0, train_run_2.starting_time, line_a_b.stations, rail))

	convoy_0.schedule = schedule_0

	# Calculate the schedule for the second convoy
	# for the first run (inverted)
	schedule_1.append(single_train_run_generator(convoy_1, train_run_3.starting_time, line_a_b.stations[::-1], rail))
	# for the second run
	schedule_1.append(single_train_run_generator(convoy_1, train_run_4.starting_time, line_a_b.stations, rail))
	# for the third run
	schedule_1.append(single_train_run_generator(convoy_1, train_run_5.starting_time, line_a_b.stations[::-1], rail))

	convoy_1.schedule = schedule_1

	# Timetable has the stations positions, the schedule times, and the velocity
	timetable_example = []

	timetable_example.append([(station_a.position, station_b.position),schedule_0[0],0.5])
	timetable_example.append([((station_b.position[0] - 1, station_b.position[1]), (station_a.position[0] - 1, station_a.position[1])),schedule_0[1],0.5])
	timetable_example.append([(station_a.position, station_b.position),schedule_0[2],0.5])
	timetable_example.append([((station_b.position[0] - 1, station_b.position[1]), (station_a.position[0] - 1, station_a.position[1])),schedule_1[0],0.5])
	timetable_example.append([(station_a.position, station_b.position),schedule_1[1],0.5])
	timetable_example.append([((station_b.position[0] - 1, station_b.position[1]), (station_a.position[0] - 1, station_a.position[1])),schedule_1[2],0.5])


	# Different velocities in different lines.
	speed_ration_map_lines = [1.       ,  # High velocity lines
							  1. / 2.  ]  # Regional lines



'''
###############################################################
######################   EXAMPLE 3  ###########################
###############################################################
'''

# Environment 1
# Schedule example 3

if example_num == 3:
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

	# Define the type of convoy
	regional_train = Type_of_convoy('Regional', 1/3)
	intercity_train = Type_of_convoy('Intercity', 1/2)
	high_velocity_train = Type_of_convoy('High velocity', 1)

	# Define the type of connection
	regional = Connection_type(0, 1/2)

	# No high velocity lines, so make a (0,0) position
	av_line = (0,0)

	# Define the stations
	station_a = Station('Quarto', (3, 2), 2, 2, 0, 1)
	station_b = Station('Quinto', (3, 9), 2, 2, 0, 1)

	# Define the rail connection beetween the two stations
	connection_a_b = Rail_connection(0, station_a, station_b, regional, 0)

	# Define the line
	line_a_b = Line(0, regional, (station_a, station_b), (1, 1), 1)

	stations = []
	stations.append([station_a.position, line_a_b.type_line.max_speed_possible])
	stations.append([station_b.position, line_a_b.type_line.max_speed_possible])

	# Define the train runs
	# Runs for convoy 1
	starting_time_0 = 3
	train_0_runs = []

	for i in range(14):  
		train_0_runs.append(Train_run(i,0,starting_time_0, True, True))
		starting_time_0 +=  41

	# Runs fo convoy 2
	starting_time_1 = 25
	train_1_runs = []

	for i in range(14):
		train_1_runs.append(Train_run(i,0,starting_time_1, True, True))
		starting_time_1 +=  41

 
	# Define the convoys
	convoy_0 = Convoy(0, intercity_train, 0)
	convoy_1 = Convoy(1, intercity_train, 1)

	schedule_0 = []

	schedule_1 = []

	# Calculate the schedule for the first convoy 
	for i in range(14):
		# One time is the right order
		if (i%2) == 0:
			schedule_0.append(single_train_run_generator(convoy_0, train_0_runs[i].starting_time, line_a_b.stations, rail))
		# One time reversed
		else:
			schedule_0.append(single_train_run_generator(convoy_0, train_0_runs[i].starting_time, line_a_b.stations[::-1], rail))

	convoy_0.schedule = schedule_0

	# Calculate the schedule for the second convoy
	for i in range(14):
		# One time reversed
		if (i%2) == 0:
			schedule_1.append(single_train_run_generator(convoy_1, train_1_runs[i].starting_time, line_a_b.stations[::-1], rail))
		# One time reversed
		else:
			schedule_1.append(single_train_run_generator(convoy_1, train_1_runs[i].starting_time, line_a_b.stations, rail))

	convoy_1.schedule = schedule_1

	# Timetable has the stations positions, the schedule times, and the velocity
	timetable_example = []

	# Convoy 1
	for i in range(14):
		if (i%2) == 0:
			timetable_example.append([(station_a.position, station_b.position), schedule_0[i],0.5])
		else:
			timetable_example.append([((station_b.position[0] - 1, station_b.position[1]), (station_a.position[0] - 1, station_a.position[1])), schedule_0[i],0.5])
	# Convoy 2
	for i in range(14):
		if (i%2) == 0:
			timetable_example.append([((station_b.position[0] - 1, station_b.position[1]), (station_a.position[0] - 1, station_a.position[1])),schedule_1[i],0.5])
		else:
			timetable_example.append([(station_a.position, station_b.position),schedule_1[i],0.5])


	# Different velocities in different lines.
	speed_ration_map_lines = [1.       ,  # High velocity lines
							  1. / 2.  ]  # Regional lines



'''
################################################################
######################   EXAMPLE 4  ############################
################################################################
'''

if example_num == 4:
	# Example generate a rail given a manual specification,
	# a map of tuples (cell_type, rotation)

	# Example 2 simple railway with 2 stations and two rails and multiple rails in the stations
	railway_example = [[(0,0)] * 20,
					   [(0,0)] * 20,
					   [(8,0)] + [(1,90)] * 3 + [(8,90)] + [(0,0)] * 10 + [(8,0)] + [(1,90)]*3 + [(8,90)],
					   [(10, 0)] + [(1,90)] * 3 + [(2, 0)] + [(0,0)] * 10 + [(10, 0)] + [(1,90)] * 3 + [(2, 0)],
					   [(10, 0)] + [(1,90)] * 3 + [(2, 0)] + [(0,0)] * 10 + [(10, 0)] + [(1,90)] * 3 + [(2, 0)],
					   [(10, 0)] + [(1,90)] * 3 + [(5, 270)] + [(1,90)] * 10 + [(5, 180)] + [(1,90)] * 3 + [(2, 0)],
					   [(8, 270)] + [(1,90)] * 3 + [(10, 270)] + [(1,90)] *  10 + [(2,90)] + [(1,90)] * 3 + [(8, 180)],
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


	# Define the type of convoy
	regional_train = Type_of_convoy('Regional', 1/3)
	intercity_train = Type_of_convoy('Intercity', 1/2)
	high_velocity_train = Type_of_convoy('High velocity', 1)

	# Define the type of connection
	regional = Connection_type(0, 1/2)

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


	# Define the stations
	station_a = Station('Quarto', (6, 2), 2, 2, 0, 1)
	station_b = Station('Quinto', (6, 17), 2, 2, 0, 1)

	# Define the rail connection beetween the two stations
	connection_a_b = Rail_connection(0, station_a, station_b, regional, 0)

	# Define the line
	line_a_b = Line(0, regional, (station_a, station_b), (1, 1), 1)

	stations = []
	stations.append([station_a.position, line_a_b.type_line.max_speed_possible])
	stations.append([station_b.position, line_a_b.type_line.max_speed_possible])

	# Station positions are defined by the position x,y on the grid
	station_a_bin_0 = (2, 2)
	station_a_bin_1 = (3, 2)
	station_a_bin_2 = (4, 2)
	station_a_bin_3 = (5, 2)
	station_a_bin_4 = (6, 2)
	station_b_bin_0 = (2, 17)
	station_b_bin_1 = (3, 17)
	station_b_bin_2 = (4, 17)
	station_b_bin_3 = (5, 17)
	station_b_bin_4 = (6, 17)


	# An array more compact to store the different stations, for each station the array conteins the position (x,y) of the station, the maximum
	# velocity supported by the line (high velocity, regional...)
	stations = [[station_a_bin_4, 0.5], [station_b_bin_4, 0.5]]	

	# Timetable conteins the station where the train should pass, from starting station to aim, and conteins the time at which
	# each train has to pass in the station, the last number represent the velocity of train (high velocity, intercity or regional)
	# Each row represent a different train
	timetable_example = [[(station_a_bin_1, station_b_bin_0), (5 ,51), (1)],   		 	# agent 0   high velocity
						 [(station_b_bin_1, station_a_bin_0), (21, 71), (1 / 2)],
						 [(station_a_bin_0, station_b_bin_1), (41, 87), (1 / 2)],
						 [(station_b_bin_1, station_a_bin_2), (50, 91), (1)]]      


	# Different velocities in different lines.
	speed_ration_map_lines = [1.       ,  # High velocity lines
							  1. / 2.  ]  # Regional lines













''' OLD
################################################################
######################   EXAMPLE 2  ############################
################################################################
'''

if example_num == 9:
	# Example generate a rail given a manual specification,
	# a map of tuples (cell_type, rotation)

	# Example 2 simple railway with 2 stations and two rails and multiple rails in the stations
	railway_example = [[(0,0)] * 20,
					   [(0,0)] * 20,
					   [(8,0)] + [(1,90)] * 3 + [(8,90)] + [(0,0)] * 10 + [(8,0)] + [(1,90)]*3 + [(8,90)],
					   [(10, 0)] + [(1,90)] * 3 + [(2, 0)] + [(0,0)] * 10 + [(10, 0)] + [(1,90)] * 3 + [(2, 0)],
					   [(10, 0)] + [(1,90)] * 3 + [(2, 0)] + [(0,0)] * 10 + [(10, 0)] + [(1,90)] * 3 + [(2, 0)],
					   [(10, 0)] + [(1,90)] * 3 + [(5, 270)] + [(1,90)] * 10 + [(5, 180)] + [(1,90)] * 3 + [(2, 0)],
					   [(8, 270)] + [(1,90)] * 3 + [(10, 270)] + [(1,90)] *  10 + [(2,90)] + [(1,90)] * 3 + [(8, 180)],
					   [(0,0)] * 20,
					   [(0,0)] * 20]
	
	# wheight and height of the grid
	height = len(railway_example)
	width = len(railway_example[0])

	# Lines of trains, default is normal line, high velocity is different line
	# Here are specified the table cells where the line is av
	av_line = [(0,0)]

	# Station positions are defined by the position x,y on the grid
	station_a_bin_0 = (2, 2)
	station_a_bin_1 = (3, 2)
	station_a_bin_2 = (4, 2)
	station_a_bin_3 = (5, 2)
	station_a_bin_4 = (6, 2)
	station_b_bin_0 = (2, 17)
	station_b_bin_1 = (3, 17)
	station_b_bin_2 = (4, 17)
	station_b_bin_3 = (5, 17)
	station_b_bin_4 = (6, 17)

	num_bin_a = 5
	num_bin_b = 5

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

	# An array more compact to store the different stations, for each station the array conteins the position (x,y) of the station, the maximum
	# velocity supported by the line (high velocity, regional...)
	stations = [[station_a_bin_4, 0.5], [station_b_bin_4, 0.5]]	

	# Timetable conteins the station where the train should pass, from starting station to aim, and conteins the time at which
	# each train has to pass in the station, the last number represent the velocity of train (high velocity, intercity or regional)
	# Each row represent a different train
	timetable_example = [[(station_a_bin_1, station_b_bin_0), (5 ,51), (1)],   		 	# agent 0   high velocity
						 [(station_b_bin_1, station_a_bin_0), (21, 71), (1 / 2)],
						 [(station_a_bin_0, station_b_bin_1), (41, 87), (1 / 2)],
						 [(station_b_bin_1, station_a_bin_2), (50, 91), (1)]]      


	# Different velocities in different lines.
	speed_ration_map_lines = [1.       ,  # High velocity lines
							  1. / 2.  ]  # Regional lines







''' OLD
################################################################
######################   EXAMPLE 3  ############################
################################################################
'''

if example_num == 99:
	# Example generate a rail given a manual specification,
	# a map of tuples (cell_type, rotation)

	# Example 3 simple railway with 3 stations and two rails and multiple rails in the stations
	railway_example = [[(0,0)] * 35,
					   [(0,0)] * 35,
					   [(8,0)] + [(1,90)] * 3 + [(8,90)] + [(0,0)] * 10 + [(8,0)] + [(1,90)]*3 + [(8,90)] + [(0,0)] * 10 + [(8,0)] + [(1,90)]*3 + [(8,90)],
					   [(10, 0)] + [(1,90)] * 3 + [(2, 0)] + [(0,0)] * 10 + [(10, 0)] + [(1,90)] * 3 + [(2, 0)] + [(0,0)] * 10 + [(10, 0)] + [(1,90)] * 3 + [(2, 0)],
					   [(10, 0)] + [(1,90)] * 3 + [(2, 0)] + [(0,0)] * 10 + [(10, 0)] + [(1,90)] * 3 + [(2, 0)] + [(0,0)] * 10 + [(10, 0)] + [(1,90)] * 3 + [(2, 0)],
					   [(10, 0)] + [(1,90)] * 3 + [(5, 270)] + [(1,90)] * 10 + [(5, 180)] + [(1,90)] * 3 + [(5, 90)] + [(1,90)] * 10 + [(5, 180)] + [(1,90)] * 3 + [(2, 0)],
					   [(8, 270)] + [(1,90)] * 3 + [(10, 270)] + [(1,90)] *  10 + [(2,90)] + [(1,90)] * 3 + [(10, 270)] + [(1,90)] * 10 + [(2,90)] + [(1,90)] * 3 + [(8, 180)],
					   [(0,0)] * 35,
					   [(0,0)] * 35,
					   [(0,0)] * 35,
					   [(0,0)] * 35,
					   [(0,0)] * 35,
					   [(0,0)] * 35,
					   [(0,0)] * 35,
					   [(0,0)] * 35,
					   [(0,0)] * 35,
					   [(0,0)] * 35,
					   [(0,0)] * 35]
	
	# wheight and height of the grid
	height = len(railway_example)
	width = len(railway_example[0])

	# Lines of trains, default is normal line, high velocity is different line
	# Here are specified the table cells where the line is av
	av_line = [(0,0)]

	# Station positions are defined by the position x,y on the grid
	station_a_bin_0 = (2, 2)
	station_a_bin_1 = (3, 2)
	station_a_bin_2 = (4, 2)
	station_a_bin_3 = (5, 2)
	station_a_bin_4 = (6, 2)
	station_b_bin_0 = (2, 17)
	station_b_bin_1 = (3, 17)
	station_b_bin_2 = (4, 17)
	station_b_bin_3 = (5, 17)
	station_b_bin_4 = (6, 17)
	station_c_bin_0 = (2, 32)
	station_c_bin_1 = (3, 32)
	station_c_bin_2 = (4, 32)
	station_c_bin_3 = (5, 32)
	station_c_bin_4 = (6, 32)


	num_bin_a = 5
	num_bin_b = 5

	# Rails where the direction is right
	right_rails = []
	for i in range(10):
		right_rails.append((5,i+5))
		right_rails.append((5,i+20))
	# Rails where the direction is left
	left_rails = []
	for i in range(10):
		left_rails.append((6,i+5))
		left_rails.append((6,i+20))

	down_rails = [(0,0)]
	up_rails = [(0,0)]

	# An array more compact to store the different stations, for each station the array conteins the position (x,y) of the station, the maximum
	# velocity supported by the line (high velocity, regional...)
	stations = [[station_a_bin_4, 0.5], [station_b_bin_4, 0.5]]	

	# Timetable conteins the station where the train should pass, from starting station to aim, and conteins the time at which
	# each train has to pass in the station, the last number represent the velocity of train (high velocity, intercity or regional)
	# Each row represent a different train
	timetable_example = [[(station_a_bin_1, station_b_bin_0), (5 ,51), (1)],   		 	# agent 0   high velocity
						 [(station_b_bin_1, station_a_bin_0), (21, 71), (1 / 2)],
						 [(station_a_bin_0, station_b_bin_1), (41, 87), (1 / 2)],
						 [(station_b_bin_1, station_a_bin_2), (50, 91), (1)],
						 [(station_a_bin_3, station_b_bin_4, station_c_bin_4), (10, 55, 56), (1)],
						 [(station_c_bin_0, station_a_bin_2), (30, 31), (1/2)],
						 [(station_b_bin_2, station_c_bin_2), (50, 51), (1)]]      


	# Different velocities in different lines.
	speed_ration_map_lines = [1.       ,  # High velocity lines
							  1. / 2.  ]  # Regional lines





''' OLD
###############################################################
#####################   EXAMPLE 4  ############################
###############################################################
'''

if example_num == 999:
	
	# Example generate a rail given a manual specification,
	# a map of tuples (cell_type, rotation)

	# Example 4 compless railway with 5 stations, two rails everywhere, switches only in big deviations
	railway_example = [[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],                          #1
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (8, 0), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (8, 90), (0, 0), (0, 0), (0, 0)],    #2
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (8, 0), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (8, 90), (1, 0), (0, 0), (0, 0), (0, 0)],      #3
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                          #4
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                          #5   
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                          #6
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),  (0,0),    (0,0),   (0,0),    (0,0),   (0,0),   (0,0),   (0,0),   (0,0),    (0,0),    (0,0),   (0,0),   (0,0),   (0,0),   (0,0),   (0,0),   (0,0),   (0,0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                          #7
						[(0, 0) , (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (8, 0), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90),(1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (2, 270), (10, 90), (1, 90), (1, 90), (1, 90), (1, 90),(8, 180), (1, 0),(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0),(0, 0)],                   #8
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0), (8, 0),  (1, 90), (1, 90), (1, 90), (1, 90),  (1, 90),(1, 90), (1, 90), (1, 90),(1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (5, 90), (5, 0), (1, 90),(1, 90), (1, 90), (1, 90), (1, 90),(8, 180), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0),(0, 0)],                #9
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0), (1, 0), (0, 0), (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0),  (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                        #10
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0), (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0),  (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],                       #11
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0), (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0),  (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],  #12
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0), (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],   #13
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0), (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],   #14
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0), (0, 0), (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],    #15
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],  #16
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),  (1, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],  #17
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0),  (1, 0),(1, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],  #18
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (1, 0), (1, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)], #19
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (1, 0),  (1, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0)],    #20
						[(8, 0),  (1, 90), (1, 90),(1, 90),(1, 90),(1, 90),(1, 90),(5, 0),(5, 90),(1, 90), (1, 90), (1, 90),  (1, 90),  (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (5, 0),(5, 90), (1, 90), (1, 90), (1, 90), (1, 90),(1, 90),(1, 90), (1, 90),(1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (8, 180), (1, 0), (0, 0), (0, 0),(0, 0)],    #21
						[(8, 270),(1, 90),(1, 90),(1, 90),(1, 90),(1, 90),(1, 90),(10, 270), (2, 90), (1, 90), (1, 90), (1, 90),  (1, 90),  (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (10, 270), (2, 90), (1, 90),(1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90),(1, 90), (1, 90), (1, 90), (1, 90),  (1, 90), (1, 90),(1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (8, 180), (0, 0), (0, 0),(0, 0)],    #22
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],   #23
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],   #24
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],   #25
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)],   #26
						[(0, 0),  (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0),  (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0),   (0, 0), (0, 0),  (0, 0),  (0, 0),   (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0),   (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]]   #27

	# width and height of the grid
	height = len(railway_example)
	width = len(railway_example[0])

	# Lines of trains, default is normal line, high velocity is different line
	# Here are specified the table cells where the line is av
	av_line = [(21, 1), (21, 2), (21, 3), (21, 4), (21, 5), (21, 6), (21, 7), (21, 8), (21, 9), (21, 10), (21, 11), (21, 12), (21, 13), (21, 14), (21, 15), (21, 16), 
	(21, 17), (21, 18), (21, 19), (21, 20), (21, 21), (21, 22), (21, 23), (21, 24), (21, 25), (21, 26), (21, 27), (21, 28), (21, 29), (21, 30), (21, 31), (21, 32), 
	(21, 33), (21, 34), (21, 35), (21, 36), (21, 37), (21, 38), (21, 39), (21, 40), (21, 41), (21, 42), (21, 43), (21, 44), (21, 45), (21, 46), (21, 47), (21, 48), 
	(21, 49), (20, 49), (19, 49), (18, 49), (17, 49), (16, 49), (15, 49), (20, 2), (20, 3), (20, 4), (20, 5), (20, 6), (20, 7), (20, 8), (20, 9), (20, 10), (20, 11), 
	(20, 12), (20, 13), (20, 14), (20, 15), (20, 16), (20, 17), (20, 18), (20, 19), (20, 20), (20, 21), (20, 22), (20, 23), (20, 24), (20, 25), (20, 26), (20, 27), 
	(20, 28), (20, 29), (20, 30), (20, 31), (20, 32), (20, 33), (20, 34), (20, 35), (20, 36), (20, 37), (20, 38), (20, 39), (20, 40), (20, 41), (20, 42), (20, 43), 
	(20, 44), (20, 45), (20, 46), (20, 47), (20, 48), (19, 48), (18, 48), (17, 48), (16, 48), (15, 48)]

	# Rails where direction of trains is right
	right_rails = []
	for i in range(47):
		right_rails.append((20, i + 2))
	for i in range(22):
		right_rails.append((7, i + 8))
	for i in range(19):
		right_rails.append((1, i + 30))

	# Rails where direction of trains is left
	left_rails = []
	for i in range(48):
		left_rails.append((21, i + 2))
	for i in range(23):
		left_rails.append((8, i + 9))
	for i in range(17):
		left_rails.append((2, i + 31))

	# TODO sistema i binari su e giù
	down_rails = [(0,0)]
	up_rails = [(0,0)]

	# Station positions are defined by the position x,y on the grid
	station_a = (21, 1)
	station_b = (21, 37)
	station_c = (8, 14)
	station_d = (1, 36)
	station_e = (15, 49)

	# An array more compact to store the different stations, for each station the array conteins the position (x,y) of the station, the maximum
	# velocity supported by the line (high velocity, regional...)
	stations = [[station_a, 1.], [station_b, 1.], [station_c, 0.5], [station_d, 0.5], [station_e, 1.]]

	# Timetable conteins the station where the train should pass, from starting station to aim, and conteins the time at which
	# each train has to pass in the station, the last number represent the velocity of train (high velocity, intercity or regional)
	# Each row represent a different train

	timetable_example = [[((20,1), (20, 37), (15, 48)), (0 ,41, 59), (1)],   		 # agent 0   high velocity
				 [(station_b, station_a), (10, 91), (1 / 2)],                     			 # agent 1   Intercity
				 [((7, 14), (20, 37)), (5, 72), (1)],                  	             # agent 2   High velocity
				 [(station_d, station_e), (15, 138), (1 / 4)],                             # agent 3   Regional
				 [((15, 48), (2, 36)), (0, 114), (1 / 4)],					         # agent 4   Regional
				 [((2, 36), station_c), (10, 73), (1 / 2)],
				 [((8,14), station_a), (12, 76), (1)]]                              # agent 5   Intercity


	# Different velocities in different lines.
	speed_ration_map_lines = [1.       ,  # High velocity lines
							  1. / 2.  ]  # Regional lines


