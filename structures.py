import numpy as np
from flatland.core.grid.grid4_astar import a_star
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from enum import Enum
import graphviz


example_num = 2

# TODO calculate the av_rails, in order to distinguish them
# TODO calculate the right,left,up,down rails, in order to distinguish them
# TODO understad if the velocities are realistic or not


# Type of a convoy, can be high velocity or regional
# The velocity are given by default depending on the type of convoy, 360 for HV, 180 for IC, 120 for Regional
class Type_of_convoy(Enum):
	HIGH_VELOCITY = 1
	INTERCITY = 2
	REGIONAL = 3

# A connection can be: High velocity or normal
# The velocity are given by default depending on the type of connection, 360 for HV and 120 for normal
class Connection_type(Enum):
	HIGH_VELOCITY_RAIL = 1
	NORMAL_RAIL = 2

# A convoy is a locomotive + wagons.
class Convoy:

	def __init__(self, identifier, train_type, schedule = [], maximum_velocity: int = 1/2):

		# identifier of the train
		self.id = identifier
		# type of train (High velocity, Intercity, regional)
		self.train_type = train_type
		# Maximum velocity possible for the train
		self.maximum_velocity = maximum_velocity
		# schedule of the train
		self.schedule = []

		if train_type == Type_of_convoy.HIGH_VELOCITY:
			maximum_velocity = 1
		if train_type == Type_of_convoy.INTERCITY:
			maximum_velocity = 1/2
		if train_type == Type_of_convoy.REGIONAL:
			maximum_velocity = 1/3

	def add_train_run(self, train_run):
		self.schedule.append(train_run)

	def calculate_schedule(self, railway_topology):
		# The timetable that should be returned
		timetable = []
		# For each train run defined
		for num_of_runs in range(len(self.schedule)):
			# The single train run
			single_train_run = []
			# The number of station to pass
			num_of_stations = len(self.schedule[num_of_runs].line_id.stations)
			# The station to stop
			stations_to_stop_position = []
			# Direction not inverted?
			if not self.schedule[num_of_runs].inverse_train_direction:
				for i in range(num_of_stations):
					# append the station position in the right order
					stations_to_stop_position.append(self.schedule[num_of_runs].line_id.stations[i].position)
			# Direction inverdet?
			else:
				for i in range(num_of_stations):
					# append the station position in inverted order
					stations_to_stop_position.append(self.schedule[num_of_runs].line_id.stations[num_of_stations - 1 - i].position)
			# Adding the starting time
			single_train_run.append(self.schedule[num_of_runs].starting_time)

			for stations in range(num_of_stations -1):
				departure_station_position = stations_to_stop_position[stations]
				arrival_station_position = stations_to_stop_position[stations + 1]
				# First thing check the distance between two stations
				result = a_star(railway_topology, departure_station_position, arrival_station_position)
				# Maximum velocity a train can achieve
				train_velocity = self.maximum_velocity 

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
				if stations != (num_of_stations - 1):
					# sum of time needed, the precedence time and the waiting time at the station
					single_train_run.append(int(time_needed + single_train_run[stations] + self.schedule[num_of_runs].line_id.stations[stations].min_wait_time[0]))

			timetable.append(single_train_run)

		return timetable


	# Discover the starting run of a certain run
	def starting_time(self, run_number):
		return self.schedule[run_number][0]

	def velocity_conversion(self):
		print(self.maximum_velocity * 360)

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

# A physical connection between two stations
class Rail_connection:

	def __init__(self, identifier, station_a, station_b, rail_connection_type, additional_runtime_percent, max_speed_usable: int = 1/2):
		# Each rail section have an identifier
		self.id = identifier
		# Station A and B are the two connected stations
		self.station_a = station_a
		self.station_b = station_b
		# A connection can be: High velocity or normal
		self.rail_connection_type = rail_connection_type
		# Maximum speed usable in the rails
		self.max_speed_usable = max_speed_usable
		# Additional Runtime Percent is the percent [0-1] of the min run time that is added to the min run time, if the train is on schedule.
		# In general, he actual run time is computed as min run time + max(0, (min run time*additionalRuntimePercent)-actual_delay).
		self.additional_runtime_percent = additional_runtime_percent

		if rail_connection_type == Connection_type.HIGH_VELOCITY_RAIL:
			self.max_speed_usable = 1
		if rail_connection_type == Connection_type.NORMAL_RAIL:
			self.max_speed_usable = 1 / 2

	def calculate_rails(self, station_a, station_b, transition_map):
		return True

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

	def __init__(self, line_id, starting_time, from_depot: bool = False, to_depot: bool = False, inverse_train_direction: bool = False):
		# Id of the line of the run
		self.line_id = line_id
		# Starting time of the run
		self.starting_time = starting_time
		# FromDepot indicates whether this run starts from a depot. Similar for ToDepot.
		self.from_depot = from_depot
		self.to_depot = to_depot
		# If inverse line direction 
		self.inverse_train_direction = inverse_train_direction


# A line is considered from a city to another, tipically joining several cities
class Line:

	def __init__(self, identifier, type_line, stations, stops):
		# ID of the line
		self.identifier = identifier
		# type of line (High velocity or regional)
		self.type_line = type_line
		# Stations where the line pass from
		self.stations = stations
		# Stops are stations where the train have to stop
		# Is an array with 0 where not stop and 1 where train stops
		self.stops = stops

		if type(stations) == int or type(stops) == int:
			print('The stations of a line should be more than one, and the dimension of the stops should be the same of the stations')
		else:
			if(len(stations)) != (len(stops)):
				print('Stations and Stops have to be the same lenght')


	def inversion_of_line(self):
		self.direction = self.direction * -1


# OLD generation train run 
'''
def single_train_run_generator(convoy_max_speed, starting_time_of_the_run, stations_to_stop, 
	railway_topology, inverse_train_direction: bool = False):
	stations_to_stop_position = []
	if not inverse_train_direction:
		for i in range(len(stations_to_stop)):
			stations_to_stop_position.append(stations_to_stop[i].position)
	else:
		for i in range(len(stations_to_stop)):
			stations_to_stop_position.append(stations_to_stop[len(stations_to_stop) - 1 - i].position)
	schedule = []
	schedule.append(starting_time_of_the_run)
	for stations in range( len(stations_to_stop) - 1 ):
		departure_station_position = stations_to_stop_position[stations]
		arrival_station_position = stations_to_stop_position[stations + 1]
		# First thing check the distance between two stations
		result = a_star(railway_topology, departure_station_position, arrival_station_position)
		# Maximum velocity a train can achieve
		train_velocity = convoy_max_speed

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
			schedule.append(int(time_needed + schedule[stations] + stations_to_stop[stations].min_wait_time[0]))
		
	return schedule
'''


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

	# No high velocity lines, so make a (0,0) position
	av_line = (0,0)

	# Define the stations
	quarto_station = Station('Quarto', position = (3, 2), capacity = 2, min_wait_time = [2, 2, 1], 
		additional_wait_percent = [0.5, 1, 1.5], importance = 80)
	quinto_station = Station('Quinto', position = (3, 9), capacity = 2, min_wait_time = [2, 2, 1], 
		additional_wait_percent = [0.5, 1, 1.5], importance = 80)

	# Define the rail connection beetween the two stations
	connection_quarto_quinto = Rail_connection(identifier = 0, station_a = quarto_station, 
		station_b = quinto_station, rail_connection_type = Connection_type.NORMAL_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])

	# Define the lines
	genova_urbana = Line(identifier = 0, type_line = Connection_type.NORMAL_RAIL, 
		stations = (quarto_station, quinto_station), stops = (1, 1))

	stations = []
	stations.append([quarto_station.position, 0.5])
	stations.append([quinto_station.position, 0.5])

	# Define the train runs
	train_run_0 = Train_run(genova_urbana, starting_time = 3, from_depot = True)
	train_run_1 = Train_run(genova_urbana, starting_time = 20, from_depot = True, inverse_train_direction = True)

	# Define the convoys
	R1079_convoy = Convoy( 'R1079', Type_of_convoy.INTERCITY)
	R1078_convoy = Convoy( 'R1078', Type_of_convoy.INTERCITY)

	# Adding the train runs to the convoys
	R1079_convoy.add_train_run(train_run_0)
	R1078_convoy.add_train_run(train_run_1)

	# Generating the schedules for the two convoys
	schedule_0 = R1079_convoy.calculate_schedule(railway_topology = rail)
	schedule_1 = R1078_convoy.calculate_schedule(railway_topology = rail)


	# Timetable has the stations positions, the schedule times, and the velocity
	timetable_example = []

	timetable_example.append([(quarto_station.position, quinto_station.position),schedule_0[0],0.5])
	timetable_example.append([((quinto_station.position[0], quinto_station.position[1]), 
		(quarto_station.position[0] - 1, quarto_station.position[1])),schedule_1[0],0.5])



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

	av_line = [(0,0)]

	# Define the stations
	quarto_station = Station('Quarto', position = (3, 2), capacity = 2, min_wait_time = [2, 2, 1], 
		additional_wait_percent = [0.5, 1, 1.5], importance = 80)
	quinto_station = Station('Quinto', position = (3, 9), capacity = 2, min_wait_time = [2, 2, 1], 
		additional_wait_percent = [0.5, 1, 1.5], importance = 80)

	# Define the rail connection beetween the two stations
	connection_quarto_quinto = Rail_connection(identifier = 0, station_a = quarto_station, 
		station_b = quinto_station, rail_connection_type = Connection_type.NORMAL_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])

	# Define the lines
	genova_urbana = Line(identifier = 0, type_line = Connection_type.NORMAL_RAIL, 
		stations = (quarto_station, quinto_station), stops = (1, 1))

	stations = []
	stations.append([quarto_station.position, 0.5])
	stations.append([quinto_station.position, 0.5])

	# Define the convoys
	R1079_convoy = Convoy( 'R1079', Type_of_convoy.INTERCITY)
	R1078_convoy = Convoy( 'R1078', Type_of_convoy.INTERCITY)


	train_runs_0 = []
	train_runs_1 = []

	starting_time_0 = 3
	starting_time_1 = 22

	for num_of_runs in range(2):
		# The first train run start from depot
		if num_of_runs == 0:
			train_runs_0.append(Train_run(genova_urbana, starting_time = starting_time_0, from_depot = True))
			starting_time_0 += 41
		# The other train runs don't start from depot
		else:
			train_runs_0.append(Train_run(genova_urbana, starting_time = starting_time_0, inverse_train_direction = True))
			starting_time_0 += 41

	for num_of_runs in range(2):
		# The first train run start from depot
		if num_of_runs == 0:
			train_runs_1.append(Train_run(genova_urbana, starting_time = starting_time_1, from_depot = True, inverse_train_direction = True))
			starting_time_1 += 41
		# The other train runs don't start from depot
		else:
			train_runs_1.append(Train_run(genova_urbana, starting_time = starting_time_1))
			starting_time_1 += 41

	schedule_0 = []
	schedule_1 = []

	for num_of_runs in range(2):
		R1079_convoy.add_train_run(train_runs_0[num_of_runs])
		R1078_convoy.add_train_run(train_runs_1[num_of_runs])

	schedule_0 = R1079_convoy.calculate_schedule(rail)
	schedule_1 = R1078_convoy.calculate_schedule(rail)


	timetable_example = []

	timetable_example.append([(quarto_station.position, quinto_station.position),schedule_0[0],0.5])
	timetable_example.append([((quinto_station.position[0] - 1, quinto_station.position[1]), (quarto_station.position[0] - 1, quarto_station.position[1])),schedule_0[1],0.5])
	timetable_example.append([((quinto_station.position[0] - 1, quinto_station.position[1]), (quarto_station.position[0] - 1, quarto_station.position[1])),schedule_1[0],0.5])
	timetable_example.append([(quarto_station.position, quinto_station.position),schedule_1[1],0.5])


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


