# import chain
from itertools import chain
from enum import IntEnum
# This is to test if the timetable is valid or not
from flatland.core.grid.grid4_astar import a_star
from configuration import av_line



class RailEnvActions(IntEnum):
	DO_NOTHING = 0  # implies change of direction in a dead-end!
	MOVE_LEFT = 1
	MOVE_FORWARD = 2
	MOVE_RIGHT = 3
	STOP_MOVING = 4
	REVERSE = 5

	@staticmethod
	def to_char(a: int):
		return {
			0: 'B',
			1: 'L',
			2: 'F',
			3: 'R',
			4: 'S',
			5: 'G'
		}[a]


# Check if the timetable is feaseble or not
def control_timetable(timetable, railway_topology):
	# Check for all the trains
	for trains in range (len(timetable)):       
		# Check for all the stations
		# Calculate the difference of two different times, so i don't need the last term to cycle          
		for stations in range (len(timetable[trains][1]) - 1):   
			if (timetable[trains][1][stations] - timetable[trains][1][stations + 1]) >= 0:
				print('===================================================================================================================================')
				print('Attention!!! The agent number', trains, 'has a problem in the timetable, times to reach stations', stations, 'and', (stations+1), 'are not right')
				print('The time to reach the successive station SHOULD BE > 0, pay attenction to the timetable')
				# Function that check if the time to reach a station defined by the timetable are possible or not,
				# Return the time minimum time to reach two different stations depending on the distance and on the line type (high velocity, regional...)
			time_to_next_station = time_to_reach_next_station(timetable[trains][0][stations], timetable[trains][0][stations + 1], railway_topology, timetable, trains)
			# Control if the time to reach the next station is possible (considering maximum velocities of lines and the distances between two stations)
			if time_to_next_station > (timetable[trains][1][stations+1]- timetable[trains][1][stations]):
				print('===================================================================================================================================')
				print('Attention!!! Agent number', trains, 'has a problem in the timetable, times to reach stations', stations, 'and', (stations+1), 'are not right')
				print('The time to reach the next station SHOULD BE HIGHER. The minimum time to reach the station should be:', time_to_next_station)
	return


# TODO spostare in station (?)
def time_in_station(station, train_velocity):
	# The len of the rails is given by the station
	len_rails = len(station.rails_in_station[0])
	# The time needed is given by the formula (len * 1/velocity + waiting time + 10% of time)
	time_needed =  len_rails * int(pow(train_velocity, -1)) + station.min_wait_time[0]
	time_needed += int(time_needed/10)
	return time_needed



# TODO aggiungere dei controlli 
# - controllare che la posizione in cui si mette il goal del secondo treno sia possibile
# - controllare il tempo necessario nella stazione (in base anche alla rialzatura in cui si mette il treno 
#   e.g. treno spostato su di due caselle, tempo necessario + 4 * velocità alla meno uno)

def check_train_in_station(timetable):
	# The number of different stations presented in the timetable
	different_stations = 0
	# The position of different stations presented in the timetable
	station_positions = []
	# Check how many different stations are in the timetable
	for i in range(len(timetable)): # for all the trains
	    for k in range(len(timetable[i][0])): # for all the stations
	        if timetable[i][0][k] not in station_positions:
	            station_positions.append(timetable[i][0][k])
	            different_stations += 1

	# Indexes contein the indexes of the station like this
	# 0 for the first station in station position 
	# 1 for the second and so on
	indexes = []
	# Local variable for the loop
	single_index = []

	for j in range(len(timetable)): # for all the trains
	    single_index = [0]*len(timetable[j][0])
	    for k in range(len(timetable[j][0])): # for all the stations
	        for i in range(len(station_positions)): # for all the different stations discovered
	            if station_positions[i] == timetable[j][0][k]:
	                single_index[k] = i
	    indexes.append(single_index)
	    
	# Here controll if two or more trains have to reach the same station at the same time
	for m in range(len(timetable)):  # for all the trains m
	    for n in range(len(timetable)): # for all the trains n with m!=n
	        for k in range(len(timetable[m][1])): # for all the stations of m
	            for l in range(len(timetable[n][1])): # for all the stations of n
	                if m!=n and m<n:
	                	# time at which the trains have to reach the stations
	                    time_train_a = timetable[m][1][k]
	                    time_train_b = timetable[n][1][l]
	                    # if they have to reach the same stations
	                    if indexes[m][k] == indexes[n][l]: 
	                    	# if the time at which they have to reach the station is similar (calculated the time needed in the stations for the trains)
	                        if time_train_a - time_train_b < 15 and time_train_a - time_train_b > -15:
	                        	# I change the gol of the convoy n (the second) 
	                            timetable[n][0][l] = (station_positions[indexes[n][l]][0] - 1, station_positions[indexes[n][l]][1])
	                            


# Define the scheduled actions the agents have to do
def action_to_do(timetable, railway_topology):
	# Path to do to arrive to the right station

	# L'idea vuole essere quella di avere un tempo necessario per stare in stazione
	# Se due treni nell'intorno di tempo devono stare nella stessa stazione bisogna indirizzarli 
	# in binari liberi diversi, quindi il passaggio diventa diverso.

	path_result = []
	# Calculate the path for all the trains
	for train_i in range (len(timetable)):
		# Number of stations in the train i
		num_of_stations = len(timetable[train_i][0])
		# The partial result a train run
		path_partial_result = []
		for station in range(num_of_stations - 1): 
			path_partial_result.append(a_star(railway_topology,timetable[train_i][0][station],timetable[train_i][0][station + 1]))

		# Final result for all the trains and train runs
		path_result.append(path_partial_result)

	# DEBUG
	'''
	print()
	print(path_result)
	print()
	'''

	# Calculate the actions that have to be done
	actions_to_do = []
	for train_i in range (len(timetable)):
		# Number of stations in the train i
		num_of_stations = len(timetable[train_i][0])
		# Flag that tells me that the next step is particular
		next = False
		# Each train occupy a row in the action_to_do matrix 
		actions_single_train = []
		# I need the direction of the last run for the reverse action
		direction_last_run = 0
		for station in range (num_of_stations - 1):
			# Each train occupy a row in the action_to_do matrix 
			actions_single_train_run = []
			for step in range (len(path_result[train_i][station])):
				# If i'm restarting from the final station of the train run, i have to wait till is the time to restart
				if len(path_result[train_i][station]) == 1:
					time_to_wait = timetable[train_i][1][station + 1] - timetable[train_i][1][station]
					for i in range(time_to_wait):
						actions_single_train_run.append(RailEnvActions.STOP_MOVING)
					continue
				# Calculate the direction of the trains at each step
				if step == 0:
					difference_y = path_result[train_i][station][step][0] - path_result[train_i][station][step + 1][0]
					difference_x = path_result[train_i][station][step][1] - path_result[train_i][station][step + 1][1]
					if difference_y == 1:
						direction = 0
					if difference_x ==  -1:
						direction = 1
					if difference_y == -1:
						direction = 2
					if difference_x == 1:
						direction = 3 
				else:
					difference_y = path_result[train_i][station][step - 1][0] - path_result[train_i][station][step][0]
					difference_x = path_result[train_i][station][step - 1][1] - path_result[train_i][station][step][1]
					if difference_y == 1:
						direction = 0
					if difference_x ==  -1:
						direction = 1
					if difference_y == -1:
						direction = 2
					if difference_x == 1:
						direction = 3 
				# Variable to count the number of possible path at each cell, is an int with the number of possible path
				if not step == 0:
					# Specific case, a train is at the boarder of two different lines, 
					# if this appen I have to consider the previous transition at the next time stamp due to the fact the velocity changes
					if next:
						multiple_path = railway_topology.get_transitions(path_result[train_i][station][step-1][0],path_result[train_i][station][step-1][1],prev_direction).count(1)
						next = False
					elif (path_result[train_i][station][step] in av_line) and not (path_result[train_i][station][step - 1] in av_line):
						multiple_path = railway_topology.get_transitions(path_result[train_i][station][step][0],path_result[train_i][station][step][1],prev_direction).count(1)
						next = True
					else:
						multiple_path = railway_topology.get_transitions(path_result[train_i][station][step-1][0],path_result[train_i][station][step-1][1],prev_direction).count(1)
				# Starting with a move forward direction for the train
				if step == 0:
					#actions_single_train.append(RailEnvActions.MOVE_FORWARD)
					prev_direction = direction
				# If I'm not at the start of the train 
				else:
					# The direction doesn't change
					if num_of_stations > 1 and station >= 1 and step <= 1:
						if (direction - direction_last_run == 2 or direction - direction_last_run == -2):
							# If I'm in an hig velocity line velocity is define only by the type of train
							actions_single_train_run.append(RailEnvActions.REVERSE)
							prev_direction = direction
							continue

					if direction - prev_direction == 0:
						# If I'm in an hig velocity line velocity is define only by the type of train
						if path_result[train_i][station][step - 1] in av_line:
							velocity = timetable[train_i][2]
						# If I'm in other line velocity is the minimum between 1/2 (the velocity of the line) and the type of train velocity
						else:
							velocity = min(timetable[train_i][2], 1/2)
						for i in range(int(pow(velocity, -1))):
							#print('Test per capire come varia',i, 'Treno numero', train_i)
							actions_single_train_run.append(RailEnvActions.MOVE_FORWARD)
						# I'm arrived at the station?
						if step == (len(path_result[train_i][station]) - 1):
							# If the next station is the last one of the train run I don't have to stop for the min wait time
							if station != (num_of_stations - 2):
								if len(path_result[train_i][station + 1]) == 1:
									continue
							# If is an intermediate station I need to stop the min wait time
							for i in range(3):   # TODO ADD THE WAITING TIMES OF THE STATIONS
								actions_single_train_run.append(RailEnvActions.STOP_MOVING)

						prev_direction = direction
					# I have to reverse the train direction when I arrive at the ending station
					elif ((direction - prev_direction) == -2) or ((direction - prev_direction) == 2):
						# If I'm in an hig velocity line velocity is define only by the type of train
						if path_result[train_i][station][step - 1] in av_line:
							velocity = timetable[train_i][2]
						else:
							velocity = min(timetable[train_i][2], 1/2)
						for i in range(int(pow(velocity, -1))):
							actions_single_train_run.append(RailEnvActions.REVERSE)
						prev_direction = direction
					# I have to move to left 
					# and I have more then one possible path, so I go left at the deviation
					# Depending on the direction of march the results can be -1 or -3
					elif ((direction - prev_direction == -1) and (multiple_path > 1)) or ((direction - prev_direction == +3)):
						# If I'm in an hig velocity line velocity is define only by the type of train
						if path_result[train_i][station][step - 1] in av_line:
							velocity = timetable[train_i][2]
						# If I'm in other line velocity is the minimum between 1/2 (the velocity of the line) and the type of train velocity
						else:
							velocity = min(timetable[train_i][2], 1/2)
						for i in range(int(pow(velocity, -1))):
							actions_single_train_run.append(RailEnvActions.MOVE_LEFT)
						prev_direction = direction
					# I have to move right 
					# and I have more then one possible path, so I go left at the deviation 
					# Depending on the direction of march the results can be +1 or -3
					elif ((direction - prev_direction == 1) and (multiple_path > 1)) or ((direction - prev_direction == -3) ):
						# If I'm in an hig velocity line velocity is define only by the type of train
						if path_result[train_i][station][step - 1] in av_line:
							velocity = timetable[train_i][2]
						# If I'm in other line velocity is the minimum between 1/2 (the velocity of the line) and the type of train velocity
						else:
							velocity = min(timetable[train_i][2], 1/2)
						for i in range(int(pow(velocity, -1))):
							actions_single_train_run.append(RailEnvActions.MOVE_RIGHT)
						prev_direction = direction
					else:
						if path_result[train_i][station][step - 1] in av_line:
							velocity = timetable[train_i][2]
						else:
							velocity = min(timetable[train_i][2], 1/2)
						for i in range(int(pow(velocity, -1))):
							actions_single_train_run.append(RailEnvActions.MOVE_FORWARD)
						# I'm arrived at the station?
						if step == (len(path_result[train_i][station]) - 1):
							for i in range(3):   # TODO ADD THE WAITING TIMES OF THE STATIONS
								actions_single_train_run.append(RailEnvActions.STOP_MOVING)
						prev_direction = direction
			direction_last_run = direction

			actions_single_train.append(actions_single_train_run)
 
		if isinstance(actions_single_train[0], list):
			len(actions_single_train)
			actions_single_train = list(chain.from_iterable(actions_single_train))
		actions_to_do.append(actions_single_train)

	return actions_to_do



# Calculate the time to reach the stations to understand if timetable is right
def time_to_reach_next_station(departure_station_position, arrival_station_position, railway_topology, schedule, train_number):
	# First thing check the distance between two stations 
	result = a_star(railway_topology, departure_station_position, arrival_station_position)
	# Maximum velocity a train can achieve
	train_velocity = schedule[train_number][2]

	lenght_path = len(result)  # distance between stations

	# Array when I put at each step the time needed to make the path
	# The total time is the sum of the numbers
	time_array = []
	# Check the at each step which train i am and which line im in
	for step in range(lenght_path):
		if (result[step]) in av_line:
			time_array.append(pow(train_velocity,-1))
		else:
			time_array.append(pow(min(train_velocity, 1/2), -1))
	time_needed = sum(time_array)

	#print((time_needed + int(time_needed/10))) DEBUG

	# Adding to the time a 10% to face with problems in case it's neaded
	return (time_needed + int(time_needed/10))