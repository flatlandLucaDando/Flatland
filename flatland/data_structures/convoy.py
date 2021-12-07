from enum import Enum
import itertools

# Type of a convoy, can be high velocity or regional
# The velocity are given by default depending on the type of convoy, 360 for HV, 180 for IC, 120 for Regional
class Type_of_convoy(Enum):
	HIGH_VELOCITY = 1
	INTERCITY = 2
	REGIONAL = 3

# A convoy is a locomotive + wagons.
class Convoy:

	# For the id
	id_iter = itertools.count()

	def __init__(self, train_type, schedule = []):

		# identifier of the train
		self.id = next(Convoy.id_iter)
		# type of train (High velocity, Intercity, regional)
		self.train_type = train_type
		# schedule of the train
		self.schedule = []

		if train_type == Type_of_convoy.HIGH_VELOCITY:
			self.maximum_velocity = 1
			self.importance = 1
		if train_type == Type_of_convoy.INTERCITY:
			self.maximum_velocity = 1/2
			self.importance = 0.75
		if train_type == Type_of_convoy.REGIONAL:
			self.maximum_velocity = 1/3
			self.importance = 0.5

	def add_train_run(self, train_run):
		self.schedule.append(train_run)

	# Discover the starting time of a certain run
	def starting_time(self, run_number):
		return self.schedule[run_number][0]

	# Convert velocity (maximum possible velocity is 360)
	def velocity_conversion(self):
		print(self.maximum_velocity * 360)

	# Verificate that a schedule is possible (if someone want to write manually)
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

'''
	# OLD FUNCTION (wait to delete to discover if i need in future)
	def calculate_schedule(self, railway_topology):
		# The timetable that should be returned
		timetable = []
		# For each train run defined
		for num_of_runs in range(len(self.schedule)):
			# The single train run
			single_train_run = []
			# The number of station to pass
			num_of_stations = len(self.schedule[num_of_runs].line_belongin.stations)
			# The station to stop
			stations_to_stop_position = []
			# Direction not inverted?
			if not self.schedule[num_of_runs].inverse_train_direction:
				for i in range(num_of_stations):
					# append the station position in the right order
					stations_to_stop_position.append(self.schedule[num_of_runs].line_belongin.stations[i].position)
			# Direction inverdet?
			else:
				for i in range(num_of_stations):
					# append the station position in inverted order
					stations_to_stop_position.append(self.schedule[num_of_runs].line_belongin.stations[num_of_stations - 1 - i].position)
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
					single_train_run.append(int(time_needed + single_train_run[stations] + self.schedule[num_of_runs].line_belongin.stations[stations].min_wait_time[0]))

			timetable.append(single_train_run)

		return timetable

	'''