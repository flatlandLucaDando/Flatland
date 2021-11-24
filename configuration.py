from typing import Type
import numpy as np
import os
# Import the structures
from flatland.data_structures.convoy import Convoy, Type_of_convoy
from flatland.data_structures.rail_connection import Rail_connection, Connection_type
from flatland.data_structures.station import Station
from flatland.data_structures.train_run import Train_run
from flatland.data_structures.line import Line
# Import the timetable utils
from flatland.envs.plan_to_follow_utils import calculate_timetable

example_num = 'training0'
'''
###############################################################
######################   EXAMPLE 1  #########################
###############################################################
'''

if example_num == 1:
	# Import the examples
	from examples.new_example_1 import rail, railway_example, av_line
	# Define the stations
	quarto_station = Station('Quarto', position = (3, 2), capacity = 2, min_wait_time = [2, 2, 1], 
		additional_wait_percent = [0.5, 1, 1.5], importance = 80, railway_topology = rail)
	quinto_station = Station('Quinto', position = (3, 9), capacity = 2, min_wait_time = [2, 2, 1], 
		additional_wait_percent = [0.5, 1, 1.5], importance = 80, railway_topology = rail)

	# Define the rail connection beetween the two stations
	connection_quarto_quinto = Rail_connection(station_a = quarto_station, 
		station_b = quinto_station, rail_connection_type = Connection_type.NORMAL_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])

	# Define the lines
	genova_urbana = Line(type_line = Connection_type.NORMAL_RAIL, 
		stations = (quarto_station, quinto_station), stops = (1, 1))

	stations = []
	stations.append([quarto_station.position, 0.5])
	stations.append([quinto_station.position, 0.5])

	stations_objects = [quarto_station, quinto_station]

	# Define the train runs
	train_run_0 = Train_run(genova_urbana, starting_time = 3, from_depot = True)
	train_run_1 = Train_run(genova_urbana, starting_time = 10, from_depot = True, inverse_train_direction = True)
	train_run_2 = Train_run(genova_urbana, starting_time = 40, inverse_train_direction = True)
	train_run_3 = Train_run(genova_urbana, starting_time = 70)

	# Define the convoys
	R1079_convoy = Convoy( Type_of_convoy.INTERCITY)
	R1078_convoy = Convoy( Type_of_convoy.INTERCITY)

	convoys = [R1079_convoy, R1078_convoy]

	# Adding the train runs to the convoys
	R1079_convoy.add_train_run(train_run_0)
	R1079_convoy.add_train_run(train_run_2)
	R1078_convoy.add_train_run(train_run_1)
	R1078_convoy.add_train_run(train_run_3)

	# Generating the timetable
	# The timetable is composed by (station positions, time at which reach the stations, maximum train velocity)
	timetable_example = calculate_timetable(convoys, rail)


	# TODO crea una funzione (per ora non è una funzione, serve quella) per esportare le timetable come excel
	from pandas import DataFrame
	import pandas as pd

	df_quarto = [0]* len(convoys)
	df_quinto = [0]* len(convoys)
	df = [0]* len(convoys)
	quarto = []
	quinto = []
	for i in range(len(convoys)):
		for j in range(len(stations_objects)):
			for k in range(len(timetable_example[i][0])):
				if j == 0:
					if timetable_example[i][0][k] == stations_objects[j].position:
						quarto.append(timetable_example[i][1][k])
				if j == 1:
					if timetable_example[i][0][k] == stations_objects[j].position:
						quinto.append(timetable_example[i][1][k])
		df_quarto[i] = DataFrame({stations_objects[0].name : quarto})
		df_quinto[i]  = DataFrame({stations_objects[1].name : quinto})
		frames = [df_quarto[i], df_quinto[i]]
		df[i] = pd.concat(frames, axis=1)
		quarto = []
		quinto = []

		# funtion
	def multiple_dfs(df_list, sheets, file_name, spaces):
	    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')   
	    row = 0
	    for dataframe in df_list:
	        dataframe.to_excel(writer,sheet_name=sheets,startrow=row , startcol=0)   
	        row = row + len(dataframe.index) + spaces + 1
	    writer.save()

	# run function
	os.makedirs("output/timetables", exist_ok=True)
	multiple_dfs(df, 'Validation', 'output/timetables/timetable_test_1.xlsx', 1)

if example_num == 2:
	# Import the examples
	from examples.new_example_2 import rail, railway_example, av_line

	# Define the stations
	# TODO raagiona sul come definire meglio la capacità delle stazioni e gestire i binari....
	quarto_station = Station('Quarto', position = (6, 2), capacity = 5, min_wait_time = [2, 2, 1], 
		additional_wait_percent = [0.5, 1, 1.5], importance = 80, railway_topology = rail)
	quinto_station = Station('Quinto', position = (6, 17), capacity = 5, min_wait_time = [2, 2, 1], 
		additional_wait_percent = [0.5, 1, 1.5], importance = 80, railway_topology = rail)


	# Define the rail connection beetween the two stations
	connection_quarto_quinto = Rail_connection(station_a = quarto_station, 
		station_b = quinto_station, rail_connection_type = Connection_type.NORMAL_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])


	# Define the lines
	genova_urbana = Line(type_line = Connection_type.NORMAL_RAIL, 
		stations = (quarto_station, quinto_station), stops = (1, 1))

	stations = []
	stations.append([quarto_station.position, 0.5])
	stations.append([quinto_station.position, 0.5])
	stations_objects = [quarto_station, quinto_station]

	# Define the train runs
	train_run_0 = Train_run(genova_urbana, starting_time = 3, from_depot = True)
	train_run_1 = Train_run(genova_urbana, starting_time = 40, from_depot = True, inverse_train_direction = True)
 
	train_run_2 = Train_run(genova_urbana, starting_time = 55, inverse_train_direction = True)
 
	train_run_3 = Train_run(genova_urbana, starting_time = 20, from_depot = True)
	train_run_5 = Train_run(genova_urbana, starting_time = 70, inverse_train_direction = True, to_depot = True)
 
	train_run_4 = Train_run(genova_urbana, starting_time = 20, from_depot = True)
	train_run_6 = Train_run(genova_urbana, starting_time = 75, inverse_train_direction = True, to_depot = True)

	# Define the convoys
	R1079_convoy = Convoy( Type_of_convoy.INTERCITY)
	R1078_convoy = Convoy( Type_of_convoy.INTERCITY)
	R1077_convoy = Convoy( Type_of_convoy.INTERCITY)

	# SOME MODIFICATION TO STUDY TRAINING
	#R1076_convoy = Convoy( Type_of_convoy.INTERCITY)

	convoys = [R1079_convoy, R1078_convoy, R1077_convoy] #, R1076_convoy]

	# Adding the train runs to the convoys
	# R1079
	R1079_convoy.add_train_run(train_run_0)
	R1079_convoy.add_train_run(train_run_2)
	# R1078
	R1078_convoy.add_train_run(train_run_1)
	# R1077
	R1077_convoy.add_train_run(train_run_3)
	R1077_convoy.add_train_run(train_run_5)
	# R1076
	#R1076_convoy.add_train_run(train_run_4)

	# Generating the timetable
	# The timetable is composed by (station positions, time at which reach the stations, maximum train velocity)
	timetable_example = calculate_timetable(convoys, rail)

'''
###############################################################
######################   EXAMPLE 3  #########################
###############################################################
'''

if example_num == 3:
	# Import the examples
	from examples.new_example_3 import rail, railway_example, av_line

	# Define the stations
	genova_station = Station('Genova', position = (21,2), capacity = 2, min_wait_time = [2, 2, 1], 
		additional_wait_percent = [0.5, 1, 1.5], importance = 80, railway_topology = rail)
	pavia_station = Station('Pavia', position = (21,18), capacity = 2, min_wait_time = [2, 2, 1],
		additional_wait_percent = [0.5, 1, 1.5], importance = 80, railway_topology = rail)
	milano_station = Station('Milano', position = (15,49), capacity = 2, min_wait_time = [2, 2, 1],
		additional_wait_percent = [0.5, 1, 1.5], importance = 80, railway_topology = rail)
	torino_station = Station('Torino', position = (9,1), capacity = 2, min_wait_time = [2, 2, 1],
		additional_wait_percent = [0.5, 1, 1.5], importance = 80, railway_topology = rail)
	novara_station = Station('Novara', position = (3,40), capacity = 2, min_wait_time = [2, 2, 1],
		additional_wait_percent = [0.5, 1, 1.5], importance = 80, railway_topology = rail)

	# Define the connections between station
	connection_genova_pavia = Rail_connection(station_a = genova_station, 
		station_b = pavia_station, rail_connection_type = Connection_type.HIGH_VELOCITY_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])
	connection_pavia_milano = Rail_connection(station_a = pavia_station, 
		station_b = milano_station, rail_connection_type = Connection_type.HIGH_VELOCITY_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])
	connection_genova_torino = Rail_connection(station_a = pavia_station, 
		station_b = milano_station, rail_connection_type = Connection_type.NORMAL_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])
	connection_torino_novara = Rail_connection(station_a = torino_station, 
		station_b = novara_station, rail_connection_type = Connection_type.NORMAL_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])
	connection_novara_pavia = Rail_connection(station_a = novara_station, 
		station_b = pavia_station, rail_connection_type = Connection_type.NORMAL_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])
	connection_torino_pavia = Rail_connection(station_a = torino_station, 
		station_b = pavia_station, rail_connection_type = Connection_type.NORMAL_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])

	# Define the lines
	genova_milano = Line(type_line = Connection_type.HIGH_VELOCITY_RAIL, 
		stations = (genova_station, pavia_station, milano_station), stops = (1, 1, 1))
	torino_milano = Line(type_line = Connection_type.NORMAL_RAIL, 
		stations = (torino_station, novara_station, milano_station), stops = (1, 1, 1))
	genova_torino = Line(type_line = Connection_type.NORMAL_RAIL, 
		stations = (genova_station, torino_station), stops = (1, 1))

	stations = []
	stations.append([genova_station.position, 1])
	stations.append([pavia_station.position, 1])
	stations.append([milano_station.position, 1])
	stations.append([torino_station.position, 0.5])
	stations.append([novara_station.position, 0.5])

	stations_objects = [genova_station, pavia_station, milano_station, torino_station, novara_station]

	# Defining the train runs
	# convoy 1
	train_run_0 = Train_run(genova_milano, starting_time = 5, from_depot = True)
	# convoy 2
	train_run_1 = Train_run(genova_milano, starting_time = 7, from_depot = True, inverse_train_direction = True)
	# convoy 3
	train_run_2 = Train_run(torino_milano, starting_time = 12, from_depot = True)
	# convoy 4
	train_run_3 = Train_run(torino_milano, starting_time = 3, from_depot = True, inverse_train_direction = True)
	# convoy 5
	train_run_4 = Train_run(genova_torino, starting_time = 11, from_depot = True)
	# convoy 6
	train_run_5 = Train_run(genova_torino, starting_time = 20, from_depot = True, inverse_train_direction = True)

	# Define the convoysss
	FR1001_convoy = Convoy( Type_of_convoy.HIGH_VELOCITY)
	R1002_convoy = Convoy( Type_of_convoy.INTERCITY)	
	R1003_convoy = Convoy( Type_of_convoy.INTERCITY)
	FR1004_convoy = Convoy( Type_of_convoy.HIGH_VELOCITY)
	R1005_convoy = Convoy( Type_of_convoy.INTERCITY)
	R1006_convoy = Convoy( Type_of_convoy.INTERCITY)

	convoys = [FR1001_convoy, R1002_convoy, R1003_convoy, FR1004_convoy, R1005_convoy, R1006_convoy]

	# Add the train runs to convoys
	FR1001_convoy.add_train_run(train_run_0)

	R1002_convoy.add_train_run(train_run_1)

	R1003_convoy.add_train_run(train_run_2)

	FR1004_convoy.add_train_run(train_run_3)

	R1005_convoy.add_train_run(train_run_4)

	R1006_convoy.add_train_run(train_run_5)

	#Calculate plan to do

	timetable_example = calculate_timetable(convoys, rail)

'''
###############################################################
######################   EXAMPLE 0  #########################
###############################################################
'''

if example_num == 0:

	# Import the examples
	from examples.esempio_prova import rail, railway_example, av_line

	stazione_prova = Station('Prova', position = (3,2), capacity = 1, min_wait_time = 1, additional_wait_percent =1, importance = 1, railway_topology = rail)
	stazione_prova_2 = Station('Prova 2', position = (3, 9), capacity = 1, min_wait_time = 1, additional_wait_percent =1, importance = 1, railway_topology = rail)

	connection_prova = Rail_connection(station_a = stazione_prova, 
		station_b = stazione_prova_2, rail_connection_type = Connection_type.NORMAL_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])

	linea_prova = Line(type_line = Connection_type.NORMAL_RAIL, 
		stations = (stazione_prova, stazione_prova_2), stops = (1, 1))

	stations = []
	stations.append([stazione_prova.position, 0.5])
	stations.append([stazione_prova_2.position, 0.5])

	# Define the train runs
	train_run_0 = Train_run(linea_prova, starting_time = 3, from_depot = True)
	train_run_1 = Train_run(linea_prova, starting_time = 40, from_depot = True, inverse_train_direction = True)

	# Define the convoys
	R1079_convoy = Convoy( Type_of_convoy.INTERCITY)
	R1078_convoy = Convoy( Type_of_convoy.INTERCITY)

	convoys = [R1079_convoy, R1078_convoy]

	# Adding the train runs to the convoys
	R1079_convoy.add_train_run(train_run_0)
	R1078_convoy.add_train_run(train_run_1)

	# Generating the timetable
	# The timetable is composed by (station positions, time at which reach the stations, maximum train velocity)
	timetable_example = calculate_timetable(convoys, rail)

'''
###############################################################
######################   Training 0  #########################
###############################################################
'''

if example_num == 'training0':
	# Import the examples
	from examples.ferrovia_luca import rail, railway_example, av_line

	genova_station = Station('Genova', position = (6,2), capacity = 3, min_wait_time = [1, 1, 1], additional_wait_percent =1, importance = 1, railway_topology = rail)
	recco_station = Station('Recco', position = (6,13), capacity = 3, min_wait_time = [1, 1, 1], additional_wait_percent =1, importance = 1, railway_topology = rail)
	chiavari_station = Station('Chiavari', position = (6,23), capacity = 3, min_wait_time = [1, 1, 1], additional_wait_percent =1, importance = 1, railway_topology = rail)

	connection_genova_recco = Rail_connection(station_a = genova_station, 
		station_b = recco_station, rail_connection_type = Connection_type.NORMAL_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])

	connection_recco_chiavari = Rail_connection(station_a = recco_station, 
		station_b = chiavari_station, rail_connection_type = Connection_type.NORMAL_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])

	linea_genova_levante = Line(type_line = Connection_type.NORMAL_RAIL, 
		stations = (genova_station, recco_station, chiavari_station), stops = (1, 1, 1))

	stations = []

	stations.append([genova_station.position, 0.5])
	stations.append([recco_station.position, 0.5])
	stations.append([chiavari_station, 0.5])

	train_run_0 = Train_run(linea_genova_levante, starting_time = 3, from_depot = True)

	R102_convoy = Convoy(Type_of_convoy.INTERCITY)

	convoys = [R102_convoy]

	R102_convoy.add_train_run(train_run_0)

	timetable_example = calculate_timetable(convoys, rail)

	# Interruption
	timetable_example.append([[(6,8), (6,11)],[1, 225], 0.5])
	timetable_example.append([[(5,8), (5,11)],[1, 225], 0.5])

'''
###############################################################
######################   Training 1  #########################
###############################################################
'''

if example_num == 'training1':
	# Import the examples
	from examples.ferrovia_luca import rail, railway_example, av_line

	genova_station = Station('Genova', position = (6,2), capacity = 3, min_wait_time = [1, 1, 1], additional_wait_percent =1, importance = 1, railway_topology = rail)
	recco_station = Station('Recco', position = (6,13), capacity = 3, min_wait_time = [1, 1, 1], additional_wait_percent =1, importance = 1, railway_topology = rail)
	chiavari_station = Station('Chiavari', position = (6,23), capacity = 3, min_wait_time = [1, 1, 1], additional_wait_percent =1, importance = 1, railway_topology = rail)

	connection_genova_recco = Rail_connection(station_a = genova_station, 
		station_b = recco_station, rail_connection_type = Connection_type.NORMAL_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])

	connection_recco_chiavari = Rail_connection(station_a = recco_station, 
		station_b = chiavari_station, rail_connection_type = Connection_type.NORMAL_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])

	linea_genova_levante = Line(type_line = Connection_type.NORMAL_RAIL, 
		stations = (genova_station, recco_station, chiavari_station), stops = (1, 1, 1))

	stations = []

	stations.append([genova_station.position, 0.5])
	stations.append([recco_station.position, 0.5])
	stations.append([chiavari_station, 0.5])

	train_run_0 = Train_run(linea_genova_levante, starting_time = 3, from_depot = True)

	train_run_1 = Train_run(linea_genova_levante, starting_time = 3, from_depot = True, inverse_train_direction = True)

	R102_convoy = Convoy(Type_of_convoy.INTERCITY)
	R103_convoy = Convoy(Type_of_convoy.INTERCITY)

	convoys = [R102_convoy, R103_convoy]

	R102_convoy.add_train_run(train_run_0)

	R103_convoy.add_train_run(train_run_1)

	timetable_example = calculate_timetable(convoys, rail)

	# Interruption
	timetable_example.append([[(6,8), (6,11)],[1, 225], 0.5])
	timetable_example.append([[(5,8), (5,11)],[1, 225], 0.5])


'''
###############################################################
######################   Training 1.1  #########################
###############################################################
'''

if example_num == 'training1.1':
	# Import the examples
	from examples.ferrovia_luca import rail, railway_example, av_line

	genova_station = Station('Genova', position = (6,2), capacity = 3, min_wait_time = [1, 1, 1], additional_wait_percent =1, importance = 1, railway_topology = rail)
	recco_station = Station('Recco', position = (6,13), capacity = 3, min_wait_time = [1, 1, 1], additional_wait_percent =1, importance = 1, railway_topology = rail)
	chiavari_station = Station('Chiavari', position = (6,23), capacity = 3, min_wait_time = [1, 1, 1], additional_wait_percent =1, importance = 1, railway_topology = rail)

	connection_genova_recco = Rail_connection(station_a = genova_station, 
		station_b = recco_station, rail_connection_type = Connection_type.NORMAL_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])

	connection_recco_chiavari = Rail_connection(station_a = recco_station, 
		station_b = chiavari_station, rail_connection_type = Connection_type.NORMAL_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])

	linea_genova_levante = Line(type_line = Connection_type.NORMAL_RAIL, 
		stations = (genova_station, recco_station, chiavari_station), stops = (1, 1, 1))

	stations = []

	stations.append([genova_station.position, 0.5])
	stations.append([recco_station.position, 0.5])
	stations.append([chiavari_station, 0.5])

	train_run_0 = Train_run(linea_genova_levante, starting_time = 3, from_depot = True)

	train_run_1 = Train_run(linea_genova_levante, starting_time = 3, from_depot = True, inverse_train_direction = True)

	R102_convoy = Convoy(Type_of_convoy.INTERCITY)
	R103_convoy = Convoy(Type_of_convoy.INTERCITY)

	convoys = [R102_convoy, R103_convoy]

	R102_convoy.add_train_run(train_run_0)

	R103_convoy.add_train_run(train_run_1)

	timetable_example = calculate_timetable(convoys, rail)

	# Interruption
	timetable_example.append([[(6,8), (6,11)],[1, 225], 0.5])