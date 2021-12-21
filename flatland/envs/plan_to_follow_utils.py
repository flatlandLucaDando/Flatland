# import chain
from itertools import chain
from enum import IntEnum
import random
# This is to test if the timetable is valid or not
from flatland.core.grid.grid4_astar import a_star
from structures_rail import av_line

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
        for stations in range (len(timetable[trains][1])-1):   
            if (timetable[trains][1][stations] - timetable[trains][1][stations + 1]) >= 0:
                print('===================================================================================================================================')
                print('Attention!!! The agent number', trains, 'has a problem in the timetable, times to reach stations', stations, 'and', (stations+1), 'are not right')
                print('The time to reach the successive station SHOULD BE > 0, pay attenction to the timetable')
                # Function that check if the time to reach a station defined by the timetable are possible or not,
                # Return the time minimum time to reach two different stations depending on the distance and on the line type (high velocity, regional...)
            time_to_next_station = time_to_reach_next_station(timetable[trains][0][stations].position , timetable[trains][0][stations + 1].position , railway_topology, timetable, trains)
            # Control if the time to reach the next station is possible (considering maximum velocities of lines and the distances between two stations)
            if time_to_next_station > (timetable[trains][1][stations+1]- timetable[trains][1][stations]):
                print('===================================================================================================================================')
                print('Attention!!! Agent number', trains, 'has a problem in the timetable, times to reach stations', stations, 'and', (stations+1), 'are not right')
                print('The time to reach the next station SHOULD BE HIGHER. The minimum time to reach the station should be:', time_to_next_station)
    return

# TODO aggiungere dei controlli 
# - controllare che la posizione in cui si mette il goal del secondo treno sia possibile
# - controllare il tempo necessario nella stazione (in base anche alla rialzatura in cui si mette il treno 
#   e.g. treno spostato su di due caselle, tempo necessario + 4 * velocità alla meno uno)

def divide_trains_in_station_rails(timetable, railway_topology):
    # The number of different stations presented in the timetable
    different_stations = 0
    # The position of different stations presented in the timetable
    station_positions = []
    # Check how many different stations are in the timetable
    for i in range(len(timetable)): # for all the trains
        for k in range(len(timetable[i][0])): # for all the stations
            if timetable[i][0][k].position not in station_positions:
                station_positions.append(timetable[i][0][k].position)
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
                if station_positions[i] == timetable[j][0][k].position:
                    single_index[k] = i
        indexes.append(single_index)
    
    # Stations have a capacity, if more trains with respect to the capacity
    # of the station are present there is a problem
    counter_of_trains = [0] * different_stations  # counter of trains in a certain station
    
    # TODO fai un ciclo for che cicli gli step nella tabella oraria per aggiornare il counter trains
    # quando più treni sono nella stessa stazione aumenta, quando un treno esce dalla stazione diminuisce
    # aggiungi una lettera che cicli sulle due stazioni	(ad ora le stazioni girano su tutta la timetable)
    # Calculating the maximum time the agents have to stay in env 
    
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
                            threshold_time = calculate_time_in_station(timetable,m,n,k,l)
                            # if the time at which they have to reach the station is similar (calculated the time needed in the stations for the trains)
                            if time_train_a - time_train_b < threshold_time and time_train_a - time_train_b > -threshold_time:
                                if railway_topology.grid[(station_positions[indexes[n][l]][0] - 1, station_positions[indexes[n][l]][1])] != 0:
                                    # I change the goal of the convoy that first reach the station
                                    timetable[n][3][l] = (station_positions[indexes[n][l]][0] - 1, station_positions[indexes[n][l]][1])
    for i in range(len(timetable)):
        for j in range(len(timetable[i][0])):
            if timetable[i][3][j] == 0:
                timetable[i][3][j] = timetable[i][0][j].position
                

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
            path_partial_result.append(a_star(railway_topology,timetable[train_i][3][station], timetable[train_i][3][station + 1]))
            if path_partial_result == []:
                raise ImportError('There s not a path between station', station, 'and station', station + 1 )

        # Final result for all the trains and train runs
        path_result.append(path_partial_result)

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
                            velocity = timetable[train_i][2].maximum_velocity
                        # If I'm in other line velocity is the minimum between 1/2 (the velocity of the line) and the type of train velocity
                        else:
                            velocity = min(timetable[train_i][2].maximum_velocity, 1/2)
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
                            velocity = timetable[train_i][2].maximum_velocity
                        else:
                            velocity = min(timetable[train_i][2].maximum_velocity, 1/2)
                        for i in range(int(pow(velocity, -1))):
                            actions_single_train_run.append(RailEnvActions.REVERSE)
                        prev_direction = direction
                    # I have to move to left 
                    # and I have more then one possible path, so I go left at the deviation
                    # Depending on the direction of march the results can be -1 or -3
                    elif ((direction - prev_direction == -1) and (multiple_path > 1)) or ((direction - prev_direction == +3)):
                        # If I'm in an hig velocity line velocity is define only by the type of train
                        if path_result[train_i][station][step - 1] in av_line:
                            velocity = timetable[train_i][2].maximum_velocity
                        # If I'm in other line velocity is the minimum between 1/2 (the velocity of the line) and the type of train velocity
                        else:
                            velocity = min(timetable[train_i][2].maximum_velocity, 1/2)
                        for i in range(int(pow(velocity, -1))):
                            actions_single_train_run.append(RailEnvActions.MOVE_LEFT)
                        prev_direction = direction
                    # I have to move right 
                    # and I have more then one possible path, so I go left at the deviation 
                    # Depending on the direction of march the results can be +1 or -3
                    elif ((direction - prev_direction == 1) and (multiple_path > 1)) or ((direction - prev_direction == -3) ):
                        # If I'm in an hig velocity line velocity is define only by the type of train
                        if path_result[train_i][station][step - 1] in av_line:
                            velocity = timetable[train_i][2].maximum_velocity
                        # If I'm in other line velocity is the minimum between 1/2 (the velocity of the line) and the type of train velocity
                        else:
                            velocity = min(timetable[train_i][2].maximum_velocity, 1/2)
                        for i in range(int(pow(velocity, -1))):
                            actions_single_train_run.append(RailEnvActions.MOVE_RIGHT)
                        prev_direction = direction
                    else:
                        if path_result[train_i][station][step - 1] in av_line:
                            velocity = timetable[train_i][2].maximum_velocity
                        else:
                            velocity = min(timetable[train_i][2].maximum_velocity, 1/2)
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
    train_velocity = schedule[train_number][2].maximum_velocity

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

# TODO calculate the av_rails, in order to distinguish them
# TODO calculate the right,left,up,down rails, in order to distinguish them
# TODO understad if the velocities are realistic or not (360 km for high velocity, 180 and 120 is realistic or not?)

def calculate_timetable(convoys, railway_topology):
    # The timetable that should be returned
    timetable = []
    # For each convoy
    for convoy_i in range(len(convoys)):
        # For each train run defined
        single_convoy_schedule = convoys[convoy_i].schedule
        single_convoy_schedule_len = len(convoys[convoy_i].schedule)
        single_convoy = []
        for num_of_runs in range(single_convoy_schedule_len):
            # The single train run
            single_train_run = []
            # The number of station to pass
            num_of_stations = len(single_convoy_schedule[num_of_runs].line_belongin.stations)
            # The station to stop
            stations_to_stop = []
            # Direction not inverted?
            if not single_convoy_schedule[num_of_runs].inverse_train_direction:
                for i in range(num_of_stations):
                    # append the station position in the right order
                    stations_to_stop.append(single_convoy_schedule[num_of_runs].line_belongin.stations[i])
            # Direction inverdet?
            else:
                for i in range(num_of_stations):
                    # append the station position in inverted order
                    stations_to_stop.append(single_convoy_schedule[num_of_runs].line_belongin.stations[num_of_stations - 1 - i])
            # Adding the starting time
            single_train_run.append(single_convoy_schedule[num_of_runs].starting_time)

            for stations in range(num_of_stations -1):
                departure_station_position = stations_to_stop[stations].position
                arrival_station_position = stations_to_stop[stations + 1].position
                # First thing check the distance between two stations
                result = a_star(railway_topology, departure_station_position, arrival_station_position)
                # Maximum velocity a train can achieve
                train_velocity = convoys[convoy_i].maximum_velocity 

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
                    if len(single_train_run) == 1 and type(single_convoy_schedule[num_of_runs].line_belongin.stations[stations].min_wait_time) == int:
                        single_train_run.append(int(time_needed + single_train_run[0] + single_convoy_schedule[num_of_runs].line_belongin.stations[stations].min_wait_time))
                    else:
                        # sum of time needed, the precedence time and the waiting time at the station
                        single_train_run.append(int(time_needed + single_train_run[stations] + single_convoy_schedule[num_of_runs].line_belongin.stations[stations].min_wait_time[0]))

            single_convoy.append(stations_to_stop)
            single_convoy.append(single_train_run)
            
        timetable.append(single_convoy)

    # This is needed in order to obtein the timetable-standard structure
    final_timetable = [] # The final timetable
    timetable_single_convoy = [] # Timetable of a single convoy
    timetable_stations_example = [] # Partial timetable with the position of the stations to pass
    timetable_time_example = []  # Partial timetable with the time at which reach the stations
    single_stations_timetable = []  # position and time of a single train run
    single_time_timetable = []


    for i in range(len(timetable)):
        for j in range(len(timetable[i])):
            if j % 2 == 0:
                timetable_stations_example.append(timetable[i][j])
            else:
                timetable_time_example.append(timetable[i][j])
        for k in range(len(timetable_stations_example)):
            single_stations_timetable += (timetable_stations_example[k])
        for k in range(len(timetable_time_example)):
            single_time_timetable += timetable_time_example[k]
        # The standard timetable form is (positions, time, train velocity)
        timetable_single_convoy.append(single_stations_timetable)
        timetable_single_convoy.append(single_time_timetable)
        timetable_single_convoy.append(convoys[i])
        timetable_single_convoy.append([0]*len(timetable[i][0]))   # Array for the rails of the stations in which the train have to stop
        # Final timetable
        final_timetable.append(timetable_single_convoy)
        # Restart the partial results
        single_stations_timetable = []
        timetable_stations_example = []
        single_time_timetable = []
        timetable_time_example = []
        timetable_single_convoy = []

    return final_timetable

def calculate_time_in_station(timetable,train_a,train_b,index_a,index_b):
    time_a = 15
    time_b = 15
    if index_a != 0:
        time_a = timetable[train_a][1][index_a] - timetable[train_a][1][index_a - 1] + 15
    if index_b != 0:
        time_b = timetable[train_b][1][index_b] - timetable[train_b][1][index_b - 1] + 15
    return max(time_a,time_b)