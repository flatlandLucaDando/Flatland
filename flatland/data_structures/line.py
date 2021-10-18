import itertools

# A line is considered from a city to another, tipically joining several cities
class Line:

	# For the id
	id_iter = itertools.count()

	def __init__(self, type_line, stations, stops):
		# ID of the line
		self.id = next(Line.id_iter)
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