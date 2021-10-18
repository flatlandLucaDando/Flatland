# A train run is the run on a line for a train (e.g. Genova-Milano --- Milano-Genova)
# The train run consider each intermediate station the train has to pass between the two "principal" stations
class Train_run:

	def __init__(self, line_belongin, starting_time, from_depot: bool = False, to_depot: bool = False, inverse_train_direction: bool = False):
		# Line in which the train run is, so it contein informations about the stations to stop and etc etc
		self.line_belongin = line_belongin
		# Starting time of the run
		self.starting_time = starting_time
		# FromDepot indicates whether this run starts from a depot. Similar for ToDepot.
		self.from_depot = from_depot
		self.to_depot = to_depot
		# If inverse line direction 
		self.inverse_train_direction = inverse_train_direction

