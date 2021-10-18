from enum import Enum
import itertools

# A connection can be: High velocity or normal
# The velocity are given by default depending on the type of connection, 360 for HV and 120 for normal
class Connection_type(Enum):
	HIGH_VELOCITY_RAIL = 1
	NORMAL_RAIL = 2


# A physical connection between two stations
class Rail_connection:

	# For the id
	id_iter = itertools.count()

	def __init__(self, station_a, station_b, rail_connection_type, additional_runtime_percent, max_speed_usable: int = 1/2):
		# each railway section has an id
		self.id = next(Rail_connection.id_iter)
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