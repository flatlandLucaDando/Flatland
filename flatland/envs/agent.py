from flatland.envs.rail_env import RailEnvActions

# Import your own Agent or use RLlib to train agents on Flatland
# As an example we use a random agent instead
class RandomAgent:

	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size


	# HERE DEFINE THE ACTIONS TO DO IN CASE THE AGENT IS NOT IN THE DETERMINISTIC PART 
	# For now the agents can only move forward (for DEBUG)
	def act(self, state):
		return RailEnvActions.MOVE_FORWARD

	def step(self, memories):
		"""

		Step function to improve agent by adjusting policy given the observations

		:param memories: SARS Tuple to be
		:return:
		"""

		return

	def save(self, filename):
		# Store the current policy
		return

	def load(self, filename):
		# Load a policy
		return