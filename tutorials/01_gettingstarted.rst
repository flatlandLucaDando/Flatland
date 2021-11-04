Getting Started Tutorial
========================

Overview
--------

Following are three short tutorials to help new users get acquainted with how
to create RailEnvs, how to train simple DQN agents on them, and how to customize
them.

To use flatland in a project:

.. code-block:: python

    import flatland


Simple Example 1 : Basic Usage
------------------------------
The basic usage of RailEnv environments consists in creating a RailEnv object
endowed with a rail generator, that generates new rail networks on each reset,
and an observation generator object, that is supplied with environment-specific
information at each time step and provides a suitable observation vector to the
agents. After the RailEnv environment is created, one need to call reset() on the
environment in order to fully initialize the environment

The simplest rail generators are envs.rail_generators.rail_from_manual_specifications_generator
and envs.rail_generators.random_rail_generator.

The first one accepts a list of lists whose each element is a 2-tuple, whose
entries represent the 'cell_type' (see core.transitions.RailEnvTransitions) and
the desired clockwise rotation of the cell contents (0, 90, 180 or 270 degrees).
The file should be added in the EXAMPLES repository, an example is given:

.. code-block:: python

    railway_example = [[(0,0)]*12,             				                                                            # 0
		       [(0,0)]*12,                                                                                                  # 1
		       [(0,0)] + [(7,270)] + [(1,90)]*2 + [(8,90)] + [(0,0)]*2 + [(8,0)] + [(1,90)]*2 + [(7,90)] + [(0,0)],         # 2
		       [(0,0)] + [(7,270)] + [(1,90)]*2 + [(10,270)] + [(1,90)]*2 + [(2,90)] + [(1,90)]*2 + [(7,90)] + [(0,0)],     # 3
		       [(0,0)]*12,                                                              				    # 4
		       [(0,0)]*12]                                                       					    # 5

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

Now is important to work in the configuration.py file.
Is important to specify the stations, stations represent the physical stations, with a capacity of different rails, a position, the minimum wait time to charge the passengers depending on the type of train and the importance of the station:

.. code-block:: python

    quarto_station = Station('Quarto', position = (3, 2), capacity = 2, min_wait_time = [2, 2, 1], 
	additional_wait_percent = [0.5, 1, 1.5], importance = 80, railway_topology = rail)
    quinto_station = Station('Quinto', position = (3, 9), capacity = 2, min_wait_time = [2, 2, 1], 
	additional_wait_percent = [0.5, 1, 1.5], importance = 80, railway_topology = rail)    

The connection between the different stations, with multiple types of connection (e.g. high velocity, regional...) and a maximum speed possible:

.. code-block:: python

	connection_quarto_quinto = Rail_connection(station_a = quarto_station, 
		station_b = quinto_station, rail_connection_type = Connection_type.NORMAL_RAIL,
		max_speed_usable = [0.9, 0.6, 0.3], additional_runtime_percent = [0.1, 0.1, 0.1])
		
The lines of the railway, with different types (e.g. regional or high velocity) and the stations to stop:

.. code-block:: python

	genova_urbana = Line(type_line = Connection_type.NORMAL_RAIL, 
		stations = (quarto_station, quinto_station), stops = (1, 1))
		
The train runs based on the starting time:

.. code-block:: python

	train_run_0 = Train_run(genova_urbana, starting_time = 3, from_depot = True)
	train_run_1 = Train_run(genova_urbana, starting_time = 10, from_depot = True, inverse_train_direction = True)
	train_run_2 = Train_run(genova_urbana, starting_time = 40, inverse_train_direction = True)
	
And the convoys, with different types (e.g. high velocity, regional..):

.. code-block:: python

	R1079_convoy = Convoy( Type_of_convoy.INTERCITY)
	R1078_convoy = Convoy( Type_of_convoy.INTERCITY)

Now we can add the train runs to the convoys and then generate the plan to do:

.. code-block:: python

	R1079_convoy.add_train_run(train_run_0)
	R1079_convoy.add_train_run(train_run_2)
	R1078_convoy.add_train_run(train_run_1)

	# Generating the PLAN 
	# The timetable is composed by (station positions, time at which reach the stations, maximum train velocity)
	timetable_example = calculate_timetable(convoys, rail)

Then in main we can calculate different things

.. code-block:: python
	
	# Specification to create the environment
	specs = railway_example
	widht = len(specs[0])
	height = len(specs)
	num_of_agents = len(timetable)
	
	# Generating the railway topology, with stations
	# Arguments of the generator (specs of the railway, position of stations, timetable)
	rail_custom = rail_custom_generator(specs, stations_position, timetable)

	transition_map_example, agent_hints = rail_custom(widht, height, num_of_agents)

	control_timetable(timetable,transition_map_example)

	# We can now initiate the schedule generator with the given speed profiles
	schedule_generator_custom = custom_schedule_generator(timetable = timetable)
	
	# Action scheduled for each agent in the environment, these action are scheduled in case of opereting in deterministic case
	actions_scheduled = action_to_do(timetable, transition_map_example)
	
	TreeObservation = GlobalObsForRailEnv()

	env = RailEnv(  width= widht,
					height= height,
					rail_generator = rail_custom,
					line_generator=schedule_generator_custom,
					number_of_agents= num_of_agents,
					obs_builder_object=TreeObservation,
					remove_agents_at_target=True,
					record_steps=True
					)


	env.reset()
	
	
Environments can be rendered using the utils.rendertools utilities, for example:

.. code-block:: python

    env_renderer = RenderTool(env,
    				screen_height=1080*2,
				screen_width=1080*2))
    env_renderer.render_env(show=True)


Finally, the environment can be run by supplying the environment step function
with a dictionary of actions whose keys are agents' handles (returned by
env.get_agent_handles() ) and the corresponding values the selected actions.
For example, for a 2-agents environment:

.. code-block:: python

	# This thing is importand for the RL part, initialize the agent with (state, action) dimension
	# Initialize the agent with the parameters corresponding to the environment and observation_builder
	controller = RandomAgent(218, env.action_space[0])

	# Lets try to enter with all of these agents at the same time
	action_dict = dict()

	# Now that you have seen these novel concepts that were introduced you will realize that agents don't need to take
	# an action at every time step as it will only change the outcome when actions are chosen at cell entry.
	# Therefore the environment provides information about what agents need to provide an action in the next step.
	# You can access this in the following way.

	# Chose an action for each agent
	for a in range(env.get_num_agents()):
		action = controller.act(0)
		action_dict.update({a: action})
	# Do the environment step

	observations ,rewards, dones ,information = env.step(action_dict)

	print("\n The following agents can register an action:")
	print("========================================")
	for info in information['action_required']:
		print("Agent {} needs to submit an action.".format(info))

	# We recommend that you monitor the malfunction data and the action required in order to optimize your training
	# and controlling code.

	# Let us now look at an episode playing out 

	print("\nStart episode...")

	# Reset the rendering system
	env_renderer.reset()

	# Here you can also further enhance the provided observation by means of normalization
	# See training navigation example in the baseline repository

	score = 0
	# Run episode
	frame_step = 0

	# How many episodes
	n_trials = 3

	os.makedirs("output/frames", exist_ok=True)

	for trials in range(1, n_trials + 1):

		# Reset environment and get initial observations for all agents
		obs, info = env.reset()
		for idx in range(env.get_num_agents()):
			tmp_agent = env.agents[idx]
			tmp_agent.speed_counter.speed = 1 / (idx + 1)
		env_renderer.reset()
		# Here you can also further enhance the provided observation by means of normalization
		# See training navigation example in the baseline repository

		score = 0
		# Run episode (one day long, 1 step is 1 minute)
		for step in range(1440):

			env_renderer.gl.save_image("output/frames/flatland_frame_step_{:04d}.bmp".format(step))

		# Here define the actions to do

			# Chose an action for each agent in the environment
			# If not interruption, the actions to do are stored in a matrix
			#       - each row of the matrix is a train
			#       - each column represent the action the train has to do at each time instant

			print('=================================')
			print('Elapsed time is', step, 'minutes')   # DEBUG

			# DEBUG print the velocities
			for agent in env.agents:
				i_agent = agent.handle
				print('Velocity of the', i_agent, 'agent is', agent.speed_counter.speed)

			for a in range(env.get_num_agents()):
				if step >= timetable[a][1][0]:
					if not interruption and (step - timetable[a][1][0]) < len(actions_scheduled[a]):
						action = actions_scheduled[a][step - timetable[a][1][0]]
					# choose random from all the possible actions
					else:
						action = np.random.choice([RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT, RailEnvActions.MOVE_LEFT, 
							RailEnvActions.STOP_MOVING, RailEnvActions.REVERSE])
					action_dict.update({a: action})
			# Environment step which returns the observations for all agents, their corresponding
			# reward and whether their are done
			next_obs, all_rewards, done, _ = env.step(action_dict)

			'''
			print('================================')
			print(env.agents[0])
			print(env.agents[1])
			print(timetable)
			print('================================')
			'''

			env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

			# Update replay buffer and train agent
			for a in range(env.get_num_agents()):
				controller.step((obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
				score += all_rewards[a]
			obs = next_obs.copy()
			if done['__all__']:
				break
		print('Episode Nr. {}\t Score = {}'.format(trials, score))

	where 'obs', 'all_rewards', and 'done' are also dictionary indexed by the agents'
	handles, whose values correspond to the relevant observations, rewards and terminal
	status for each agent. Further, the 'dones' dictionary returns an extra key
	'__all__' that is set to True after all agents have reached their goals.


	In the specific case a TreeObsForRailEnv observation builder is used, it is
	possible to print a representation of the returned observations with the
	following code. Also, tree observation data is displayed by RenderTool by default.

	.. code-block:: python

	    for i in range(env.get_num_agents()):
		env.obs_builder.util_print_obs_subtree(
			tree=obs[i],
			)

The complete code for this part of the Getting Started guide can be found in

*`main.py <https://github.com/LucaFronda/Flatland_Luca_Fronda/blob/flatland_v_3_deterministic/main.py>`_
*`configuration.py <https://github.com/LucaFronda/Flatland_Luca_Fronda/blob/flatland_v_3_deterministic/configuration.py>`_
*`examples/new_example_1.py <https://github.com/LucaFronda/Flatland_Luca_Fronda/blob/flatland_v_3_deterministic/examples/new_example_1.py>`_

Part 2 : Training a Simple an Agent on Flatland
---------------------------------------------------------

This is a brief tutorial on how to train an agent on Flatland.
Here we use a simple random agent to illustrate the process on how to interact with the environment.
The corresponding code can be found in examples/training_example.py and in the baselines repository
you find a tutorial to train a `DQN <https://arxiv.org/abs/1312.5602>`_ agent to solve the navigation task.

We start by importing the necessary Flatland libraries

.. code-block:: python

    from flatland.envs.rail_generators import complex_rail_generator
    from flatland.envs.schedule_generators import complex_schedule_generator
    from flatland.envs.rail_env import RailEnv

The complex_rail_generator is used in order to guarantee feasible railway network configurations for training.
Next we configure the difficulty of our task by modifying the complex_rail_generator parameters.

.. code-block:: python

    env = RailEnv(  width=15,
                    height=15,
                    rail_generator=complex_rail_generator(
                                        nr_start_goal=10,
                                        nr_extra=10,
                                        min_dist=10,
                                        max_dist=99999,
                                        seed=1),
                    number_of_agents=5)
    env.reset()

The difficulty of a railway network depends on the dimensions (`width` x `height`) and the number of agents in the network.
By varying the number of start and goal connections (nr_start_goal) and the number of extra railway elements added (nr_extra)
the number of alternative paths of each agents can be modified. The more possible paths an agent has to reach its target the easier the task becomes.
Here we don't specify any observation builder but rather use the standard tree observation. If you would like to use a custom obervation please follow
the instructions in the next tutorial.
Feel free to vary these parameters to see how your own agent holds up on different setting. The evalutation set of railway configurations will
cover the whole spectrum from easy to complex tasks.

Once we are set with the environment we can load our preferred agent from either RLlib or any other ressource. Here we use a random agent to illustrate the code.

.. code-block:: python

    agent = RandomAgent(state_size, action_size)

We start every trial by resetting the environment

.. code-block:: python

    obs, info = env.reset()

Which provides the initial observation for all agents (obs = array of all observations).
In order for the environment to step forward in time we need a dictionar of actions for all active agents.

.. code-block:: python

        for handle in range(env.get_num_agents()):
            action = agent.act(obs[handle])
            action_dict.update({handle: action})

This dictionary is then passed to the environment which checks the validity of all actions and update the environment state.

.. code-block:: python

    next_obs, all_rewards, done, _ = env.step(action_dict)

The environment returns an array of new observations, reward dictionary for all agents as well as a flag for which agents are done.
This information can be used to update the policy of your agent and if done['__all__'] == True the episode terminates.

The full source code of this example can be found in `examples/training_example.py <https://gitlab.aicrowd.com/flatland/flatland/blob/master/examples/training_example.py>`_.
