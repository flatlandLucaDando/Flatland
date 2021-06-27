from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_manual_specifications_generator
from flatland.utils.rendertools import RenderTool

# Example generate a rail given a manual specification,
# a map of tuples (cell_type, rotation)
#            1      2       3         4       5     6       7      8       9     10     11     12     13     14     15     16    17     18      19     20
specs = [[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0)],
#            1      2        3        4        5         6        7       8         9      10        11       12        13     14         15      16      17        18       19       20
         [(0, 0), (8, 0), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (0, 0)],
         [(7, 270), (2, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (1, 90), (0, 0)],
         [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]]

env = RailEnv(width=100, height=4, rail_generator=rail_from_manual_specifications_generator(specs), number_of_agents=1)

env.reset()

env_renderer = RenderTool(env)
env_renderer.render_env(show=True, show_predictions=True, show_observations=False)

# uncomment to keep the renderer open
input("Press Enter to continue...")


