
from argparse import ArgumentParser, Namespace


parser = ArgumentParser()
parser.add_argument("-n", "--n_episodes", help="number of episodes to run", default=5000, type=int)
parser.add_argument("--n_agent_fixed", help="hold the number of agent fixed", action='store_true')
parser.add_argument("--n_agent_iterate", help="iterate the number of agent fixed", action='store_true')
parser.add_argument("-t", "--training_env_config", help="training config id (eg 0 for Test_0)", default=1, type=int)
parser.add_argument("-e", "--evaluation_env_config", help="evaluation config id (eg 0 for Test_0)", default=3, type=int)
parser.add_argument("--n_evaluation_episodes", help="number of evaluation episodes", default=10, type=int)
parser.add_argument("--checkpoint_interval", help="checkpoint interval", default=200, type=int)
parser.add_argument("--eps_start", help="max exploration", default=1.0, type=float)
parser.add_argument("--eps_end", help="min exploration", default=0.01, type=float)
parser.add_argument("--eps_decay", help="exploration decay", default=0.99975, type=float)
parser.add_argument("--buffer_size", help="replay buffer size", default=int(32_000), type=int)
parser.add_argument("--buffer_min_size", help="min buffer size to start training", default=0,
                    type=int)
parser.add_argument("--restore_replay_buffer", help="replay buffer to restore", default="",
                    type=str)
parser.add_argument("--save_replay_buffer",
                    help="save replay buffer at each evaluation interval", default=False,
                    type=bool)
parser.add_argument("--batch_size", help="minibatch size", default=64, type=int) #1024
parser.add_argument("--gamma", help="discount factor", default=0.99, type=float)
parser.add_argument("--tau", help="soft update of target parameters", default=0.5e-3,
                    type=float)
parser.add_argument("--learning_rate", help="learning rate", default=1e-4, type=float)    # 10-3 0.5e-4
parser.add_argument("--hidden_size", help="hidden size (2 fc layers)", default=128, type=int)
parser.add_argument("--update_every", help="how often to update the network", default=8, 
                    type=int)   #200
parser.add_argument("--use_gpu", help="use GPU if available", default=True, type=bool)
parser.add_argument("--num_threads", help="number of threads PyTorch can use", default=16,        # Prova ad alzare di tanto
                    type=int)

parser.add_argument("--load_policy", help="policy filename (reference) to load", default="",
                    type=str)
parser.add_argument("--use_observation",
                    help="observation name [TreeObs, FastTreeObs, FlatlandObs]",
                    default='FlatlandObs')
parser.add_argument("--max_depth", help="max depth", default=2, type=int)
parser.add_argument("--K_epoch", help="K_epoch", default=10, type=int)
parser.add_argument("--skip_unfinished_agent", default=9999.0, type=float)
parser.add_argument("--render", help="render while training", action='store_true')
parser.add_argument("--eval_render", help="render evaluation", action='store_true')
parser.add_argument("--render_deadlocked", default=None, type=str)
parser.add_argument("--policy", help="policy name [DDDQN, PPO, PPORCS, DecisionPointAgent, DecisionPointAgent_DDDQN,"
                    "DeadLockAvoidance, DeadLockAvoidanceWithDecisionAgent, MultiDecisionAgent, MultiPolicy]", default="DDDQN")
parser.add_argument("--action_size", help="define the action size [reduced,full]", default="reduced", type=str)
parser.add_argument('-f')


training_params = parser.parse_args()

obs_params = {
    "observation_tree_depth": 2,
    "observation_radius": 10,
    "observation_max_path_depth": 20
}