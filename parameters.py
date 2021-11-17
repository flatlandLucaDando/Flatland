
from argparse import ArgumentParser, Namespace


parser = ArgumentParser()
parser.add_argument("-n", "--n_episodes", help="number of episodes to run", default=2500, type=int)
parser.add_argument("-t", "--training_env_config", help="training config id (eg 0 for Test_0)", default=0, type=int)
parser.add_argument("-e", "--evaluation_env_config", help="evaluation config id (eg 0 for Test_0)", default=0, type=int)
parser.add_argument("--n_evaluation_episodes", help="number of evaluation episodes", default=25, type=int)
parser.add_argument("--checkpoint_interval", help="checkpoint interval", default=100, type=int)
parser.add_argument("--eps_start", help="max exploration", default=1.0, type=float)
parser.add_argument("--eps_end", help="min exploration", default=0.01, type=float)
parser.add_argument("--eps_decay", help="exploration decay", default=0.99, type=float)
parser.add_argument("--buffer_size", help="replay buffer size", default=int(1e5), type=int)
parser.add_argument("--buffer_min_size", help="min buffer size to start training", default=0, type=int)
parser.add_argument("--restore_replay_buffer", help="replay buffer to restore", default="", type=str)
parser.add_argument("--save_replay_buffer", help="save replay buffer at each evaluation interval", default=False, type=bool)
parser.add_argument("--batch_size", help="minibatch size", default=128, type=int)
parser.add_argument("--gamma", help="discount factor", default=0.99, type=float)
parser.add_argument("--tau", help="soft update of target parameters", default=1e-3, type=float)
parser.add_argument("--learning_rate", help="learning rate", default=0.5e-4, type=float)
parser.add_argument("--hidden_size", help="hidden size (2 fc layers)", default=128, type=int)
parser.add_argument("--update_every", help="how often to update the network", default=8, type=int)
parser.add_argument("--use_gpu", help="use GPU if available", default=True, type=bool)
parser.add_argument("--num_threads", help="number of threads PyTorch can use", default=1, type=int)
parser.add_argument("--render", help="render 1 episode in 100", default=False, type=bool)
parser.add_argument("--track", help="whether to track using wandb", default=False, type=bool)
training_params = parser.parse_args()


obs_params = {
    "observation_tree_depth": 5,
    "observation_radius": 10,
    "observation_max_path_depth": 30
}