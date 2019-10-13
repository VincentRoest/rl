import argparse
import matplotlib.pyplot as plt
import atexit

from pprint import pprint

from data_utils.cartpole_env import CartpoleEnv 
from data_utils.pong_env import PongEnv

from model import DQN
from train import train_model

import torch
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='RL training')
parser.add_argument("--use_env", type=str, default='cartpole', help="Specify environments"),
parser.add_argument("--batch_size", type=int, default=32),
parser.add_argument("--gamma", type=float, default=0.9),
parser.add_argument("--eps_start", type=float, default=0.9, help="Starting exploration rate")
parser.add_argument("--eps_end", type=float, default=0.05, help="Ending exploration rate")
parser.add_argument("--eps_decay", type=int, default=200, help="How many episodes to decay exploration rate")
parser.add_argument("--target_update", type=int, default=-1, help="Use Target Policy updates (default NO)")
parser.add_argument("--clip_rewards", type=bool, default=False, help="Do clipping of rewards (default false)")
# todo
parser.add_argument("--double_q", type=bool, default=False, help="Do Double Q Learning (default off)")

parser.add_argument("--num_episodes", type=int, default=1200, help="Number of episodes to train for")
parser.add_argument("--replay_size", type=int, default=10000, help="Size of replay memory")

parser.add_argument("--save_path", type=str, default='saved_checkpoints/MODEL_CHECKPOINT.pth', help="Path to save model checkpoints to")
parser.add_argument("--load_path", type=str, default=None, help="Path to load model checkpoint from")
parser.add_argument("--save_every", type=int, default=10, help="Amount of episodes after which model checkpoint is saved")

parser.add_argument("--show_example", action='store_true', help="Show example of environment before training")
parser.add_argument("--show_screen", action='store_true', help="Show screen during training")
parser.add_argument("--show_progress", action='store_true', help="Show plotted training progress")

parser.add_argument("--seed", type=int, default=42, help="Seed used for randomness")

if __name__ == '__main__':
  params, _ = parser.parse_known_args()

  print('Parameters:')
  pprint(params.__dict__)

  plt.ion()

  if (params.use_env == 'cartpole'):
    env = CartpoleEnv()
  if (params.use_env == 'pong'):
    env = PongEnv()
  atexit.register(env.env.close)

  if params.show_example == True:
    env.show_example()

  env.env.reset()
  init_screen = env.get_screen()
  _, _, screen_height, screen_width = init_screen.shape

  # Get number of actions from gym action space
  n_actions = env.env.action_space.n

  policy_net = DQN(screen_height, screen_width, n_actions).to(device)
  target_net = DQN(screen_height, screen_width, n_actions).to(device)

  def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
  print('parameter count: {}'.format(count_parameters(policy_net)))

  optimizer = optim.Adam(policy_net.parameters())

  episode_durations, rewards = train_model(env, optimizer, policy_net, target_net, params)

  print('episode durations: {}'.format(episode_durations))
  print('Complete')
  plt.ioff()
  plt.show()


