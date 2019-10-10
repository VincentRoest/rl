import argparse
import matplotlib.pyplot as plt

from data_utils.cartpole_env import CartpoleEnv 
from data_utils.pong_env import PongEnv

from model import DQN
from memory import ReplayMemory
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

parser.add_argument("--num_episodes", type=int, default=500, help="Number of episodes to train for")


if __name__ == '__main__':
  params, _ = parser.parse_known_args()

  plt.ion()

  if (params.use_env == 'cartpole'):
    env = CartpoleEnv()
    env.show_example()
    env.env.close()
  if (params.use_env == 'pong'):
    env = PongEnv()
    env.show_example()
    env.env.close()

  init_screen = env.get_screen()
  _, _, screen_height, screen_width = init_screen.shape

  # Get number of actions from gym action space
  n_actions = env.env.action_space.n

  policy_net = DQN(screen_height, screen_width, n_actions).to(device)

  target_net = DQN(screen_height, screen_width, n_actions).to(device)

  def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
  print('parameter count: {}'.format(count_parameters(policy_net)))

  target_net.load_state_dict(policy_net.state_dict())
  target_net.eval()

  optimizer = optim.Adam(policy_net.parameters())
  memory = ReplayMemory(1000)

  episode_durations, rewards = train_model(env, optimizer, policy_net, target_net, memory, params)

  print('episode durations: {}'.format(episode_durations))
  print('Complete')

  
  env.env.close()
  plt.ioff()
  plt.show()



